import yaml
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.my_deepfashion_dataset import DeepFashionDataset
from torch.utils.data import DataLoader
from models.my_unet_cond_base import Unet
from models.my_vqvae import VQVAE
from scheduler.my_linear_noise_scheduler import LinearNoiseScheduler
from utils.my_text_utils import *
from utils.my_config_utils import *
from utils.my_diffusion_utils import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs_pose_inpainting_spec_200/sd')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
	"""
    Read the config file
    args.config_path: "../config/my_celebhq_text_image_cond.yaml"
    config: {
        dataset_params: {...},
        diffusion_params: {...},
        ldm_params: {...},
        autoencoder_params: {...},
        train_params: {...}
    }
    dataset_config: {
        im_path: 'data/CelebAMask-HQ',
        im_channels : 3,
        im_size : 256,
        name: 'celebhq'
    }
    diffusion_config: {
        num_timesteps : 1000,
        beta_start: 0.00085,
        beta_end: 0.012
    }
    diffusion_model_config: {
        down_channels: [ 256, 384, 512, 768 ],
        mid_channels: [ 768, 512 ],
        down_sample: [ True, True, True],
        attn_down : [True, True, True],
        time_emb_dim: 512,
        norm_channels: 32,
        num_heads: 16,
        conv_out_channels : 128,
        num_down_layers : 2,
        num_mid_layers : 2,
        num_up_layers : 2,
        condition_config: {...}
    }
    autoencoder_model_config: {
        z_channels: 3,
        codebook_size : 8192,
        down_channels : [64, 128, 256, 256],
        mid_channels : [256, 256],
        down_sample : [True, True, True],
        attn_down : [False, False, False],
        norm_channels: 32,
        num_heads: 4,
        num_down_layers : 2,
        num_mid_layers : 2,
        num_up_layers : 2
    }
    train_config: {
        seed : 1111,
        task_name: 'celebhq',
        ldm_batch_size: 16,
        autoencoder_batch_size: 4,
        disc_start: 15000,
        disc_weight: 0.5,
        codebook_weight: 1,
        commitment_beta: 0.2,
        perceptual_weight: 1,
        kl_weight: 0.000005,
        ldm_epochs: 100,
        autoencoder_epochs: 100,
        num_samples: 1,
        num_grid_rows: 1,
        ldm_lr: 0.000005,
        autoencoder_lr: 0.00001,
        autoencoder_acc_steps: 4,
        autoencoder_img_save_steps: 64,
        save_latents : False,
        cf_guidance_scale : 1.0,
        vae_latent_dir_name: 'vae_latents',
        vqvae_latent_dir_name: 'vqvae_latents',
        ldm_ckpt_name: 'ddpm_ckpt_text_image_cond_clip.pth',
        vqvae_autoencoder_ckpt_name: 'vqvae_autoencoder_ckpt.pth',
        vae_autoencoder_ckpt_name: 'vae_autoencoder_ckpt.pth',
        vqvae_discriminator_ckpt_name: 'vqvae_discriminator_ckpt.pth',
        vae_discriminator_ckpt_name: 'vae_discriminator_ckpt.pth'
    }
    """
	with open(args.config_path, 'r') as file:
		try:
			config = yaml.safe_load(file)
		except yaml.YAMLError as exc:
			print(exc)
	# print(config)
	dataset_config = config['dataset_params']
	diffusion_config = config['diffusion_params']
	diffusion_model_config = config['ldm_params']
	autoencoder_model_config = config['autoencoder_params']
	train_config = config['train_params']

	'''
    create the noise scheduler
    diffusion_config['num_timesteps']: 1000
    diffusion_config['beta_start']: 0.00085
    diffusion_config['beta_end']: 0.012
    '''
	scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
	                                 beta_start=diffusion_config['beta_start'],
	                                 beta_end=diffusion_config['beta_end'])

	# instantiate condition related components
	text_tokenizer = None
	text_model = None
	empty_text_embed = None
	condition_types = []

	"""
    condition_config: {
        condition_types: [ 'text', 'image' ]
        text_condition_config: {...}
        image_condition_config: {...}
    }
    """
	condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
	if condition_config is not None:
		assert 'condition_types' in condition_config, "condition type missing in conditioning config"

		# condition_types: ['text', 'image']
		condition_types = condition_config['condition_types']
		if 'text' in condition_types:
			# ensure text config is correct
			validate_text_config(condition_config)
			with torch.no_grad():
				"""
                load tokenizer and text model based on config
                condition_config['text_condition_config']['text_embed_model']: "clip"
                text_tokenizer: CLIPTokenizer()
                text_model: CLIPTextModel() is on GPU
                """
				text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
				                                                     ['text_embed_model'], device=device)
				"""
                get empty text representation
                empty_text_embed.shape: torch.Size([1, 77, 512]), 512 to 768  
                """
				empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
				print(empty_text_embed.size())
	# im_dataset_cls: "deepfashion"
	im_dataset_cls = {
		'deepfashion': DeepFashionDataset
	}.get(dataset_config['name'])

	im_dataset = im_dataset_cls(split='train',
	                            im_path=f"../{dataset_config['im_path']}",
	                            im_size=dataset_config['im_size'],
	                            im_channels=dataset_config['im_channels'],
	                            use_latents=False,
	                            latent_path=f"../{train_config['task_name']}/vqgan_500_epochs/{train_config['vqvae_latent_dir_name']}",
	                            condition_config=condition_config)

	data_loader = DataLoader(im_dataset,
	                         batch_size=train_config['ldm_batch_size'],
	                         shuffle=True)
	"""
    instantiate the unet model
    autoencoder_model_config['z_channels']: 3
    """
	model = Unet(im_channels=autoencoder_model_config['z_channels'],
	             model_config=diffusion_model_config).to(device)
	# set model to training mode
	model.train()
	model.load_state_dict(
		torch.load(f"../{train_config['task_name']}/vqgan_500_epochs/{train_config['ldm_ckpt_name']}",
		           map_location=device))
	vae = None
	# Load VAE ONLY if latents are not to be saved or some are missing
	if not im_dataset.use_latents:
		print('Loading vqvae model as latents not present')
		vae = VQVAE(im_channels=dataset_config['im_channels'],
		            model_config=autoencoder_model_config).to(device)
		# set model to evaluation mode
		vae.eval()
		# Load vae if found
		if os.path.exists(
				f"../{train_config['task_name']}/vqgan_500_epochs/{train_config['vqvae_autoencoder_ckpt_name']}"):
			print('Loaded vae checkpoint')
			vae.load_state_dict(torch.load(
				f"../{train_config['task_name']}/vqgan_500_epochs/{train_config['vqvae_autoencoder_ckpt_name']}",
				map_location=device))

		else:
			raise Exception('VAE checkpoint not found and use_latents was disabled')

	# pecify training parameters
	num_epochs = train_config['ldm_epochs']
	optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
	criterion = torch.nn.MSELoss()

	# Load vae and freeze parameters ONLY if latents already not saved
	if not im_dataset.use_latents:
		assert vae is not None
		# set all parameters in vae do not require gradient
		for param in vae.parameters():
			param.requires_grad = False

	# Run training
	for epoch_idx in range(num_epochs):
		losses = []
		for data in tqdm(data_loader):
			cond_input = None
			if condition_config is not None:
				"""
                shape of im: torch.Size([16, 3, 256, 256])
                cond_input: {"text": ..., "image": ...}
                shape of cond_input["image"]: torch.Size([16, 18, 512, 512])
                length of cond_input["text"]: 16
                """
				im, cond_input = data
			else:
				im = data
			optimizer.zero_grad()
			im = im.float().to(device)
			if not im_dataset.use_latents:
				with torch.no_grad():
					# torch.Size([16, 3, 32, 32])
					im, _ = vae.encode(im)

			# Handling Conditional Input
			if 'text' in condition_types:
				with torch.no_grad():
					assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
					validate_text_config(condition_config)
					# torch.Size([16, 77, 512])
					text_condition = get_text_representation(cond_input['text'],
					                                         text_tokenizer,
					                                         text_model,
					                                         device)
					# text_drop_prob: 0.1
					text_drop_prob = get_config_value(condition_config['text_condition_config'], 'cond_drop_prob', 0.)
					# torch.Size([16, 77, 512])
					text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
					# torch.Size([16, 77, 512])
					cond_input['text'] = text_condition
			if 'image' in condition_types:
				assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
				validate_image_config(condition_config)
				# torch.Size([16, 18, 512, 512])
				cond_input_image = cond_input['image'].to(device)
				# 0.1
				im_drop_prob = get_config_value(condition_config['image_condition_config'], 'cond_drop_prob', 0.)
				# torch.Size([16, 18, 512, 512])
				cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
				if 'pose' in condition_types:
					# torch.Size([16, 18, 512, 512])
					cond_input_image = cond_input['pose'].to(device)
					# 0.1
					im_drop_prob = get_config_value(condition_config['pose_condition_config'], 'cond_drop_prob', 0.)
					# torch.Size([16, 18, 512, 512])
					cond_input['pose'] = drop_image_condition(cond_input_image, im, im_drop_prob)
				if 'binary_mask' in condition_types:
					cond_input['binary_mask'] = cond_input['binary_mask'].to(device)

					cond_input['outer_latent'] = im * (1 - (
						torch.nn.functional.interpolate(cond_input['binary_mask'].detach(), size=(64, 32))))

			# if 'class' in condition_types:
			#     assert 'class' in cond_input, 'Conditioning Type Class but no class conditioning input present'
			#     validate_class_config(condition_config)
			#     class_condition = torch.nn.functional.one_hot(
			#         cond_input['class'],
			#         condition_config['class_condition_config']['num_classes']).to(device)
			#     class_drop_prob = get_config_value(condition_config['class_condition_config'],
			#                                        'cond_drop_prob', 0.)
			#     # Drop condition
			#     cond_input['class'] = drop_class_condition(class_condition, class_drop_prob, im)

			# sample random noise
			# shape: torch.Size([16, 3, 32, 32])
			noise = torch.randn_like(im).to(device)
			# sample timestep from 0 ~ 1000
			# shape: torch.Size([16])
			t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
			# add noise to images according to timestep
			# shape: torch.Size([16, 3, 32, 32])
			noisy_im = scheduler.add_noise(im, noise, t)
			"""
            shape of cond_input["text"]: torch.Size([16, 77, 512])
            shape of cond_input["image"]: torch.Size([16, 18, 512, 512])
            noise_pred: 
            """
			# torch.Size([16, 3, 32, 32])
			noise_pred = model(noisy_im, t, cond_input=cond_input)
			loss = criterion(noise_pred, noise)
			losses.append(loss.item())
			loss.backward()
			optimizer.step()
		writer.add_scalar("training loss", np.mean(losses), epoch_idx + 1)
		print('Finished epoch:{} | Loss : {:.4f}'.format(
			epoch_idx + 1,
			np.mean(losses)))
		torch.save(model.state_dict(),
		           f"../{train_config['task_name']}/vqgan_500_epochs/{train_config['ldm_ckpt_name']}")
	print('Done Training ...')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments for ddpm training')
	parser.add_argument('--config', dest='config_path',
	                    default='../config/my_deepfashion_text_image_inpainting_cond.yaml', type=str)
	args = parser.parse_args()
	train(args)
