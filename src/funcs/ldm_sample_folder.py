import numpy as np
import sys
import cv2
import time
import torch
import random
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from src.models.my_unet_cond_base import Unet
from src.models.my_vqvae import VQVAE
from src.scheduler.my_linear_noise_scheduler import LinearNoiseScheduler
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel
from src.utils.my_config_utils import *
from src.utils.my_text_utils import *
from src.dataset.my_deepfashion_dataset import DeepFashionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn.functional as F


def getModelSize(model):
	param_size = 0
	param_sum = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
		param_sum += param.nelement()
	buffer_size = 0
	buffer_sum = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
		buffer_sum += buffer.nelement()
	all_size = (param_size + buffer_size) / 1024 / 1024
	print('Model size: {:.3f}MB'.format(all_size))
	print(f'Sum of parameters: {param_sum}')
	return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def low_pass_filter(image_tensor, N):
	# Downsample the image by a factor of N
	downsampled = F.interpolate(image_tensor, scale_factor=1 / N, mode='bilinear', align_corners=False)

	# Upsample the image back to the original size
	upsampled = F.interpolate(downsampled, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)

	return upsampled

def sample(model, scheduler, train_config, diffusion_model_config,
		  autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model):
	condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
	validate_image_config(condition_config)
	dataset = DeepFashionDataset(split='train',
								im_path=f"src/{dataset_config['im_path']}",
								im_size=dataset_config['im_size'],
								im_channels=dataset_config['im_channels'],
								use_latents=True,
								latent_path=os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name']),
								condition_config=condition_config,
								is_train=False)
	im_size_h = dataset_config['im_size'][0] // 2 ** sum(autoencoder_model_config['down_sample'])
	im_size_w = dataset_config['im_size'][1] // 2 ** sum(autoencoder_model_config['down_sample'])
	for count in range(len(dataset)):
		# sample random noise latent
		xt = torch.randn((1, autoencoder_model_config['z_channels'], im_size_h, im_size_w)).to(device)
		noise_temp = xt.clone()
		# create conditional input
		mask_idx = count
		pose_idx = count
		text_idx = count
		text_prompt = dataset.get_text(text_idx, text_index=0)
		text_prompt = [text_prompt]
		empty_prompt = ['']
		text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
		empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
		assert empty_text_embed.shape == text_prompt_embed.shape
		mask = dataset.get_mask(mask_idx).unsqueeze(0).to(device)
		pose = dataset.get_pose(pose_idx).unsqueeze(0).to(device)
		uncond_input = {
	        'text': empty_text_embed,
	        'image': torch.zeros_like(mask),
	        'pose': torch.zeros_like(pose),
		}
		cond_input = {
	        'text': text_prompt_embed,
	        'image': mask,
	        'pose': pose,
		}
		only_text_cond_input = {
	        'text': text_prompt_embed,
	        'image': torch.zeros_like(mask),
	        'pose': torch.zeros_like(pose),
		}

		only_mask_cond_input = {
			'text': empty_text_embed,
			'image': mask,
			'pose': torch.zeros_like(pose),
		}

		only_pose_cond_input = {
			'text': empty_text_embed,
			'image': mask,
			'pose': pose,
		}

		cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)

		# sampling Loop
		# for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
		ddim_step = 50
		eta = 0
		start_time = time.time()
		for i in tqdm(range(diffusion_config['num_timesteps']-1, -ddim_step, -ddim_step)):
			if i < 0:
				i = 0
			# Get prediction of noise
			t = (torch.ones((xt.shape[0],)) * i).long().to(device)
			noise_pred_cond = model(xt, t, cond_input)
			if cf_guidance_scale > 1:
				noise_pred_uncond = model(xt, t, uncond_input)
				noise_pred_only_text_cond = model(xt, t, only_text_cond_input)
				noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond) + cf_guidance_scale * (noise_pred_only_text_cond - noise_pred_uncond)


			else:
				noise_pred = noise_pred_cond

			# use scheduler to get x0 and xt-1
			xt, x0_pred = scheduler.sample_prev_timestep_ddim(xt, noise_pred, torch.as_tensor(i).to(device), ddim_step, eta)

			# save x0
			if i == 0:
				# decode only the final image to save time
				ims = vae.decode(xt)
			else:
				ims = x0_pred

			ims = torch.clamp(ims, -1., 1.).detach().cpu()
			ims = (ims + 1) / 2
			grid = make_grid(ims, nrow=10)
			img = torchvision.transforms.ToPILImage()(grid)

			if not os.path.exists(f"src/{train_config['task_name']}/cond_text_image_samples"):
				os.mkdir(f"src/{train_config['task_name']}/cond_text_image_samples")

			if i == 0:
				img.save(f"src/{train_config['task_name']}/cond_text_image_samples/result.png")
			else:
				pass
			img.close()
		end_time = time.time()
		print(f"Time taken: {end_time - start_time:.3f} seconds")

def infer(args, model, vae):
	# Read the config file #
	with open(args.config_path, 'r') as file:
		try:
			config = yaml.safe_load(file)
		except yaml.YAMLError as exc:
			print(exc)
	print(config)
	########################

	diffusion_config = config['diffusion_params']
	dataset_config = config['dataset_params']
	diffusion_model_config = config['ldm_params']
	autoencoder_model_config = config['autoencoder_params']
	train_config = config['train_params']

	# Create the noise scheduler
	scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
	                                 beta_start=diffusion_config['beta_start'],
	                                 beta_end=diffusion_config['beta_end'])

	##################Validate the config
	condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
	assert condition_config is not None, ("This sampling script is for image and text conditional "
                                          "but no conditioning config found")
	condition_types = get_config_value(condition_config, 'condition_types', [])
	assert 'text' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no text condition found in config")
	assert 'image' in condition_types, ("This sampling script is for image and text conditional "
                                        "but no image condition found in config")
	validate_text_config(condition_config)
	validate_image_config(condition_config)
	###############################################

	############# Load tokenizer and text model #################
	with torch.no_grad():
		# Load tokenizer and text model based on config
		# Also get empty text representation
		text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                             ['text_embed_model'], device=device)
	###############################################

	########## Load Unet #############
	if model == None:
		model = Unet(im_channels=autoencoder_model_config['z_channels'],
	                 model_config=diffusion_model_config).to(device)
		model.eval()

		# model_memory = torch.cuda.memory_allocated()
		# print(f"Model memory usage (LDM): {model_memory / (1024 ** 3):.2f} GB")
		# total_params = sum(p.numel() for p in model.parameters())
		# print(f"Total Parameters: {total_params}")

		if os.path.exists(f"src/{train_config['task_name']}/{train_config['ldm_ckpt_name']}"):
			print('Loaded unet checkpoint')
			model.load_state_dict(torch.load(f"src/{train_config['task_name']}/{train_config['ldm_ckpt_name']}",
	                                         map_location=device))
		else:
			raise Exception(
	            'Model checkpoint {} not found'.format(f"src/{train_config['task_name']}/{train_config['ldm_ckpt_name']}"))
		#####################################

	# Create output directories
	if not os.path.exists(train_config['task_name']):
		os.mkdir(train_config['task_name'])

	########## Load VQVAE #############
	if vae == None:
		vae = VQVAE(im_channels=dataset_config['im_channels'],
	                model_config=autoencoder_model_config).to(device)
		vae.eval()

		# model_memory = torch.cuda.memory_allocated()
		# print(f"Model memory usage (LDM+VQVAE): {model_memory / (1024 ** 3):.2f} GB")
		# total_params = sum(p.numel() for p in vae.parameters())
		# print(f"Total Parameters: {total_params}")

		# Load vae if found
		if os.path.exists(f"src/{train_config['task_name']}/{train_config['vqvae_autoencoder_ckpt_name']}"):
			print('Loaded vae checkpoint')
			vae.load_state_dict(torch.load(f"src/{train_config['task_name']}/{train_config['vqvae_autoencoder_ckpt_name']}",
	                                       map_location=device))
		else:
			raise Exception('VAE checkpoint {} not found'.format(
	            f"src/{train_config['task_name']}/{train_config['vqvae_autoencoder_ckpt_name']}"))


	with torch.no_grad():
		sample(model, scheduler, train_config, diffusion_model_config,
				autoencoder_model_config, diffusion_config, dataset_config, vae, text_tokenizer, text_model)

	getModelSize(vae)
	getModelSize(model)
	return model, vae


def run(model, vae):
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_path',
	                    default='src/config/my_deepfashion_text_image_cond.yaml', type=str)
	args = parser.parse_args()
	model, vae = infer(args, model, vae)
	return model, vae

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', dest='config_path',
                        default='src/config/my_deepfashion_text_image_cond.yaml', type=str)
	args = parser.parse_args()
	infer(args)
