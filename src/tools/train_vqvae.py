import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from models.my_vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from dataset.mnist_dataset import MnistDataset
from dataset.my_celeb_dataset import CelebDataset
from dataset.my_deepfashion_dataset import DeepFashionDataset

from torch.optim import Adam
from torchvision.utils import make_grid

# device: cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    """
    read the config file
    config: {
        'dataset_params': {…},
        'diffusion_params': {…},
        'ldm_params': {…},
        'autoencoder_params': {…},
        'train_params': {…}
    }
    dataset_config: {
        'im_path': 'data/CelebAMask-HQ',
        'im_channels': 3,
        'im_size': 256,
        'name': 'celebhq'
    }
    autoencoder_config: {
        'z_channels': 3,
        'codebook_size': 8192,
        'down_channels': [64, 128, 256, 256],
        'mid_channels': [256, 256],
        'down_sample': [True, True, True],
        'attn_down': [False, False, False],
        'norm_channels': 32,
        'num_heads': 4,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2
    }
    """
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    """
    set random seed
    train_config['seed']: 1111
    """
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    """
    build the vqvae model
    dataset_config['im_channels']: 3
    """
    model = VQVAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)
    print(f"model: {model}")

    # create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'deepfashion': DeepFashionDataset,
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=f"../{dataset_config['im_path']}",
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    # iterator = iter(data_loader)
    # print(next(iterator).shape)

    # create output directories
    if not os.path.exists(f"../{train_config['task_name']}"):
        os.mkdir(f"../{train_config['task_name']}")

    # num_epochs: 100
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()

    # no need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)

    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    disc_step_start = train_config['disc_start']
    step_count = 0

    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0

    model.load_state_dict(torch.load(f"../{train_config['task_name']}/{train_config['vqvae_autoencoder_ckpt_name']}",
                       map_location=device))
    discriminator.load_state_dict(torch.load(f"../{train_config['task_name']}/{train_config['vqvae_discriminator_ckpt_name']}",
               map_location=device))

    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        # commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)

            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output

            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(f"../{train_config['task_name']}/vqvae_autoencoder_samples"):
                    os.mkdir(f"../{train_config['task_name']}/vqvae_autoencoder_samples")
                img.save(f"../{train_config['task_name']}/vqvae_autoencoder_samples/current_autoencoder_sample_{img_save_count}.png")
                img_save_count += 1
                img.close()

            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            lpips_loss = torch.mean(lpips_model(output, im)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            #####################################

            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred, torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred, torch.ones(disc_real_pred.shape, device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################

            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1, np.mean(recon_losses), np.mean(perceptual_losses), np.mean(codebook_losses),
                       np.mean(gen_losses), np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))

        torch.save(model.state_dict(), f"../{train_config['task_name']}/{train_config['vqvae_autoencoder_ckpt_name']}")
        torch.save(discriminator.state_dict(), f"../{train_config['task_name']}/{train_config['vqvae_discriminator_ckpt_name']}")
    print('Done Training...')

if __name__ == '__main__':
    """
    parser: ArgumentParser()
    args: Namespace(config_path='../config/my_celebhq.yaml')
    After using assign dest value，we can use args.config_path to get argument value.
    args.config_path: "../config/my_celebhq.yaml"
    """
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path', default='../config/my_deepfashion.yaml', type=str)
    args = parser.parse_args()
    train(args)