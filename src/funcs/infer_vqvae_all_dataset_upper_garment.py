import argparse
import glob
import os
import sys
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

# 獲取當前文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
# 獲取上級目錄
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 添加上級目錄到sys.path中
sys.path.append(parent_dir)
from dataset.my_deepfashion_dataset_upper_garment import DeepFashionDataset
from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    # Create the dataset
    im_dataset_cls = {
        'deepfashion_upper_garment': DeepFashionDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=f"../{dataset_config['im_path']}",
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                is_train=True,
                                )

    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']

    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(f"../{train_config['task_name']}/{train_config['vqvae_autoencoder_ckpt_name']}",
                                     map_location=device))
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(im_dataset))):
            ims = torch.cat([im_dataset[i][None, :]]).float()
            ims = ims.to(device)

            encoded_output, _ = model.encode(ims)
            decoded_output = model.decode(encoded_output)
            encoded_output = torch.clamp(encoded_output, -1., 1.)
            encoded_output = (encoded_output + 1) / 2
            decoded_output = torch.clamp(decoded_output, -1., 1.)
            decoded_output = (decoded_output + 1) / 2
            ims = (ims + 1) / 2

            encoder_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
            decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
            input_grid = make_grid(ims.cpu(), nrow=ngrid)
            encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
            decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
            input_grid = torchvision.transforms.ToPILImage()(input_grid)

            input_grid.save(f"../{train_config['task_name']}/sample_vqvae/vqvae_input_images/{os.path.split(im_dataset.get_name(i))[1]}")
            encoder_grid.save(f"../{train_config['task_name']}/sample_vqvae/vqvae_encoded_images/{os.path.split(im_dataset.get_name(i))[1]}")
            decoder_grid.save(f"../{train_config['task_name']}/sample_vqvae/vqvae_reconstructed_images/{os.path.split(im_dataset.get_name(i))[1]}")

            if train_config['save_latents']:
                # save Latents (but in a very unoptimized way)
                latent_path = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
                latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'],
                                                       '*.pkl'))
                assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
                if not os.path.exists(latent_path):
                    os.mkdir(latent_path)
                print('Saving Latents for {}'.format(dataset_config['name']))

                fname_latent_map = {}
                part_count = 0
                count = 0
                for idx, im in enumerate(tqdm(data_loader)):
                    encoded_output, _ = model.encode(im.float().to(device))
                    fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                    # Save latents every 1000 images
                    if (count + 1) % 1000 == 0:
                        pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                        '{}.pkl'.format(part_count)), 'wb'))
                        part_count += 1
                        fname_latent_map = {}
                    count += 1
                if len(fname_latent_map) > 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    '{}.pkl'.format(part_count)), 'wb'))
                print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='../config/my_deepfashion_upper_garment.yaml', type=str)
    args = parser.parse_args()
    infer(args)
