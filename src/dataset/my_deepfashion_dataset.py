import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from src.utils.my_diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class DeepFashionDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    """
    split: "train"
    im_path: "../data/CelebAMask-HQ"
    im_size: 256
    im_channels: 3
    im_ext: "jpg"
    use_latents=False
    latent_path=None
    condition_config=None
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, condition_config=None, is_train=True):
        self.is_train = is_train

        # self.split: "train"
        self.split = split

        # self.im_size: [512, 256]
        self.im_size = im_size

        # self.im_channels: 3
        self.im_channels = im_channels

        # self.im_ext: "jpg"
        self.im_ext = im_ext

        # self.im_path: "../data/DeepFashion"
        self.im_path = im_path

        # self.latent_maps: None
        self.latent_maps = None

        # self.use_latents: False
        self.use_latents = False

        # self.condition_type: []
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.idx_to_cls_map = {}
        self.cls_to_idx_map = {}

        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']

        if 'pose' in self.condition_types:
            self.pose_channels = condition_config['pose_condition_config']['pose_condition_input_channels']
            self.pose_h = condition_config['pose_condition_config']['pose_condition_h']
            self.pose_w = condition_config['pose_condition_config']['pose_condition_w']

        if 'binary_mask' in self.condition_types:
            self.binary_mask_channels = condition_config['binary_mask_condition_config']['binary_mask_condition_input_channels']
            self.binary_mask_h = condition_config['binary_mask_condition_config']['binary_mask_condition_h']
            self.binary_mask_w = condition_config['binary_mask_condition_config']['binary_mask_condition_w']

        # self.images: ["0.jpg", "1.jpg", ...]
        self.texts, self.masks, self.poses, self.binary_masks = self.load_images(im_path)

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        # get all the file names in specific folder with ".jpg" file extension
        texts = []
        masks = []
        poses = []
        binary_masks = []
        if 'image' in self.condition_types:
            label_list = ['top', 'outer', 'skirt', 'dress', 'pants', 'leggings', 'headwear', 'eyeglass', 'neckwear',
                          'belt',
                          'footwear', 'bag', 'hair', 'face', 'skin', 'ring', 'wrist wearing', 'socks', 'gloves',
                          'necklace',
                          'rompers', 'earrings', 'tie']
            self.idx_to_cls_map = {idx: label_list[idx] for idx in range(len(label_list))}
            self.cls_to_idx_map = {label_list[idx]: idx for idx in range(len(label_list))}

        # ims: ["0.jpg", "1.jpg", ...]

        if 'text' in self.condition_types:
            captions_im = []
            with open(f"src/deepfashion/cond_text_image_samples/text.txt") as f:
                for line in f.readlines():
                    captions_im.append(line.strip())
            texts.append(captions_im)

        if 'image' in self.condition_types:
            masks.append(f"src/deepfashion/cond_text_image_samples/parsing.png")

        if 'pose' in self.condition_types:
            poses.append(f"src/deepfashion/cond_text_image_samples/pose.png")

        if 'binary_mask' in self.condition_types:
            binary_masks.append(f"src/deepfashion/cond_text_image_samples/mask.png")

        print('Found {} masks'.format(len(masks)))
        print('Found {} poses'.format(len(poses)))
        print('Found {} captions'.format(len(texts)))
        print('Found {} binary_masks'.format(len(binary_masks)))
        return texts, masks, poses, binary_masks

    def get_name(self, index):
        return "result.png"

    def get_text(self, index, text_index=0):
        return self.texts[index][text_index]

    def get_mask(self, index):
        r"""
        Method to get the mask of WxH
        for given index and convert it into
        Classes x W x H mask image
        :param index:
        :return:
        """
        # print(self.masks[index])
        mask_im = Image.open(self.masks[index])
        mask_im = np.array(mask_im)
        im_base = np.zeros((self.mask_h, self.mask_w, self.mask_channels))
        for orig_idx in range(len(self.idx_to_cls_map)):
            im_base[mask_im == (orig_idx + 1), orig_idx] = 1
        mask = torch.from_numpy(im_base).permute(2, 0, 1).float()
        print(mask.shape)
        return mask


    def get_pose(self, index):
        # pose_ims = []
        #np.concatenate(images, axis=-1)
        #for i in range(0, 25):
        pose_im = Image.open(self.poses[index])
        pose_im = np.array(pose_im)
        #pose_ims.append(pose_im)
        # shape: [512, 256, 25]
        #pose_ims = np.stack(pose_ims, axis=-1)
        pose = torch.from_numpy(pose_im).float()
        pose = pose.permute(2, 0, 1)
        return pose

    def get_binary_mask(self, index, type=None):
        choose = None
        if type is None:
            choose = random.randint(1, 4)
            if choose == 1:
                choose = random.randint(1, 23)
            elif choose == 2:
                choose = 24
            elif choose == 3:
                choose = 25
        else:
            choose = type
        binary_mask_im = Image.open(self.binary_masks[index].replace("choose", str(choose))).convert("L")
        binary_mask_im = np.array(binary_mask_im)
        binary_mask_im[binary_mask_im < 128] = 0
        binary_mask_im[binary_mask_im >= 128] = 1
        binary_mask = torch.from_numpy(binary_mask_im).float()
        binary_mask = binary_mask.unsqueeze(-1)
        binary_mask = binary_mask.permute(2, 0, 1)
        return binary_mask

    def __len__(self):
        return 1

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]
        if 'image' in self.condition_types:
            mask = self.get_mask(index)
            cond_inputs['image'] = mask
        if 'pose' in self.condition_types:
            pose = self.get_pose(index)
            cond_inputs['pose'] = pose
        if 'binary_mask' in self.condition_types:
            binary_mask = self.get_binary_mask(index)
            cond_inputs['binary_mask'] = binary_mask
        #######################################

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])

            # im_tensor: (3, 256, 256)
            # pixel values between 0~1
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize((self.im_size[0], self.im_size[1])),
                # torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
