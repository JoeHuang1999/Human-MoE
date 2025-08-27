import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

class DeepFashionDataset(Dataset):
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

		# self.images: ["0.jpg", "1.jpg", ...]
		self.images, self.texts, self.masks, self.poses = self.load_images(im_path)


	def load_images(self, im_path):
		r"""
		Gets all images from the path specified
		and stacks them all up
		"""
		ims = []
		ims.append(f"src/deepfashion/cond_text_image_samples/face-ori.png")
		# get all the file names in specific folder with ".jpg" file extension
		texts = []
		masks = []
		poses = []
		if 'image' in self.condition_types:
			label_list = ['top', 'outer', 'skirt', 'dress', 'pants', 'leggings', 'headwear', 'eyeglass', 'neckwear',
			              'belt',
			              'footwear', 'bag', 'hair', 'face', 'skin', 'ring', 'wrist wearing', 'socks', 'gloves',
			              'necklace',
			              'rompers', 'earrings', 'tie']
			self.idx_to_cls_map = {idx: label_list[idx] for idx in range(len(label_list))}
			self.cls_to_idx_map = {label_list[idx]: idx for idx in range(len(label_list))}
		if 'text' in self.condition_types:
			captions_im = []
			with open(f"src/deepfashion/cond_text_image_samples/face-text.txt") as f:
				for line in f.readlines():
					captions_im.append(line.strip())
			texts.append(captions_im)
		if 'image' in self.condition_types:
			masks.append(f"src/deepfashion/cond_text_image_samples/face-parsing.png")
		if 'pose' in self.condition_types:
			poses.append(f"src/deepfashion/cond_text_image_samples/face-pose.png")
		if 'text' in self.condition_types:
			assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
		if 'image' in self.condition_types:
			assert len(masks) == len(ims), "Condition Type Image but could not find masks for all images"
		if 'pose' in self.condition_types:
			assert len(poses) == len(ims), "Condition Type Image but could not find poses for all images"
		print('Found {} images'.format(len(ims)))
		print('Found {} masks'.format(len(masks)))
		print('Found {} poses'.format(len(poses)))
		print('Found {} captions'.format(len(texts)))
		return ims, texts, masks, poses

	def get_name(self, index):
		return self.images[index]

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
		mask_im = Image.open(self.masks[index])
		mask_im = np.array(mask_im)
		im_base = np.zeros((self.mask_h, self.mask_w, self.mask_channels))
		for orig_idx in range(len(self.idx_to_cls_map)):
			im_base[mask_im == (orig_idx + 1), orig_idx] = 1
		mask = torch.from_numpy(im_base).permute(2, 0, 1).float()
		return mask

	def get_pose(self, index):
		pose_im = Image.open(self.poses[index])
		pose_im = np.array(pose_im)
		pose = torch.from_numpy(pose_im).float()
		pose = pose.permute(2, 0, 1)
		return pose

	def __len__(self):
		return len(self.images)

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
			#flip = random.random() > 0.5
			flip  =False
			if flip:
				# print("Yes!")
				im_tensor = F.hflip(im_tensor)
				if 'image' in self.condition_types:
					cond_inputs['image'] = F.hflip(cond_inputs['image'])
				if 'pose' in self.condition_types:
					cond_inputs['pose'] = F.hflip(cond_inputs['pose'])
			else:
				# print("No!")
				pass
			im.close()

			# Convert input to -1 to 1 range.
			im_tensor = (2 * im_tensor) - 1

			if len(self.condition_types) == 0:
				return im_tensor
			else:
				return im_tensor, cond_inputs
