import torch
from einops import einsum
import torch.nn as nn
from src.models.my_blocks import get_time_embedding
from src.models.my_blocks import DownBlock, MidBlock, UpBlockUnet
from src.utils.my_config_utils import *


class Unet(nn.Module):
	r"""
	Unet model comprising
	Down blocks, Midblocks and Uplocks
	"""

	def __init__(self, im_channels, model_config):
		super().__init__()
		# self.down_channels: [256, 384, 512, 768]
		self.down_channels = model_config['down_channels']
		# self.mid_channels: [768, 512]
		self.mid_channels = model_config['mid_channels']
		# self.t_emb_dim: 512
		self.t_emb_dim = model_config['time_emb_dim']
		# self.down_sample: [True, True, True]
		self.down_sample = model_config['down_sample']
		# self.num_down_layers: 2
		self.num_down_layers = model_config['num_down_layers']
		# self.num_mid_layers: 2
		self.num_mid_layers = model_config['num_mid_layers']
		# self.num_up_layers: 2
		self.num_up_layers = model_config['num_up_layers']
		# self.attns: [True, True, True]
		self.attns = model_config['attn_down']
		# self.norm_channels: 32
		self.norm_channels = model_config['norm_channels']
		# self.num_heads: 16
		self.num_heads = model_config['num_heads']
		# self.conv_out_channels: 128
		self.conv_out_channels = model_config['conv_out_channels']
		self.im = None
		# validating unet model configurations
		assert self.mid_channels[0] == self.down_channels[-1]
		assert self.mid_channels[-1] == self.down_channels[-2]
		assert len(self.down_sample) == len(self.down_channels) - 1
		assert len(self.attns) == len(self.down_channels) - 1

		# Class, Mask and Text Conditioning Config
		self.class_cond = False
		self.text_cond = False
		self.image_cond = False
		self.pose_cond = False
		self.binary_mask_cond = False
		self.text_embed_dim = None
		"""
		self.condition_config: {
			condition_types: [ 'text', 'image' ]
			text_condition_config: {...}
			image_condition_config: {...}
		}
		"""
		self.condition_config = get_config_value(model_config, 'condition_config', None)
		if self.condition_config is not None:
			assert 'condition_types' in self.condition_config, 'Condition Type not provided in model config'
			# condition_types: ['text', 'image']
			condition_types = self.condition_config['condition_types']
			if 'class' in condition_types:
				validate_class_config(self.condition_config)
				self.class_cond = True
				self.num_classes = self.condition_config['class_condition_config']['num_classes']
			if 'text' in condition_types:
				validate_text_config(self.condition_config)
				self.text_cond = True
				# self.text_embed_dim: 512
				self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
			if 'image' in condition_types:
				self.image_cond = True
				# self.im_cond_input_ch: 18
				self.im_cond_input_ch = self.condition_config['image_condition_config'][
					'image_condition_input_channels']
				# self.im_cond_output_ch: 3
				self.im_cond_output_ch = self.condition_config['image_condition_config'][
					'image_condition_output_channels']

				if 'pose' in condition_types:
					self.pose_cond = True
					self.pose_cond_input_ch = self.condition_config['pose_condition_config'][
						'pose_condition_input_channels']
					self.pose_cond_output_ch = self.condition_config['pose_condition_config'][
						'pose_condition_output_channels']
				if 'binary_mask' in condition_types:
					self.binary_mask_cond = True

		if self.class_cond:
			# Rather than using a special null class we dont add the
			# class embedding information for unconditional generation
			self.class_emb = nn.Embedding(self.num_classes,
			                              self.t_emb_dim)

		if self.image_cond:
			"""
			map the mask image to a N channel image and
			concat that with input across channel dimension
			self.cond_conv_in: nnConv2d(18, 3, 1, 1, 0)
			"""
			self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
			                              out_channels=self.im_cond_output_ch,
			                              kernel_size=1,
			                              bias=False)

			if self.pose_cond and not self.binary_mask_cond:
				self.pose_conv_in = nn.Conv2d(in_channels=self.pose_cond_input_ch,
				                              out_channels=self.pose_cond_output_ch,
				                              kernel_size=1,
				                              bias=False)
				self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch + self.pose_cond_output_ch, self.down_channels[0], kernel_size=3, padding=1)
			elif self.pose_cond and self.binary_mask_cond:
				self.pose_conv_in = nn.Conv2d(in_channels=self.pose_cond_input_ch,
											  out_channels=self.pose_cond_output_ch,
											  kernel_size=1,
											  bias=False)
				self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch + self.pose_cond_output_ch + 1 + 3,
												self.down_channels[0], kernel_size=3, padding=1)
			else:
				self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch, self.down_channels[0], kernel_size=3, padding=1)

		else:
			self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

		# True
		self.cond = self.text_cond or self.image_cond or self.class_cond

		# initial projection from sinusoidal time embedding
		self.t_proj = nn.Sequential(
			# nn.Linear(512, 512)
			nn.Linear(self.t_emb_dim, self.t_emb_dim),
			nn.SiLU(),
			# nn.Linear(512, 512)
			nn.Linear(self.t_emb_dim, self.t_emb_dim)
		)
		# self.up_sample: [True, True, True]
		self.up_sample = list(reversed(self.down_sample))

		self.downs = nn.ModuleList([])

		# Build the Downblocks
		for i in range(len(self.down_channels) - 1):
			# cross attention and context dim only needed if text condition is present
			self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
			                            down_sample=self.down_sample[i],
			                            num_heads=self.num_heads,
			                            num_layers=self.num_down_layers,
			                            attn=self.attns[i], norm_channels=self.norm_channels,
			                            cross_attn=self.text_cond,
			                            context_dim=self.text_embed_dim))

		self.mids = nn.ModuleList([])
		# Build the Midblocks
		for i in range(len(self.mid_channels) - 1):
			self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
			                          num_heads=self.num_heads,
			                          num_layers=self.num_mid_layers,
			                          norm_channels=self.norm_channels,
			                          cross_attn=self.text_cond,
			                          context_dim=self.text_embed_dim))

		self.ups = nn.ModuleList([])
		# Build the Upblocks
		for i in reversed(range(len(self.down_channels) - 1)):
			self.ups.append(
				UpBlockUnet(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
				            self.t_emb_dim, up_sample=self.down_sample[i],
				            num_heads=self.num_heads,
				            num_layers=self.num_up_layers,
				            norm_channels=self.norm_channels,
				            cross_attn=self.text_cond,
				            context_dim=self.text_embed_dim))
		# self.norm_out: GroupNorm(32, 128, eps=1e-05, affine=True)
		self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
		# self.conv_out: Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
	def set_original_image(self, im):
		self.im = im

	def forward(self, x, t, cond_input=None):
		# Shapes assuming downblocks are [C1, C2, C3, C4]
		# Shapes assuming midblocks are [C4, C4, C3]
		# Shapes assuming downsamples are [True, True, False]
		if self.cond:
			assert cond_input is not None, "Model initialized with conditioning so cond_input cannot be None"
		if self.image_cond:
			"""
			mask conditioning
			shape of cond_input["text"]: torch.Size([16, 77, 512])
			shape of cond_input["image"]: torch.Size([16, 18, 512, 512])
			shape of x: torch.Size([16, 3, 32, 32])
			shape of t: torch.Size([16])
			"""
			validate_image_conditional_input(cond_input, x)
			# shape of im_cond: torch.Size([16, 18, 512, 512])
			im_cond = cond_input['image']
			# shape of im_cond: torch.Size([16, 18, 32, 32])
			im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
			# shape of im_cond: torch.Size([16, 3, 32, 32])
			im_cond = self.cond_conv_in(im_cond)
			assert im_cond.shape[-2:] == x.shape[-2:]


			if self.pose_cond and not self.binary_mask_cond:
				# shape of im_cond: torch.Size([16, 18, 512, 512])
				pose_cond = cond_input['pose']
				# shape of im_cond: torch.Size([16, 18, 32, 32])
				pose_cond = torch.nn.functional.interpolate(pose_cond, size=x.shape[-2:])
				# shape of im_cond: torch.Size([16, 3, 32, 32])
				pose_cond = self.pose_conv_in(pose_cond)
				assert pose_cond.shape[-2:] == x.shape[-2:]
				# shape of x: torch.Size([16, 8, 32, 32])
				x = torch.cat([x, im_cond, pose_cond], dim=1)
			elif self.pose_cond and self.binary_mask_cond:
				# shape of im_cond: torch.Size([16, 18, 512, 512])
				pose_cond = cond_input['pose']
				# shape of im_cond: torch.Size([16, 18, 32, 32])
				pose_cond = torch.nn.functional.interpolate(pose_cond, size=x.shape[-2:])
				# shape of im_cond: torch.Size([16, 3, 32, 32])
				pose_cond = self.pose_conv_in(pose_cond)
				assert pose_cond.shape[-2:] == x.shape[-2:]

				binary_mask_cond = cond_input['binary_mask']
				binary_mask_cond = torch.nn.functional.interpolate(binary_mask_cond, size=x.shape[-2:])
				outer_latent_cond = cond_input['outer_latent']

				# shape of x: torch.Size([16, 8, 32, 32])
				x = torch.cat([x, im_cond, pose_cond, binary_mask_cond, outer_latent_cond], dim=1)
			else:
				# shape of x: torch.Size([16, 6, 32, 32])
				x = torch.cat([x, im_cond], dim=1)
			# B x (C+N) x H x W
			# shape of out: torch.Size([16, 256, 32, 32])
			out = self.conv_in_concat(x)
		else:
			# B x C x H x W
			out = self.conv_in(x)
		"""
		torch.as_tensor(t).long(): make t to tensor type and round down (1.5 -> 1)
		self.t_emb_dim: 512
		shape of t_emb: torch.Size([16, 512])
		"""
		t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
		# shape of t_emb: torch.Size([16, 512])
		t_emb = self.t_proj(t_emb)
		# # class conditioning
		# if self.class_cond:
		#     validate_class_conditional_input(cond_input, x, self.num_classes)
		#     class_embed = einsum(cond_input['class'].float(), self.class_emb.weight, 'b n, n d -> b d')
		#     t_emb += class_embed
		context_hidden_states = None
		if self.text_cond:
			assert 'text' in cond_input, "Model initialized with text conditioning but cond_input has no text information"
			# shape of context_hidden_states: torch.Size([16, 77, 512])
			context_hidden_states = cond_input['text']

		down_outs = []
		for idx, down in enumerate(self.downs):
			down_outs.append(out)
			out = down(out, t_emb, context_hidden_states)
		# down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
		# out B x C4 x H/4 x W/4

		for mid in self.mids:
			out = mid(out, t_emb, context_hidden_states)
		# out B x C3 x H/4 x W/4

		for up in self.ups:
			down_out = down_outs.pop()
			out = up(out, down_out, t_emb, context_hidden_states)
		# out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
		out = self.norm_out(out)
		out = nn.SiLU()(out)
		out = self.conv_out(out)
		# out B x C x H x W
		return out
