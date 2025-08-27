import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    """
    shape of time_steps: torch.Size([16])
    shape of temb_dim: 512
    torch.arange(start=1, end=5): tensor([1, 2, 3, 4])
    factor: 10000^(2i/d_model)
    """
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    """
    shape of time_steps[:, None]: torch.Size([16, 1])
    shape of t_emb: torch.Size([16, 256])
    """
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    # shape of t_emb: torch.Size([16, 512])
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    r"""
    Down conv block with attention.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        """
        vqvae: 2
        ldm: 2
        """
        self.num_layers = num_layers
        """
        vqvae: True
        ldm: True
        """
        self.down_sample = down_sample
        """
        vqvae: False
        ldm: True
        """
        self.attn = attn
        """
        vqvae: None
        ldm: 512
        """
        self.context_dim = context_dim
        """
        vqvae: False
        ldm: True
        """
        self.cross_attn = cross_attn
        """
        vqvae: None
        ldm: 512
        """
        self.t_emb_dim = t_emb_dim

        """
        vqvae:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 64, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 128, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        ldm:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    # divide param2 channels into param1 groups
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            """
            ldm:
            ModuleList(
                (0): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=384, bias=True)
                )
                (1): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=384, bias=True)
                )
            )
            """
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        """
        vqvae:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 128, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 128, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        ldm:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            """
            ldm:
            ModuleList(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): GroupNorm(32, 384, eps=1e-05, affine=True)
            )
            """
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            """
            ldm:
            ModuleList(
                (0): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
                (1): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
            )
            """
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            """
            ldm:
            ModuleList(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): GroupNorm(32, 384, eps=1e-05, affine=True)
            )
            """
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            """
            ldm:
            ModuleList(
                (0): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
                (1): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
            )
            """
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            """
            ldm:
            ModuleList(
                (0): Linear(in_features=512, out_features=384, bias=True)
                (1): Linear(in_features=512, out_features=384, bias=True)
            )
            """
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        """
        vqvae:
        ModuleList(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        ldm:
        ModuleList(
            (0): Conv2d(256, 384, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
        )
        """
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        """
        vqvae:
        Conv2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        ldm:
        Conv2d(384, 384, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        """
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            # Resnet block of Unet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                # Attention block of Unet
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None,
                 context_dim=None):
        super().__init__()
        """
        vqvae: 2
        ldm: 2
        """
        self.num_layers = num_layers
        """
        vqvae: 512
        ldm: 512
        """
        self.t_emb_dim = t_emb_dim
        """
        vqvae: 512
        ldm: 512
        """
        self.context_dim = context_dim
        """
        vqvae: None
        ldm: True
        """
        self.cross_attn = cross_attn
        """
        vqvae:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        ldm:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 768, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 512, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): Sequential(
                (0): GroupNorm(32, 512, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )

        if self.t_emb_dim is not None:
            """
            ldm:
            ModuleList(
                (0): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=512, bias=True)
                )
                (1): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=512, bias=True)
                )
                (2): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=512, bias=True)
                )
            )
            """
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        """
        vqvae:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        ldm:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 512, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 512, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): Sequential(
                (0): GroupNorm(32, 512, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        """
        vqvae:
        ModuleList(
            (0): GroupNorm(32, 256, eps=1e-05, affine=True)
            (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        )
        ldm:
        ModuleList(
            (0): GroupNorm(32, 512, eps=1e-05, affine=True)
            (1): GroupNorm(32, 512, eps=1e-05, affine=True)
        )
        """
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )
        """
        vqvae:
        ModuleList(
            (0): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
            )
            (1): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
            )
        )
        ldm:
        ModuleList(
            (0): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (1): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
        )
        """
        self.attentions = nn.ModuleList(
            # param1: the embedding dimensions that should be same as output channels
            # param2: number of heads
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            """
            ldm:
            ModuleList(
                (0): GroupNorm(32, 512, eps=1e-05, affine=True)
                (1): GroupNorm(32, 512, eps=1e-05, affine=True)
            )
            """
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            """
            ldm:
            ModuleList(
                (0): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
                )
                (1): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
                )
            )
            """
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            """
            ldm:
            ModuleList(
                (0): Linear(in_features=512, out_features=512, bias=True)
                (1): Linear(in_features=512, out_features=512, bias=True)
            )
            """
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        """
        vqvae:
        ModuleList(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        ldm:
        ModuleList(
            (0): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        """
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        out = x

        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        """
        vqvae: 2
        lm: 2
        """
        self.num_layers = num_layers
        """
        vqvae: True 
        ldm: True
        """
        self.up_sample = up_sample
        """
        vqvae: None
        ldm: 512
        """
        self.t_emb_dim = t_emb_dim
        """
        vqvae: None
        ldm: 
        """
        # self.attn: None
        self.attn = attn

        """
        self.resnet_conv_first:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])

        """
        self.resnet_conv_second:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 256, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )

            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        """
        self.residual_input_conv:
        ModuleList(
            (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        """
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        """
        self.up_sample_conv:
        ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        """
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        # Upsample
        x = self.up_sample_conv(x)

        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Self Attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return out


class UpBlockUnet(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
                 num_heads, num_layers, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        """
        ldm: 2
        """
        self.num_layers = num_layers
        """
        ldm : True
        """
        self.up_sample = up_sample
        """
        ldm: 512
        """
        self.t_emb_dim = t_emb_dim
        """
        ldm: True
        """
        self.cross_attn = cross_attn
        """
        ldm: 512
        """
        self.context_dim = context_dim
        """
        ldm: 
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 1024, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(1024, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            """
            ModuleList(
                (0): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=384, bias=True)
                )
                (1): Sequential(
                    (0): SiLU()
                    (1): Linear(in_features=512, out_features=384, bias=True)
                )
            )
            """
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        """
        ldm:
        ModuleList(
            (0): Sequential(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): Sequential(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): SiLU()
                (2): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        """
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        """
        ldm:
        ModuleList(
            (0): GroupNorm(32, 384, eps=1e-05, affine=True)
            (1): GroupNorm(32, 384, eps=1e-05, affine=True)
        )
        """
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ]
        )
        """
        ldm:
        ModuleList(
            (0): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
            )
            (1): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
            )
        )
        """
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            """
            ldm:
            ModuleList(
                (0): GroupNorm(32, 384, eps=1e-05, affine=True)
                (1): GroupNorm(32, 384, eps=1e-05, affine=True)
            )
            """
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            """
            ldm:
            ModuleList(
                (0): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
                (1): MultiheadAttention(
                    (out_proj): NonDynamicallyQuantizableLinear(in_features=384, out_features=384, bias=True)
                )
            )
            """
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            """
            ldm: 
            ModuleList(
                (0): Linear(in_features=512, out_features=384, bias=True)
                (1): Linear(in_features=512, out_features=384, bias=True)
            )
            """
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        """
        ldm:
        ModuleList(
            (0): Conv2d(1024, 384, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
        )
        """
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        """
        ldm:
        ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        """
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None, context=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            #print(len(x[0]))
            #x[0][:len(x[0]) // 2] *= 1.2
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Self Attention
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            # Cross Attention
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert len(context.shape) == 3, \
                    "Context shape does not match B,_,CONTEXT_DIM"
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim, \
                    "Context shape does not match B,_,CONTEXT_DIM"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        return out


