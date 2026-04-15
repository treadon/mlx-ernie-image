"""FLUX.2-style VAE decoder in MLX.

Layer names match diffusers exactly so weights load directly.
Structure from safetensors:
  decoder.conv_in
  decoder.mid_block.resnets.{0,1} + decoder.mid_block.attentions.0
  decoder.up_blocks.{0,1,2,3}.resnets.{0,1,2} + upsamplers.0
  decoder.conv_norm_out + decoder.conv_out
  + post_quant_conv (outside decoder)
"""

import mlx.core as mx
import mlx.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x):
        B, H, W, C = x.shape
        G = self.num_groups
        x = x.reshape(B, H, W, G, C // G)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x.reshape(B, H, W, C)
        return x * self.weight + self.bias


class ResnetBlock(nn.Module):
    """Matches diffusers naming: norm1, conv1, norm2, conv2, [conv_shortcut]"""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.norm1 = GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.conv_shortcut = None

    def __call__(self, x):
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class AttentionBlock(nn.Module):
    """Matches diffusers: group_norm, to_q, to_k, to_v, to_out.0"""

    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.group_norm = GroupNorm(groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        # Match diffusers ModuleList naming: to_out.0
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.group_norm(x)
        x = x.reshape(B, H * W, C)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        scale = C ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v
        out = self.to_out[0](out)
        return out.reshape(B, H, W, C) + residual


class Upsample(nn.Module):
    """Matches diffusers: conv"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x):
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        return self.conv(x)


class UpBlock(nn.Module):
    """Matches diffusers: resnets.{0,1,2}, [upsamplers.0]"""

    def __init__(self, in_ch: int, out_ch: int, has_upsample: bool = True):
        super().__init__()
        self.resnets = [
            ResnetBlock(in_ch, out_ch),
            ResnetBlock(out_ch, out_ch),
            ResnetBlock(out_ch, out_ch),
        ]
        if has_upsample:
            self.upsamplers = [Upsample(out_ch)]
        else:
            self.upsamplers = None

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class MidBlock(nn.Module):
    """Matches diffusers: resnets.{0,1}, attentions.0"""

    def __init__(self, channels: int):
        super().__init__()
        self.resnets = [
            ResnetBlock(channels, channels),
            ResnetBlock(channels, channels),
        ]
        self.attentions = [AttentionBlock(channels)]

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class Decoder(nn.Module):
    """Full VAE decoder. Names match diffusers exactly."""

    def __init__(self):
        super().__init__()
        # conv_in: 32 -> 512
        self.conv_in = nn.Conv2d(32, 512, kernel_size=3, padding=1)

        # Mid block: 512
        self.mid_block = MidBlock(512)

        # Up blocks: reversed channel order
        # Block 0: 512 -> 512 (upsample)
        # Block 1: 512 -> 512 (upsample)
        # Block 2: 512 -> 256 (upsample)
        # Block 3: 256 -> 128 (no upsample)
        self.up_blocks = [
            UpBlock(512, 512, has_upsample=True),
            UpBlock(512, 512, has_upsample=True),
            UpBlock(512, 256, has_upsample=True),
            UpBlock(256, 128, has_upsample=False),
        ]

        # Final
        self.conv_norm_out = GroupNorm(32, 128)
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def __call__(self, z):
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = nn.silu(self.conv_norm_out(x))
        x = self.conv_out(x)
        return x


class VAEDecoder(nn.Module):
    """Wraps decoder + post_quant_conv. Top-level weight names match diffusers."""

    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.decoder = Decoder()

    def __call__(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)
