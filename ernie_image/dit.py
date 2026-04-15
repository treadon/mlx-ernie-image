"""ERNIE-Image DiT (Diffusion Transformer) in MLX — Optimized.

Single-stream DiT with:
- 36 layers, hidden size 4096, 32 heads
- 3D RoPE (text_idx, y, x)
- AdaLN modulation from timestep
- QK RMSNorm
- Gated FFN (SwiGLU-style with GELU)
- mx.fast.scaled_dot_product_attention for fused attention
"""

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


def rope(pos, dim: int, theta: int = 256):
    """Compute rotary position embedding angles."""
    scale = mx.arange(0, dim, 2).astype(mx.float32) / dim
    omega = 1.0 / (theta ** scale)
    out = mx.expand_dims(pos.astype(mx.float32), -1) * mx.expand_dims(omega, 0)
    return out


class EmbedND3(nn.Module):
    """3D RoPE embedding for (text_idx, y, x) position encoding."""

    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def __call__(self, ids):
        embs = [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        emb = mx.concatenate(embs, axis=-1)
        emb = mx.expand_dims(emb, 2)
        emb = mx.stack([emb, emb], axis=-1)
        return emb.reshape(*emb.shape[:3], -1)


def apply_rotary_emb(x, freqs_cis):
    """Apply rotary embeddings. Non-interleaved rotate_half."""
    rot_dim = freqs_cis.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos_ = mx.cos(freqs_cis).astype(x.dtype)
    sin_ = mx.sin(freqs_cis).astype(x.dtype)
    x1 = x_rot[..., :rot_dim // 2]
    x2 = x_rot[..., rot_dim // 2:]
    x_rotated = mx.concatenate([-x2, x1], axis=-1)
    out = x_rot * cos_ + x_rotated * sin_
    return mx.concatenate([out, x_pass], axis=-1)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def __call__(self, x):
        x = self.proj(x)
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D)


class Attention(nn.Module):
    """Multi-head attention with QK RMSNorm, RoPE, and fused SDPA."""

    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

    def __call__(self, x, freqs_cis=None, mask=None):
        B, S, _ = x.shape

        q = self.to_q(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.to_k(x).reshape(B, S, self.num_heads, self.head_dim)
        v = self.to_v(x).reshape(B, S, self.num_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        # MLX SDPA expects [B, H, S, D]
        q = q.transpose(0, 2, 1, 3)  # [B, heads, S, D]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=1.0 / self.scale,
            mask=mask,  # [B, 1, 1, S] broadcasts fine with [B, H, S, D]
        )

        # Back to [B, S, H*D]
        out = out.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.linear_fc2(self.up_proj(x) * nn.gelu(self.gate_proj(x)))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.self_attention = Attention(hidden_size, num_heads, eps=eps)
        self.adaLN_mlp_ln = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = FeedForward(hidden_size, ffn_hidden_size)

    def __call__(self, x, freqs_cis, temb, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = x
        x = self.adaLN_sa_ln(x)
        x = x * (1 + scale_msa) + shift_msa

        x_bsh = x.transpose(1, 0, 2)
        attn_out = self.self_attention(x_bsh, freqs_cis=freqs_cis, mask=mask)
        attn_out = attn_out.transpose(1, 0, 2)

        x = residual + gate_msa * attn_out

        residual = x
        x = self.adaLN_mlp_ln(x)
        x = x * (1 + scale_mlp) + shift_mlp
        x = residual + gate_mlp * self.mlp(x)

        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def __call__(self, t):
        return self.linear_2(self.act(self.linear_1(t)))


class AdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, affine=False)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def __call__(self, x, conditioning):
        chunks = self.linear(conditioning)
        scale = chunks[..., :chunks.shape[-1] // 2]
        shift = chunks[..., chunks.shape[-1] // 2:]
        x = self.norm(x)
        return x * (1 + mx.expand_dims(scale, 0)) + mx.expand_dims(shift, 0)


def timestep_embedding(t, dim: int):
    half = dim // 2
    freqs = mx.exp(-math.log(10000) * mx.arange(0, half).astype(mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None, :]
    embedding = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class ErnieImageDiT(nn.Module):
    """ERNIE-Image Diffusion Transformer — Optimized for MLX."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: Tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim

        self.x_embedder = PatchEmbed(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_embedding = TimestepEmbedding(hidden_size)
        self.pos_embed = EmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)
        self.adaLN_modulation_silu = nn.SiLU()
        self.adaLN_modulation_linear = nn.Linear(hidden_size, 6 * hidden_size)
        self.layers = [
            DiTBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps)
            for _ in range(num_layers)
        ]
        self.final_norm = AdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def __call__(self, hidden_states, timestep, text_bth, text_lens):
        B = hidden_states.shape[0]
        H, W = hidden_states.shape[1], hidden_states.shape[2]
        p = self.patch_size
        Hp, Wp = H // p, W // p
        N_img = Hp * Wp

        img_bsh = self.x_embedder(hidden_states)

        if self.text_proj is not None and text_bth.size > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]

        x_bsh = mx.concatenate([img_bsh, text_bth], axis=1)
        x = x_bsh.transpose(1, 0, 2)
        S = x.shape[0]

        # Position IDs
        grid_y = mx.arange(Hp).astype(mx.float32)
        grid_x = mx.arange(Wp).astype(mx.float32)
        gy, gx = mx.meshgrid(grid_y, grid_x, indexing="ij")
        grid_yx = mx.stack([gy.reshape(-1), gx.reshape(-1)], axis=-1)

        text_lens_f = text_lens.astype(mx.float32)
        image_ids = mx.concatenate([
            mx.broadcast_to(text_lens_f[:, None, None], (B, N_img, 1)),
            mx.broadcast_to(grid_yx[None], (B, N_img, 2)),
        ], axis=-1)

        if Tmax > 0:
            text_seq = mx.arange(Tmax).astype(mx.float32)
            text_ids = mx.concatenate([
                mx.broadcast_to(text_seq[None, :, None], (B, Tmax, 1)),
                mx.zeros((B, Tmax, 2)),
            ], axis=-1)
            all_ids = mx.concatenate([image_ids, text_ids], axis=1)
        else:
            all_ids = image_ids

        rotary_pos_emb = self.pos_embed(all_ids)

        # Attention mask for fused SDPA: additive mask, 0=attend, -inf=ignore
        img_valid = mx.ones((B, N_img), dtype=mx.bool_)
        if Tmax > 0:
            text_valid = mx.arange(Tmax)[None, :] < text_lens[:, None]
            valid = mx.concatenate([img_valid, text_valid], axis=1)
        else:
            valid = img_valid
        # SDPA needs [B, 1, 1, S] additive mask
        mask = mx.where(valid[:, None, None, :], mx.array(0.0), mx.array(-1e9))

        # Timestep
        t_emb = timestep_embedding(timestep, self.hidden_size).astype(x.dtype)
        c = self.time_embedding(t_emb)

        modulation = self.adaLN_modulation_linear(self.adaLN_modulation_silu(c))
        chunk_size = self.hidden_size
        mod_chunks = [modulation[..., i * chunk_size:(i + 1) * chunk_size] for i in range(6)]
        temb = [mx.broadcast_to(m[None], (S, B, chunk_size)) for m in mod_chunks]

        for layer in self.layers:
            x = layer(x, rotary_pos_emb, temb, mask=mask)

        x = self.final_norm(x, c).astype(x.dtype)
        patches = self.final_linear(x)[:N_img]
        patches = patches.transpose(1, 0, 2)
        return patches.reshape(B, Hp, Wp, self.out_channels)
