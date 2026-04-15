"""ERNIE-Image MLX Pipeline — Optimized.

Key optimizations:
- Pre-converted weights (no JIT transposition)
- mx.fast.scaled_dot_product_attention
- Single mx.eval() at end of denoising loop, not per step
"""

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .dit import ErnieImageDiT
from .vae import VAEDecoder
from .scheduler import FlowMatchEulerScheduler


def unpatchify(latents):
    B, H, W, C = latents.shape
    latents = latents.reshape(B, H, W, C // 4, 2, 2)
    latents = latents.transpose(0, 1, 4, 2, 5, 3)
    return latents.reshape(B, H * 2, W * 2, C // 4)


class ErnieImagePipeline:

    def __init__(self, dit, vae, scheduler, bn_mean, bn_var):
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.bn_mean = bn_mean
        self.bn_var = bn_var

    @staticmethod
    def from_pretrained(model_id: str = "treadon/ERNIE-Image-Turbo-MLX", quantize: Optional[int] = None):
        """Load from HuggingFace MLX weights repo."""
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model_id))
        return ErnieImagePipeline.from_weights(str(path), quantize=quantize)

    @staticmethod
    def from_weights(weights_dir: str, quantize: Optional[int] = None):
        """Load from local pre-converted weights."""
        path = Path(weights_dir)

        with open(path / "config.json") as f:
            config = json.load(f)

        dit = ErnieImageDiT(
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            num_layers=config["num_layers"],
            ffn_hidden_size=config["ffn_hidden_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            patch_size=config["patch_size"],
            text_in_dim=config["text_in_dim"],
            rope_theta=config["rope_theta"],
            rope_axes_dim=tuple(config["rope_axes_dim"]),
        )
        dit.load_weights(list(dict(mx.load(str(path / "dit.npz"))).items()))

        if quantize:
            nn.quantize(dit, bits=quantize)

        vae = VAEDecoder()
        vae.load_weights(list(dict(mx.load(str(path / "vae.npz"))).items()))

        bn = dict(mx.load(str(path / "bn_stats.npz")))
        bn_mean = bn["bn.running_mean"]
        bn_var = bn["bn.running_var"]

        return ErnieImagePipeline(dit, vae, FlowMatchEulerScheduler(shift=4.0), bn_mean, bn_var)


    def generate(
        self,
        text_embeddings=None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 8,
        seed: Optional[int] = None,
    ) -> Image.Image:
        t_start = time.time()

        if text_embeddings is None:
            text_embeddings = mx.zeros((1, self.dit.text_in_dim))

        text_bth = mx.expand_dims(text_embeddings, 0)
        text_lens = mx.array([text_embeddings.shape[0]])

        latent_h, latent_w = height // 16, width // 16
        if seed is not None:
            mx.random.seed(seed)
        latents = mx.random.normal((1, latent_h, latent_w, 128))

        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising — no per-step eval, let MLX build the full graph
        for i, t in enumerate(self.scheduler.timesteps):
            pred = self.dit(latents, mx.array([t.item()]), text_bth, text_lens)
            latents = self.scheduler.step(pred, i, latents)

        mx.eval(latents)
        denoise_time = time.time() - t_start

        # Decode
        bn_mean = self.bn_mean.reshape(1, 1, 1, -1)
        bn_std = mx.sqrt(self.bn_var.reshape(1, 1, 1, -1) + 1e-5)
        latents = latents * bn_std + bn_mean
        latents = unpatchify(latents)
        images = self.vae(latents)
        mx.eval(images)
        total_time = time.time() - t_start

        print(f"  Denoise: {denoise_time:.1f}s | Decode: {total_time - denoise_time:.1f}s | Total: {total_time:.1f}s")

        images = mx.clip(images, -1, 1)
        images = ((images + 1) / 2 * 255).astype(mx.uint8)
        return Image.fromarray(np.array(images[0]))
