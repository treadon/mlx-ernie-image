"""Weight loading for ERNIE-Image models.

Loads safetensors weights from HuggingFace and maps them to our MLX model.
Handles the key name mapping between PyTorch (diffusers) and MLX conventions.
"""

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from safetensors import safe_open


def download_model(model_id: str) -> Path:
    """Download model from HuggingFace."""
    path = snapshot_download(model_id)
    return Path(path)


def load_safetensors(path: Path, pattern: str = "*.safetensors"):
    """Load all safetensors files from a directory into a flat dict."""
    weights = {}
    for f in sorted(path.glob(pattern)):
        # Use mlx's native safetensors loading (handles bfloat16)
        file_weights = mx.load(str(f))
        weights.update(file_weights)
    return weights


def load_dit_weights(model_path: Path):
    """Load and map DiT (transformer) weights.

    The safetensors use diffusers naming:
        transformer/diffusion_pytorch_model*.safetensors

    Key mapping examples:
        PyTorch: x_embedder.proj.weight -> MLX: x_embedder.proj.weight
        PyTorch: layers.0.self_attention.to_q.weight -> MLX: layers.0.self_attention.to_q.weight

    Most names should match directly since we named our MLX layers
    to mirror the PyTorch structure.
    """
    transformer_path = model_path / "transformer"
    if not transformer_path.exists():
        transformer_path = model_path

    weights = load_safetensors(transformer_path)

    # Map weight names
    mapped = {}
    for key, value in weights.items():
        new_key = key

        # Conv2d weights: PyTorch [out_ch, in_ch, kH, kW] -> MLX [out_ch, kH, kW, in_ch]
        if "weight" in key and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)

        # Handle the adaLN_modulation sequential -> split into silu + linear
        if "adaLN_modulation.1." in new_key:
            new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation_linear.")

        # Handle to_out ModuleList -> single linear
        if "to_out.0." in new_key:
            new_key = new_key.replace("to_out.0.", "to_out.")

        mapped[new_key] = value

    return mapped


def load_vae_weights(model_path: Path):
    """Load VAE decoder weights."""
    vae_path = model_path / "vae"
    if not vae_path.exists():
        vae_path = model_path

    weights = load_safetensors(vae_path)

    mapped = {}
    for key, value in weights.items():
        # Only load decoder weights (skip encoder)
        if key.startswith("encoder."):
            continue

        new_key = key

        # Remove 'decoder.' prefix if present
        if new_key.startswith("decoder."):
            new_key = new_key[len("decoder."):]

        # Conv2d weight transposition
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)

        mapped[new_key] = value

    return mapped, weights


def load_bn_stats(model_path: Path):
    """Load batch norm running stats for latent denormalization."""
    vae_path = model_path / "vae"
    if not vae_path.exists():
        vae_path = model_path

    weights = load_safetensors(vae_path)

    bn_mean = weights.get("bn.running_mean", None)
    bn_var = weights.get("bn.running_var", None)

    if bn_mean is None:
        # Try alternative names
        for k, v in weights.items():
            if "bn" in k and "running_mean" in k:
                bn_mean = v
            if "bn" in k and "running_var" in k:
                bn_var = v

    return bn_mean, bn_var
