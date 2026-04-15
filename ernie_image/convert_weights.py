"""Pre-convert and save MLX-ready weights.

Converts PyTorch safetensors to MLX format:
- Transposes Conv2d weights (NCHW → NHWC)
- Splits DiT and VAE into separate files
- Saves BN stats separately
- No JIT conversion at runtime
"""

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download


def convert(model_id: str = "baidu/ERNIE-Image-Turbo", output_dir: str = "weights"):
    """Download and convert weights to MLX format."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id}...")
    model_path = Path(snapshot_download(model_id))

    # --- DiT ---
    print("Converting DiT weights...")
    transformer_path = model_path / "transformer"
    dit_weights = {}
    for f in sorted(transformer_path.glob("*.safetensors")):
        dit_weights.update(mx.load(str(f)))

    # Remap keys + transpose conv weights
    dit_mapped = {}
    for key, value in dit_weights.items():
        new_key = key
        if "adaLN_modulation.1." in new_key:
            new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation_linear.")
        if "to_out.0." in new_key:
            new_key = new_key.replace("to_out.0.", "to_out.")
        if "weight" in new_key and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        dit_mapped[new_key] = value

    mx.savez(str(output / "dit.npz"), **dit_mapped)
    print(f"  Saved {len(dit_mapped)} tensors → dit.npz")

    # --- VAE ---
    print("Converting VAE weights...")
    vae_raw = mx.load(str(model_path / "vae" / "diffusion_pytorch_model.safetensors"))

    vae_mapped = {}
    bn_stats = {}
    for key, value in vae_raw.items():
        if key.startswith("encoder.") or key.startswith("quant_conv."):
            continue
        if key.startswith("bn."):
            bn_stats[key] = value
            continue
        if "weight" in key and value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        vae_mapped[key] = value

    mx.savez(str(output / "vae.npz"), **vae_mapped)
    mx.savez(str(output / "bn_stats.npz"), **bn_stats)
    print(f"  Saved {len(vae_mapped)} tensors → vae.npz")
    print(f"  Saved {len(bn_stats)} tensors → bn_stats.npz")

    # --- Config ---
    with open(transformer_path / "config.json") as f:
        config = json.load(f)
    # Strip non-essential keys
    keep = ["hidden_size", "num_attention_heads", "num_layers", "ffn_hidden_size",
            "in_channels", "out_channels", "patch_size", "text_in_dim",
            "rope_theta", "rope_axes_dim", "eps"]
    config = {k: config[k] for k in keep if k in config}
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    # Summary
    dit_size = sum(v.nbytes for v in dit_mapped.values()) / 1e9
    vae_size = sum(v.nbytes for v in vae_mapped.values()) / 1e9
    print(f"\nDone:")
    print(f"  DiT: {dit_size:.1f} GB ({len(dit_mapped)} tensors)")
    print(f"  VAE: {vae_size:.2f} GB ({len(vae_mapped)} tensors)")
    print(f"  Output: {output}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baidu/ERNIE-Image-Turbo")
    parser.add_argument("--output", "-o", default="weights")
    args = parser.parse_args()
    convert(args.model, args.output)
