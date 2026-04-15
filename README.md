# mlx-ernie-image

ERNIE-Image text-to-image generation on Apple Silicon.

Uses MLX for the 8B diffusion transformer and VAE (where all the compute is), PyTorch for the 3.8B text encoder (runs once in 0.1s).

## Quick Start

```bash
# Install
pip install -e .

# One-time weight conversion
python generate.py --convert-weights

# Generate
python generate.py -p "A vibrant manga comic about a cat and a dragon"

# Interactive mode
python generate.py --interactive

# With seed
python generate.py -p "A movie poster" --seed 42 -o poster.png
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- ~30GB RAM (model loads in BF16)
- ~16GB disk for pre-converted weights

## Benchmarks (1024x1024, 8 steps, M4 Pro 64GB)

| Component | Framework | Time |
|-----------|-----------|------|
| Text encode | PyTorch/MPS | 0.1s |
| Denoise (8 steps) | MLX | 128s |
| VAE decode | MLX | 6s |
| **Total** | | **~134s** |

## Architecture

```
"A cat in space"
    → Text Encoder (Mistral-3, PyTorch, 0.1s)
    → DiT (8B, 36 layers, MLX, 128s)
    → VAE Decoder (84M, MLX, 6s)
    → image.png
```

## Project Structure

```
ernie_image/
├── dit.py              # 8B Diffusion Transformer (MLX)
├── vae.py              # FLUX.2 VAE Decoder (MLX)
├── scheduler.py        # Flow Matching Euler Scheduler
├── text_encoder.py     # Mistral-3 text encoding (PyTorch hybrid)
├── pipeline.py         # Ties everything together
├── weights.py          # HuggingFace weight loading
└── convert_weights.py  # One-time weight pre-conversion
```

## Base Model

[baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) — Apache 2.0

## License

Code: MIT. Model weights: Apache 2.0 (Baidu).
