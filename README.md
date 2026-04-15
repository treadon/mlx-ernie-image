# mlx-ernie-image

ERNIE-Image text-to-image generation on Apple Silicon.

Uses MLX for the 8B diffusion transformer and VAE (where all the compute is), PyTorch for the 3.8B text encoder (runs once in 0.1s).

Pre-converted MLX weights: [treadon/ERNIE-Image-Turbo-MLX](https://huggingface.co/treadon/ERNIE-Image-Turbo-MLX)

## Quick Start

```bash
# Install
pip install -e .

# Generate (weights download automatically from HuggingFace)
python generate.py -p "A vibrant manga comic about a cat and a dragon"

# Interactive mode
python generate.py --interactive

# With seed
python generate.py -p "A movie poster" --seed 42 -o poster.png
```

## Python API

```python
from ernie_image import ErnieImagePipeline, TextEncoder

te = TextEncoder.from_pretrained()
pipe = ErnieImagePipeline.from_pretrained("treadon/ERNIE-Image-Turbo-MLX")

emb = te.encode("A cat discovering a tiny dragon in its food bowl")
img = pipe.generate(text_embeddings=emb)
img.save("output.png")
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+
- ~30GB RAM (model loads in BF16)
- ~16GB disk (weights cached after first download)

## Benchmarks (1024x1024, 8 steps, M4 Pro 64GB)

### MLX vs PyTorch/MPS

| Pipeline | Total | Per Step |
|----------|-------|----------|
| PyTorch/MPS (diffusers) | 137.0s | 17.1s/step |
| **MLX (this repo)** | **134.2s** | **16.0s/step** |

### Breakdown

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
├── weights.py          # Weight loading utilities
└── convert_weights.py  # Weight conversion (for developers)
```

## Base Model

[baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) — Apache 2.0

## License

Code: MIT. Model weights: Apache 2.0 (Baidu).
