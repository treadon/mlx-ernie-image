"""ERNIE-Image MLX — Generate images from text prompts.

Text encoding: PyTorch (0.1s, runs once)
Diffusion + decode: MLX (fast on Apple Silicon)

Usage:
    python generate.py -p "A cat in space"
    python generate.py -p "A movie poster" --steps 8 --seed 42
    python generate.py --interactive
    python generate.py --convert-weights  # one-time setup
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"


def convert_weights():
    """One-time weight conversion from HuggingFace → MLX format."""
    from ernie_image.convert_weights import convert
    convert(output_dir=str(WEIGHTS_DIR))


def load_pipeline(quantize=None):
    """Load text encoder + DiT + VAE."""
    from ernie_image.pipeline import ErnieImagePipeline
    from ernie_image.text_encoder import TextEncoder

    print("Loading text encoder...")
    te = TextEncoder.from_pretrained()

    print("Loading DiT + VAE...")
    if WEIGHTS_DIR.exists():
        pipe = ErnieImagePipeline.from_weights(str(WEIGHTS_DIR), quantize=quantize)
    else:
        pipe = ErnieImagePipeline.from_pretrained("treadon/ERNIE-Image-Turbo-MLX", quantize=quantize)

    return te, pipe


def generate_image(te, pipe, prompt, height=1024, width=1024, steps=8, seed=None):
    """Encode text + generate image."""
    t0 = time.time()
    emb = te.encode(prompt)
    print(f"  Text: {time.time() - t0:.1f}s ({emb.shape[0]} tokens)")

    img = pipe.generate(
        text_embeddings=emb,
        height=height,
        width=width,
        num_inference_steps=steps,
        seed=seed,
    )
    return img


def main():
    parser = argparse.ArgumentParser(description="ERNIE-Image MLX Generator")
    parser.add_argument("--prompt", "-p", type=str, help="Text prompt")
    parser.add_argument("--steps", "-s", type=int, default=8, help="Inference steps (default: 8)")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--quantize", "-q", type=int, default=None, choices=[4, 8], help="Quantize DiT")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--convert-weights", action="store_true", help="Convert weights (one-time setup)")

    args = parser.parse_args()

    if args.convert_weights:
        convert_weights()
        return

    if not args.prompt and not args.interactive:
        parser.print_help()
        print("\nExamples:")
        print('  python generate.py -p "A cat in space"')
        print('  python generate.py -p "A movie poster" --seed 42')
        print("  python generate.py --interactive")
        return

    te, pipe = load_pipeline(quantize=args.quantize)

    if args.interactive:
        os.makedirs("outputs", exist_ok=True)
        print("\nReady. Type a prompt to generate an image.\n")
        while True:
            prompt = input("Prompt (or 'q' to quit): ").strip()
            if prompt.lower() == "q":
                break
            if not prompt:
                continue

            default_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
            filename = input(f"Filename [{default_name}]: ").strip() or default_name
            if not filename.endswith(".png"):
                filename += ".png"

            img = generate_image(te, pipe, prompt,
                                 height=args.height, width=args.width,
                                 steps=args.steps, seed=args.seed)
            path = os.path.join("outputs", filename)
            img.save(path)
            print(f"  → {path}\n")
    else:
        img = generate_image(te, pipe, args.prompt,
                             height=args.height, width=args.width,
                             steps=args.steps, seed=args.seed)
        output = args.output or datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
        img.save(output)
        print(f"  → {output}")


if __name__ == "__main__":
    main()
