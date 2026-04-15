"""Text encoder for ERNIE-Image — PyTorch hybrid.

Uses PyTorch/transformers for text encoding (1 second, runs once),
MLX for everything else. The text encoder is <1% of total compute
so there's no meaningful speed loss from the hybrid approach.
"""

import torch
import numpy as np
import mlx.core as mx
from transformers import AutoModel, AutoTokenizer


class TextEncoder:
    """Mistral-3 text encoder via PyTorch."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def from_pretrained(model_id="baidu/ERNIE-Image-Turbo"):
        tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        model = model.to("mps")
        model.eval()
        return TextEncoder(model, tokenizer)

    @torch.no_grad()
    def encode(self, text: str) -> mx.array:
        """Encode text → mx.array [T, 3072]."""
        ids = self.tokenizer(text, add_special_tokens=True, truncation=True, padding=False)["input_ids"]
        if not ids:
            ids = [self.tokenizer.bos_token_id or 0]

        input_ids = torch.tensor([ids], device="mps")
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-2][0]  # second-to-last layer, [T, H]

        # Convert to MLX
        return mx.array(hidden.cpu().float().numpy())
