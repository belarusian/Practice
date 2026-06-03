"""From-scratch transformer implementation (GPT-2 style).

This module implements a complete transformer language model from scratch
using only NumPy. No deep learning frameworks are used.

Key components:
- Tensor operations (manual matmul, softmax, etc.)
- BPE tokenizer (built from scratch)
- Transformer architecture (attention, MLP, layer norm)
- Training loop (forward, backward, AdamW optimizer)

Example usage:
    from industry_ml_lab.training.from_scratch.tensor import Tensor
    from industry_ml_lab.training.from_scratch.tokenizer import BPETokenizer
    from industry_ml_lab.training.from_scratch.model import GPT2

    # Create tokenizer and train on corpus
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(text_corpus)

    # Create tiny model
    config = GPT2Config(
        vocab_size=512,
        n_embd=64,
        n_heads=1,
        n_layers=2,
        block_size=64
    )
    model = GPT2(config)

    # Train
    model.train(tokenizer, text_corpus, num_steps=1000)

"""

from __future__ import annotations

from .tensor import Tensor
from .tokenizer import BPETokenizer
from .model import GPT2, GPT2Config

__all__ = ["Tensor", "BPETokenizer", "GPT2", "GPT2Config"]
