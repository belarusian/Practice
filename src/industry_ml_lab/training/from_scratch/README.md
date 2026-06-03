# From-Scratch Transformer Implementations

This directory contains TWO variants of GPT-2 style transformer implementations:

## 1. `from_scratch/` - With PyTorch (Easier, Faster)

Uses PyTorch for tensor operations while still implementing the full training
pipeline manually. Good for:
- Faster training (uses optimized BLAS)
- Familiar PyTorch API
- Easier gradient computation

```bash
uv sync --extra from-scratch
uv run python -m industry_ml_lab.training.from_scratch.examples.train_tiny
```

## 2. `from_scratch_np/` - Pure NumPy (True From Scratch)

Uses ONLY NumPy - no deep learning frameworks at all. This is the "true from scratch"
version where you implement EVERYTHING. Good for:
- Deep educational understanding
- No framework dependencies
- See exactly how transformers work under the hood

```bash
uv sync --extra from-scratch-np
uv run python -m industry_ml_lab.training.from_scratch_np.examples.train_tiny
```

## Comparison

| Aspect | from_scratch | from_scratch_np |
|--------|--------------|-----------------|
| Framework | PyTorch + NumPy | NumPy only |
| Speed | Faster (BLAS) | Slower (pure Python) |
| Code | ~1,500 lines | ~1,500 lines |
| Learning | Practical usage | Deep theoretical |
| Use case | Quick experiments | Education/research |

## Files

Both variants have the same structure:
```
├── __init__.py          # Package exports
├── tensor.py            # Tensor class with manual operations
├── tokenizer.py         # BPE tokenizer implementation
├── model.py             # GPT-2 model architecture
├── train.py             # Training utilities and loop
├── cli.py               # CLI for training different sizes
└── examples/
    ├── train_tiny.py    # Train ~50K parameter model
    ├── train_small.py   # Train ~200K parameter model
    ├── train_medium.py  # Train ~4M parameter model
    └── train_marcus.py  # Train on Marcus Aurelius
```

## Quick Start

```bash
# Install dependencies
uv sync --extra from-scratch-np

# Train tiny model on Shakespeare
uv run python -m industry_ml_lab.training.from_scratch_np.examples.train_tiny

# Train on Marcus Aurelius
uv run python -m industry_ml_lab.training.from_scratch_np.examples.train_marcus
```

## References

- Feste (Rust): https://github.com/tag1consulting/feste
- Building an LLM From Scratch: https://www.tag1.com/how-to/part1-tokenization-building-an-llm-from-scratch-in-rust/
