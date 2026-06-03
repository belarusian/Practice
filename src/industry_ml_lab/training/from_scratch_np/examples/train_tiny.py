"""Example: Train a tiny GPT-2 model on Shakespeare (NumPy-only).

This example trains a small GPT-2 model (~50K parameters) on a sample
of Shakespeare's text using ONLY NumPy. No deep learning frameworks.

Usage:
    python -m industry_ml_lab.training.from_scratch_np.examples.train_tiny
"""

from __future__ import annotations

import os
import numpy as np

from ..model import GPT2, GPT2Config
from ..tokenizer import BPETokenizer
from ..train import train_gpt2


def get_shakespeare_sample() -> str:
    """Get a sample of Shakespeare text for training."""
    sample = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep—
No more—and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.

For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovere'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?

Thus conscience doth make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment,
With this regard their currents turn awry
And lose the name of action.

—Hamlet, Act III, Scene I
"""
    return sample


def main():
    """Train a tiny GPT-2 model."""
    print("=" * 70)
    print("  Training Tiny GPT-2 on Shakespeare Sample (NumPy-only)")
    print("=" * 70)
    print()
    
    text = get_shakespeare_sample()
    print(f"Training text: {len(text):,} characters, {len(text.split()):,} words")
    print()
    
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(text, verbose=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print()
    
    config = GPT2Config.tiny(vocab_size=len(tokenizer))
    print(f"Model configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  block_size: {config.block_size}")
    print()
    
    model = GPT2(config)
    print(f"Model: {model.num_parameters:,} parameters")
    print()
    
    history = train_gpt2(
        model=model,
        tokenizer=tokenizer,
        text=text,
        num_steps=500,
        learning_rate=1e-3,
        batch_size=2,
        seq_len=32,
        warmup_fraction=0.1,
        min_lr=1e-4,
        grad_clip=1.0,
        print_every=50,
        val_fraction=0.2,
        run_dir="run_tiny_shakespeare_np",
    )
    
    print()
    print("=" * 70)
    print("  Testing Generation")
    print("=" * 70)
    print()
    
    prompt = "To be, or not to be"
    prompt_ids = np.array(tokenizer.encode(prompt))
    generated = model.generate(prompt_ids, max_tokens=20, temperature=0.8)
    generated_text = tokenizer.decode(generated.tolist())
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
    
    print("Training complete! Checkpoints saved to: run_tiny_shakespeare_np/")


if __name__ == "__main__":
    main()
