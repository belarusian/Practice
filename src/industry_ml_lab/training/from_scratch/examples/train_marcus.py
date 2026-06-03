"""Example: Train on Marcus Aurelius Meditations.

This example trains a GPT-2 model on Marcus Aurelius' Meditations.
It demonstrates how to adapt the from-scratch framework to a different domain.

Usage:
    python -m industry_ml_lab.training.from_scratch.examples.train_marcus
"""

from __future__ import annotations

import os
import numpy as np

from ..model import GPT2, GPT2Config
from ..tokenizer import BPETokenizer
from ..train import train_gpt2


def get_marcus_text() -> str:
    """Get Marcus Aurelius Meditations text."""
    # Sample of Marcus Aurelius-style philosophical text
    sample = """
The happiness of your life depends upon the quality of your thoughts.
Time is a sort of river of passing events, and strong is its current;
no sooner is a thing brought to sight than it is carried past,
and another brings it in, and this too will be carried past.

Accept whatever alights in your life as made by the gods,
and further, if you can, as made for you.
No power is able to prevent what is agreeable to the gods,
nor can anything happen that is not for the advantage of the whole,
and the advantage of the universe is what the gods intend.

Look back on the past, on all the generations that have passed away,
and their troubles, and their disturbances, and their disputes,
and all the things they were anxious about, and all the things they desired,
and all the things they enjoyed, and where are they now?

The universe is change; our life is what our thoughts make it.
You have power over your mind - not outside events. Realize this,
and you will find strength.

He who lives in harmony with himself lives in harmony with the universe.

Waste no more time arguing about what a good man should be.
Be one.

The happiness of your life depends upon the quality of your thoughts:
for mind governs man.

Very little is needed to make a happy life; it is all within yourself,
in your way of thinking.

If it is not right, do not do it; if it is not true, do not say it.

The impediment to action advances action. What stands in the way
becomes the way.

You have power over your mind - not outside events. Realize this,
and you will find strength.

Don't explain your philosophy. Embody it.

The soul becomes dyed with the color of its thoughts.

Love only that which happens to you and is woven into your destiny.
For what could be more in harmony?

When you arise in the morning, think of what a precious privilege
it is to be alive, to breathe, to think, to enjoy, to love.

Only a fraction of the book you read is useful. How much more useless
is the reading of multiple books!

Do not act as if you were going to live ten thousand years.
Waste of time is the greatest waste of all.

The best way of avenging yourself is not to become like the wrongdoer.

Take care that you are not made into a Caesar, that you are not
dyed with this dye. For it will happen.

The object of life is not to be on the side of the majority,
but to escape finding oneself in the ranks of the insane.

Marcus Aurelius, Meditations
"""
    return sample


def main():
    """Train a GPT-2 model on Marcus Aurelius."""
    print("=" * 70)
    print("  Training GPT-2 on Marcus Aurelius Meditations")
    print("=" * 70)
    print()
    
    # Get training data
    text = get_marcus_text()
    print(f"Training text: {len(text):,} characters, {len(text.split()):,} words")
    print()
    
    # Create tokenizer and train
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(text, verbose=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print()
    
    # Create model configuration (tiny model for quick demo)
    config = GPT2Config.tiny(vocab_size=len(tokenizer))
    print(f"Model configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  block_size: {config.block_size}")
    print()
    
    # Create model
    model = GPT2(config)
    print(f"Model: {model.num_parameters:,} parameters")
    print()
    
    # Train
    history = train_gpt2(
        model=model,
        tokenizer=tokenizer,
        text=text,
        num_steps=500,  # Small for quick demo
        learning_rate=1e-3,
        batch_size=2,
        seq_len=32,
        warmup_fraction=0.1,
        min_lr=1e-4,
        grad_clip=1.0,
        print_every=50,
        val_fraction=0.2,
        run_dir="run_marcus_aurelius",
    )
    
    # Test generation
    print()
    print("=" * 70)
    print("  Testing Generation")
    print("=" * 70)
    print()
    
    prompt = "The happiness of your life"
    prompt_ids = np.array(tokenizer.encode(prompt))
    generated = model.generate(prompt_ids, max_tokens=30, temperature=0.8)
    generated_text = tokenizer.decode(generated.tolist())
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
    
    print("Training complete! Checkpoints saved to: run_marcus_aurelius/")


if __name__ == "__main__":
    main()
