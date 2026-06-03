"""Example: Train a medium GPT-2 model on Shakespeare.

This example trains a medium GPT-2 model (~4M parameters) on a sample
of Shakespeare's text. This is the optimal size for the Shakespeare corpus.

Usage:
    python -m industry_ml_lab.training.from_scratch.examples.train_medium
"""

from __future__ import annotations

import os
import numpy as np

from ..model import GPT2, GPT2Config
from ..tokenizer import BPETokenizer
from ..train import train_gpt2


def get_shakespeare_sample() -> str:
    """Get a medium-sized sample of Shakespeare text for training."""
    # Medium sample combining multiple plays
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

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms;
And then the whining school-boy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths, and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slipper'd pantaloon,
With spectacles on nose and pouch on side,
His youthful hose, well sav'd, a world too wide
For his shrunk shank; and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

—As You Like It, Act II, Scene VII

Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

—Romeo and Juliet, Act II, Scene II

Mad world! Mad kings! Mad composition!
John of Gaunt, thou art in present death;
When thou art dead, come to me as thou art,
And tell me how the world is governed now.

—Richard II, Act I, Scene IV

Beware the ides of March.

—Julius Caesar, Act I, Scene II

Something is rotten in the state of Denmark.

—Hamlet, Act I, Scene IV

The lady doth protest too much, methinks.

—Hamlet, Act III, Scene II

O, what a noble mind is here o'erthrown!

—Hamlet, Act III, Scene I

The very ecstasy of love!

—Hamlet, Act II, Scene I

Love all, trust a few, do wrong to none.

—All's Well That Ends Well, Act I, Scene II

Some are born great, some achieve greatness,
And some have greatness thrust upon them.

—Twelfth Night, Act II, Scene V

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.

—Richard III, Act I, Scene I

Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?

—Macbeth, Act II, Scene I

If you can look into the seeds of time,
And say which grain will grow and which will not,
Speak then to me, who neither beg nor fear
Your favours nor your hate.

—Macbeth, Act I, Scene III

What's in a name? That which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.

—Romeo and Juliet, Act II, Scene II

The quality of mercy is not strain'd,
It droppeth as the gentle rain from heaven
Upon the place beneath: it is twice blest;
It blesseth him that gives and him that takes:
'Tis mightiest in the mightiest: it becomes
The throned monarch better than his crown;

—The Merchant of Venice, Act IV, Scene I

Men at some time are masters of their fates:
The fault, dear Brutus, is not in our stars,
But in ourselves, that we are underlings.

—Julius Caesar, Act I, Scene II
"""
    return sample


def main():
    """Train a medium GPT-2 model."""
    print("=" * 70)
    print("  Training Medium GPT-2 on Shakespeare Sample")
    print("=" * 70)
    print()
    
    # Get training data
    text = get_shakespeare_sample()
    print(f"Training text: {len(text):,} characters, {len(text.split()):,} words")
    print()
    
    # Create tokenizer and train
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=512)
    tokenizer.train(text, verbose=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print()
    
    # Create model configuration (medium model)
    config = GPT2Config.medium(vocab_size=len(tokenizer))
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
        num_steps=2000,
        learning_rate=1e-3,
        batch_size=8,
        seq_len=128,
        warmup_fraction=0.02,
        min_lr=1e-4,
        grad_clip=1.0,
        print_every=200,
        val_fraction=0.1,
        run_dir="run_medium_shakespeare",
    )
    
    # Test generation
    print()
    print("=" * 70)
    print("  Testing Generation")
    print("=" * 70)
    print()
    
    prompt = "To be, or not to be"
    prompt_ids = np.array(tokenizer.encode(prompt))
    generated = model.generate(prompt_ids, max_tokens=50, temperature=0.8)
    generated_text = tokenizer.decode(generated.tolist())
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
    
    print("Training complete! Checkpoints saved to: run_medium_shakespeare/")


if __name__ == "__main__":
    main()
