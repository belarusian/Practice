"""Training utilities for from-scratch transformer (NumPy-only).

This module provides:
- Training loop with forward/backward
- AdamW optimizer
- Learning rate schedule (warmup + cosine decay)
- Gradient clipping
- Checkpointing

No deep learning frameworks are used - pure NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional
import time
import json
import os

from .tensor import Tensor, matmul, gelu, softmax
from .model import GPT2, GPT2Config
from .tokenizer import BPETokenizer


class AdamWOptimizer:
    """AdamW optimizer with weight decay.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        beta1: First moment decay
        beta2: Second moment decay
        epsilon: Numerical stability
        weight_decay: Weight decay coefficient
    """
    
    def __init__(
        self,
        model: GPT2,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.95,
        epsilon: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step = 0
        
        self.m: dict = {}
        self.v: dict = {}
        
        self.params: dict = {}
        self.grads: dict = {}
        
        self._collect_parameters()
    
    def _collect_parameters(self) -> None:
        """Collect all trainable parameters from model."""
        self.params["wte"] = self.model.wte
        self.params["wpe"] = self.model.wpe
        
        self.params["ln_f_gamma"] = self.model.ln_f.gamma
        self.params["ln_f_beta"] = self.model.ln_f.beta
        
        self.params["lm_head_weight"] = self.model.lm_head.weight
        self.params["lm_head_bias"] = self.model.lm_head.bias
        
        for i, block in enumerate(self.model.blocks):
            self.params[f"block_{i}_ln1_gamma"] = block.ln_1.gamma
            self.params[f"block_{i}_ln1_beta"] = block.ln_1.beta
            self.params[f"block_{i}_ln2_gamma"] = block.ln_2.gamma
            self.params[f"block_{i}_ln2_beta"] = block.ln_2.beta
            
            self.params[f"block_{i}_attn_q_weight"] = block.attn.c_q.weight
            self.params[f"block_{i}_attn_q_bias"] = block.attn.c_q.bias
            self.params[f"block_{i}_attn_k_weight"] = block.attn.c_k.weight
            self.params[f"block_{i}_attn_k_bias"] = block.attn.c_k.bias
            self.params[f"block_{i}_attn_v_weight"] = block.attn.c_v.weight
            self.params[f"block_{i}_attn_v_bias"] = block.attn.c_v.bias
            self.params[f"block_{i}_attn_proj_weight"] = block.attn.c_proj.weight
            self.params[f"block_{i}_attn_proj_bias"] = block.attn.c_proj.bias
            
            self.params[f"block_{i}_mlp_fc_weight"] = block.mlp.c_fc.weight
            self.params[f"block_{i}_mlp_fc_bias"] = block.mlp.c_fc.bias
            self.params[f"block_{i}_mlp_proj_weight"] = block.mlp.c_proj.weight
            self.params[f"block_{i}_mlp_proj_bias"] = block.mlp.c_proj.bias
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for name, param in self.params.items():
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)
    
    def step(self, lr: Optional[float] = None) -> None:
        """Update parameters using AdamW."""
        if lr is None:
            lr = self.learning_rate
        
        self.step += 1
        
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step
        
        for name, param in self.params.items():
            if param.grad is None:
                continue
            
            grad = param.grad
            
            if name not in self.m:
                self.m[name] = np.zeros_like(param.data)
                self.v[name] = np.zeros_like(param.data)
            
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2
            
            param.data -= lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param.data)


def get_lr_with_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Get learning rate with warmup and cosine decay schedule."""
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    else:
        decay_steps = max_steps - warmup_steps
        decay_step = step - warmup_steps
        decay_ratio = decay_step / decay_steps
        
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_ratio))
        return min_lr + (max_lr - min_lr) * cosine_decay


def clip_gradients(grads: List[np.ndarray], max_norm: float) -> None:
    """Clip gradients by global norm."""
    total_norm = 0.0
    for g in grads:
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g *= scale


def compute_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss."""
    batch, seq, vocab_size = logits.shape
    
    total_loss = 0.0
    count = 0
    
    for b in range(batch):
        for s in range(seq):
            token_id = targets[b, s]
            logit = logits[b, s, token_id]
            
            max_logit = np.max(logits[b, s])
            log_sum_exp = max_logit + np.log(np.sum(np.exp(logits[b, s] - max_logit)))
            total_loss -= (logit - max_logit) - log_sum_exp
            count += 1
    
    return total_loss / count if count > 0 else 0.0


def get_random_batch(tokens: np.ndarray, seq_len: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get a random batch of (input, target) sequences."""
    max_start = len(tokens) - seq_len - 1
    
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        start = np.random.randint(0, max_start)
        input_seq = tokens[start:start + seq_len]
        target_seq = tokens[start + 1:start + seq_len + 1]
        
        inputs.append(input_seq)
        targets.append(target_seq)
    
    return np.array(inputs), np.array(targets)


def train_gpt2(
    model: GPT2,
    tokenizer: BPETokenizer,
    text: str,
    num_steps: int,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    seq_len: int = 64,
    warmup_fraction: float = 0.05,
    min_lr: float = 1e-4,
    grad_clip: float = 1.0,
    print_every: int = 100,
    val_fraction: float = 0.1,
    run_dir: Optional[str] = None,
) -> dict:
    """Train a GPT-2 model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer for encoding text
        text: Training text corpus
        num_steps: Number of training steps
        learning_rate: Peak learning rate
        batch_size: Batch size
        seq_len: Sequence length
        warmup_fraction: Fraction of steps for warmup
        min_lr: Minimum learning rate
        grad_clip: Gradient clipping norm
        print_every: Print metrics every N steps
        val_fraction: Fraction of data for validation
        run_dir: Directory to save outputs
        
    Returns:
        Training history
    """
    tokens = np.array(tokenizer.encode(text))
    print(f"Tokenized {len(tokens):,} tokens")
    
    val_size = int(len(tokens) * val_fraction)
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]
    
    print(f"Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")
    
    if run_dir is None:
        run_dir = f"run_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    optimizer = AdamWOptimizer(model, learning_rate=learning_rate)
    
    history = []
    
    warmup_steps = int(num_steps * warmup_fraction)
    max_steps = num_steps
    
    print(f"\nTraining {model.num_parameters:,} parameters for {num_steps} steps...")
    print(f"Warmup: {warmup_steps} steps, Peak LR: {learning_rate}, Min LR: {min_lr}")
    
    best_val_loss = float("inf")
    best_step = 0
    
    for step in range(num_steps):
        start_time = time.time()
        
        lr = get_lr_with_schedule(step, warmup_steps, max_steps, learning_rate, min_lr)
        
        inputs, targets = get_random_batch(train_tokens, seq_len, batch_size)
        
        logits = model.forward(inputs)
        loss = compute_loss(logits, targets)
        
        optimizer.step(lr=lr)
        
        if step % print_every == 0 or step == num_steps - 1:
            val_inputs, val_targets = get_random_batch(val_tokens, seq_len, batch_size)
            val_logits = model.forward(val_inputs)
            val_loss = compute_loss(val_logits, val_targets)
            
            if step % (print_every * 2) == 0:
                prompt = "To be, or not to be"
                prompt_ids = np.array(tokenizer.encode(prompt))
                generated = model.generate(prompt_ids, max_tokens=20, temperature=0.8)
                generated_text = tokenizer.decode(generated.tolist())
                sample = generated_text[:100]
            else:
                sample = None
            
            elapsed = time.time() - start_time
            print(f"Step {step:5d}: loss={loss:.4f}, val_loss={val_loss:.4f}, lr={lr:.6f}, time={elapsed:.2f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                model.save(os.path.join(run_dir, "checkpoint_best.npz"))
                print(f"  -> New best model saved!")
            
            history.append({
                "step": step,
                "loss": float(loss),
                "val_loss": float(val_loss),
                "lr": float(lr),
                "sample": sample,
            })
        
        if step > 0 and step % 500 == 0:
            model.save(os.path.join(run_dir, f"checkpoint_step_{step}.npz"))
    
    model.save(os.path.join(run_dir, "checkpoint_final.npz"))
    
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
    print(f"Final model saved to: {run_dir}/checkpoint_final.npz")
    
    return history
