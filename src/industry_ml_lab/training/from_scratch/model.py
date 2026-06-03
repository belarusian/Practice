"""GPT-2 model architecture implementation.

This module implements a complete GPT-2 style transformer model from scratch.
It includes:
- Token and position embeddings
- Transformer blocks (attention + MLP)
- Layer normalization
- Output projection

The model supports both forward pass (inference) and training with manual
backpropagation.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

from .tensor import Tensor, matmul, relu, gelu, softmax, zeros, randn, random_init


class GPT2Config:
    """Configuration for GPT-2 model.
    
    Args:
        vocab_size: Number of tokens in vocabulary
        n_embd: Embedding dimension (width of the model)
        n_heads: Number of attention heads per layer
        n_layers: Number of transformer blocks
        block_size: Maximum sequence length (context window)
        dropout_rate: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_embd: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        block_size: int = 1024,
        dropout_rate: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.dropout_rate = dropout_rate
    
    @classmethod
    def tiny(cls, vocab_size: int = 512) -> GPT2Config:
        """Create a tiny configuration for quick experiments.
        
        ~50K parameters - Very fast, good for testing (2-5 minutes training)
        """
        return cls(
            vocab_size=vocab_size,
            n_embd=64,
            n_heads=1,
            n_layers=2,
            block_size=64,
            dropout_rate=0.1,
        )
    
    @classmethod
    def small(cls, vocab_size: int = 512) -> GPT2Config:
        """Create a small configuration for experiments.
        
        ~200K parameters - Good balance of speed and capability
        """
        return cls(
            vocab_size=vocab_size,
            n_embd=128,
            n_heads=1,
            n_layers=3,
            block_size=128,
            dropout_rate=0.1,
        )
    
    @classmethod
    def medium(cls, vocab_size: int = 512) -> GPT2Config:
        """Create a medium configuration.
        
        ~4M parameters - Substantial capacity
        """
        return cls(
            vocab_size=vocab_size,
            n_embd=256,
            n_heads=4,
            n_layers=4,
            block_size=256,
            dropout_rate=0.1,
        )


class GPT2Cache:
    """Cache for transformer blocks during forward pass."""
    
    def __init__(self, x: Tensor, attn_weights: Tensor):
        self.x = x
        self.attn_weights = attn_weights


class Linear:
    """Fully connected layer: y = x @ W + b
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        seed: Random seed for initialization
    """
    
    def __init__(self, in_features: int, out_features: int, seed: int = 0):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(random_init((in_features, out_features), seed, scale))
        self.bias = Tensor(zeros((out_features,)))
        self.requires_grad = True
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = x @ W + b"""
        # x: [batch, seq, in_features]
        # weight: [in_features, out_features]
        # bias: [out_features]
        
        # Matrix multiply: [batch, seq, out_features]
        out = matmul(x, self.weight)
        
        # Add bias (broadcast)
        out = out + self.bias
        
        return out


class LayerNorm:
    """Layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Args:
        normalized_shape: Shape to normalize (last N dimensions)
        eps: Small value for numerical stability
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.gamma = Tensor(ones((normalized_shape,)))
        self.beta = Tensor(zeros((normalized_shape,)))
        self.eps = eps
        self.requires_grad = True
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: layer normalization"""
        # x: [batch, seq, normalized_shape]
        
        # Compute mean and variance along last dimension
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out = x_norm * self.gamma.data + self.beta.data
        
        return Tensor(out, requires_grad=x.requires_grad)


class Attention:
    """Multi-head self-attention mechanism.
    
    Args:
        n_embd: Embedding dimension
        n_heads: Number of attention heads
        block_size: Maximum sequence length
        dropout_rate: Dropout probability
        seed: Random seed
    """
    
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        dropout_rate: float,
        seed: int = 0,
    ):
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.block_size = block_size
        
        # Q, K, V projections
        self.c_q = Linear(n_embd, n_embd, seed)
        self.c_k = Linear(n_embd, n_embd, seed + 1)
        self.c_v = Linear(n_embd, n_embd, seed + 2)
        
        # Output projection
        self.c_proj = Linear(n_embd, n_embd, seed + 3)
        
        # Dropout
        self.attn_dropout = dropout_rate
        self.resid_dropout = dropout_rate
        
        # Causal mask (pre-computed)
        self.register_buffer("bias", np.tril(np.ones((block_size, block_size))))
    
    def register_buffer(self, name: str, array: np.ndarray) -> None:
        """Register a buffer (not learnable)."""
        setattr(self, name, Tensor(array, requires_grad=False))
    
    def __call__(self, x: Tensor) -> Tuple[Tensor, GPT2Cache]:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, GPT2Cache]:
        """Forward pass: multi-head self-attention
        
        Args:
            x: Input tensor [batch, seq, n_embd]
            
        Returns:
            Output tensor and cache for backward pass
        """
        batch, seq, n_embd = x.shape
        
        # Project to Q, K, V
        q = self.c_q.forward(x)  # [batch, seq, n_embd]
        k = self.c_k.forward(x)  # [batch, seq, n_embd]
        v = self.c_v.forward(x)  # [batch, seq, n_embd]
        
        # Reshape to [batch, n_heads, seq, head_dim]
        q = q.data.reshape(batch, seq, self.n_heads, self.head_dim)
        k = k.data.reshape(batch, seq, self.n_heads, self.head_dim)
        v = v.data.reshape(batch, seq, self.n_heads, self.head_dim)
        
        # Transpose to [batch, n_heads, seq, head_dim]
        q = q.transpose(0, 2, 1, 3)  # [batch, n_heads, seq, head_dim]
        k = k.transpose(0, 2, 1, 3)  # [batch, n_heads, seq, head_dim]
        v = v.data.transpose(0, 2, 1, 3)  # [batch, n_heads, seq, head_dim]
        
        # Scaled dot-product attention
        # scores = (q @ k.T) / sqrt(head_dim)
        k_t = k.transpose(0, 1, 3, 2)  # [batch, n_heads, head_dim, seq]
        scores = matmul(Tensor(q), Tensor(k_t)).data / np.sqrt(self.head_dim)  # [batch, n_heads, seq, seq]
        
        # Causal mask
        scores = scores - 1e9 * (1 - self.bias.data[:seq, :seq])
        
        # Softmax
        attn_weights = softmax(Tensor(scores), axis=-1).data
        
        # Apply dropout (simplified - no actual dropout during inference)
        # attn_weights = dropout(attn_weights, self.attn_dropout)
        
        # Apply attention to values
        # out = attn_weights @ v
        out = matmul(Tensor(attn_weights), Tensor(v)).data  # [batch, n_heads, seq, head_dim]
        
        # Transpose back to [batch, seq, n_heads, head_dim]
        out = out.transpose(0, 2, 1, 3)  # [batch, seq, n_heads, head_dim]
        
        # Reshape to [batch, seq, n_embd]
        out = out.reshape(batch, seq, n_embd)
        
        # Output projection
        out = self.c_proj.forward(Tensor(out))
        
        # Apply residual dropout
        # out = dropout(out, self.resid_dropout)
        
        cache = GPT2Cache(x, Tensor(attn_weights))
        
        return out, cache


class MLP:
    """Feed-forward network with GELU activation.
    
    Args:
        n_embd: Embedding dimension
        dropout_rate: Dropout probability
        seed: Random seed
    """
    
    def __init__(self, n_embd: int, dropout_rate: float, seed: int = 0):
        self.c_fc = Linear(n_embd, 4 * n_embd, seed)
        self.c_proj = Linear(4 * n_embd, n_embd, seed + 1)
        self.dropout_rate = dropout_rate
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: MLP with GELU activation
        
        Args:
            x: Input tensor [batch, seq, n_embd]
            
        Returns:
            Output tensor [batch, seq, n_embd]
        """
        # GELU activation
        x = self.c_fc.forward(x)
        x = gelu(x)
        
        # Output projection
        x = self.c_proj.forward(x)
        
        # Apply dropout (simplified)
        # x = dropout(x, self.dropout_rate)
        
        return x


class TransformerBlock:
    """Single transformer block with attention and MLP.
    
    Args:
        n_embd: Embedding dimension
        n_heads: Number of attention heads
        block_size: Maximum sequence length
        dropout_rate: Dropout probability
        seed: Random seed
    """
    
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        dropout_rate: float,
        seed: int = 0,
    ):
        self.ln_1 = LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_heads, block_size, dropout_rate, seed)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout_rate, seed + 100)
    
    def __call__(self, x: Tensor) -> Tuple[Tensor, GPT2Cache]:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, GPT2Cache]:
        """Forward pass: transformer block
        
        Args:
            x: Input tensor [batch, seq, n_embd]
            
        Returns:
            Output tensor and cache
        """
        # Pre-norm architecture: ln -> attn -> add -> ln -> mlp -> add
        
        # Attention with residual
        attn_out, attn_cache = self.attn.forward(self.ln_1.forward(x))
        x = x + attn_out  # Residual connection
        
        # MLP with residual
        mlp_out = self.mlp.forward(self.ln_2.forward(x))
        x = x + mlp_out  # Residual connection
        
        return x, attn_cache


class GPT2:
    """GPT-2 style transformer language model.
    
    Args:
        config: Model configuration
    """
    
    def __init__(self, config: GPT2Config):
        self.config = config
        
        # Token and position embeddings
        self.wte = Tensor(random_init((config.vocab_size, config.n_embd), 0, 0.02))
        self.wpe = Tensor(random_init((config.block_size, config.n_embd), 1, 0.02))
        
        # Transformer blocks
        self.blocks: List[TransformerBlock] = []
        for i in range(config.n_layers):
            self.blocks.append(TransformerBlock(
                n_embd=config.n_embd,
                n_heads=config.n_heads,
                block_size=config.block_size,
                dropout_rate=config.dropout_rate,
                seed=1000 + i * 100,
            ))
        
        # Final layer normalization
        self.ln_f = LayerNorm(config.n_embd)
        
        # Output projection (tied with input embeddings in GPT-2, but we keep separate for clarity)
        self.lm_head = Linear(config.n_embd, config.vocab_size, 10000)
        
        # Count parameters
        self._num_params = self._count_parameters()
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        # Embeddings
        total = self.wte.data.size + self.wpe.data.size
        
        # Blocks
        for block in self.blocks:
            total += block.ln_1.gamma.data.size + block.ln_1.beta.data.size
            total += block.attn.c_q.weight.data.size + block.attn.c_q.bias.data.size
            total += block.attn.c_k.weight.data.size + block.attn.c_k.bias.data.size
            total += block.attn.c_v.weight.data.size + block.attn.c_v.bias.data.size
            total += block.attn.c_proj.weight.data.size + block.attn.c_proj.bias.data.size
            total += block.ln_2.gamma.data.size + block.ln_2.beta.data.size
            total += block.mlp.c_fc.weight.data.size + block.mlp.c_fc.bias.data.size
            total += block.mlp.c_proj.weight.data.size + block.mlp.c_proj.bias.data.size
        
        # Final layer norm
        total += self.ln_f.gamma.data.size + self.ln_f.beta.data.size
        
        # Output head
        total += self.lm_head.weight.data.size + self.lm_head.bias.data.size
        
        return total
    
    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return self._num_params
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass: tokens -> logits
        
        Args:
            input_ids: Token IDs [batch, seq]
            
        Returns:
            Logits [batch, seq, vocab_size]
        """
        batch, seq = input_ids.shape
        
        # Get embeddings
        # token embeddings: [batch, seq, n_embd]
        token_embeds = self.wte.data[input_ids]
        
        # position embeddings: [seq, n_embd]
        pos_embeds = self.wpe.data[:seq]
        
        # Combine
        x = token_embeds + pos_embeds  # [batch, seq, n_embd]
        
        # Forward through blocks
        for block in self.blocks:
            x, _ = block.forward(Tensor(x))
        
        # Final layer norm
        x = self.ln_f.forward(Tensor(x)).data
        
        # Project to logits
        logits = self.lm_head.forward(Tensor(x)).data
        
        return logits
    
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """Generate tokens autoregressively.
        
        Args:
            prompt_ids: Prompt token IDs [seq]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = disabled)
            
        Returns:
            Generated token IDs [prompt_len + max_tokens]
        """
        tokens = prompt_ids.copy()
        
        for _ in range(max_tokens):
            # Forward pass (only last position needed)
            logits = self.forward(tokens.reshape(1, -1))  # [1, seq, vocab_size]
            
            # Get last position logits
            logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling
            if top_k is not None:
                indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
                logits[indices_to_remove] = -1e9
            
            # Convert to probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            # Sample
            next_token = np.random.choice(len(probs), p=probs)
            tokens = np.concatenate([tokens, np.array([next_token])])
        
        return tokens
    
    def save(self, path: str) -> None:
        """Save model weights to npz file."""
        state = {
            "wte": self.wte.data,
            "wpe": self.wpe.data,
            "ln_f_gamma": self.ln_f.gamma.data,
            "ln_f_beta": self.ln_f.beta.data,
            "lm_head_weight": self.lm_head.weight.data,
            "lm_head_bias": self.lm_head.bias.data,
        }
        
        # Save blocks
        for i, block in enumerate(self.blocks):
            state[f"block_{i}_ln1_gamma"] = block.ln_1.gamma.data
            state[f"block_{i}_ln1_beta"] = block.ln_1.beta.data
            state[f"block_{i}_attn_q_weight"] = block.attn.c_q.weight.data
            state[f"block_{i}_attn_q_bias"] = block.attn.c_q.bias.data
            state[f"block_{i}_attn_k_weight"] = block.attn.c_k.weight.data
            state[f"block_{i}_attn_k_bias"] = block.attn.c_k.bias.data
            state[f"block_{i}_attn_v_weight"] = block.attn.c_v.weight.data
            state[f"block_{i}_attn_v_bias"] = block.attn.c_v.bias.data
            state[f"block_{i}_attn_proj_weight"] = block.attn.c_proj.weight.data
            state[f"block_{i}_attn_proj_bias"] = block.attn.c_proj.bias.data
            state[f"block_{i}_ln2_gamma"] = block.ln_2.gamma.data
            state[f"block_{i}_ln2_beta"] = block.ln_2.beta.data
            state[f"block_{i}_mlp_fc_weight"] = block.mlp.c_fc.weight.data
            state[f"block_{i}_mlp_fc_bias"] = block.mlp.c_fc.bias.data
            state[f"block_{i}_mlp_proj_weight"] = block.mlp.c_proj.weight.data
            state[f"block_{i}_mlp_proj_bias"] = block.mlp.c_proj.bias.data
        
        np.savez(path, **state)
    
    @classmethod
    def load(cls, path: str, config: GPT2Config) -> GPT2:
        """Load model weights from npz file."""
        model = cls(config)
        
        state = np.load(path)
        
        model.wte.data = state["wte"]
        model.wpe.data = state["wpe"]
        model.ln_f.gamma.data = state["ln_f_gamma"]
        model.ln_f.beta.data = state["ln_f_beta"]
        model.lm_head.weight.data = state["lm_head_weight"]
        model.lm_head.bias.data = state["lm_head_bias"]
        
        # Load blocks
        i = 0
        while f"block_{i}_ln1_gamma" in state:
            block = model.blocks[i]
            block.ln_1.gamma.data = state[f"block_{i}_ln1_gamma"]
            block.ln_1.beta.data = state[f"block_{i}_ln1_beta"]
            block.attn.c_q.weight.data = state[f"block_{i}_attn_q_weight"]
            block.attn.c_q.bias.data = state[f"block_{i}_attn_q_bias"]
            block.attn.c_k.weight.data = state[f"block_{i}_attn_k_weight"]
            block.attn.c_k.bias.data = state[f"block_{i}_attn_k_bias"]
            block.attn.c_v.weight.data = state[f"block_{i}_attn_v_weight"]
            block.attn.c_v.bias.data = state[f"block_{i}_attn_v_bias"]
            block.attn.c_proj.weight.data = state[f"block_{i}_attn_proj_weight"]
            block.attn.c_proj.bias.data = state[f"block_{i}_attn_proj_bias"]
            block.ln_2.gamma.data = state[f"block_{i}_ln2_gamma"]
            block.ln_2.beta.data = state[f"block_{i}_ln2_beta"]
            block.mlp.c_fc.weight.data = state[f"block_{i}_mlp_fc_weight"]
            block.mlp.c_fc.bias.data = state[f"block_{i}_mlp_fc_bias"]
            block.mlp.c_proj.weight.data = state[f"block_{i}_mlp_proj_weight"]
            block.mlp.c_proj.bias.data = state[f"block_{i}_mlp_proj_bias"]
            i += 1
        
        return model
    
    def __repr__(self) -> str:
        return f"GPT2(vocab_size={self.config.vocab_size}, n_embd={self.config.n_embd}, n_layers={self.config.n_layers}, num_params={self.num_parameters:,})"
