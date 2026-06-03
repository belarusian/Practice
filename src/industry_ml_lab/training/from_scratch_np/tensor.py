"""Tensor operations for neural networks (NumPy-only implementation).

This module provides a minimal Tensor class built on top of NumPy.
It's designed specifically for transformer models with support for:
- Matrix multiplication
- Softmax
- Broadcasting
- Gradient computation

No deep learning frameworks are used - pure NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import Union, List, Tuple, Optional


class Tensor:
    """A multi-dimensional array for neural network computations.
    
    This is a thin wrapper around NumPy arrays with additional methods
    for neural network operations. All gradients are stored separately
    and computed manually (no autograd).
    
    Args:
        data: NumPy array or list of values
        requires_grad: Whether to track gradients for this tensor
    """
    
    def __init__(self, data: Union[np.ndarray, List], requires_grad: bool = False):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)
        else:
            raise TypeError(f"Expected np.ndarray or list, got {type(data)}")
        
        self.data: np.ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        
        if requires_grad:
            self.grad = np.zeros_like(data)
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array."""
        return self.data
    
    def zero_grad(self) -> None:
        """Zero out the gradient."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.grad)
    
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Compute gradients using the chain rule.
        
        Args:
            grad: Gradient from upstream (default: ones like output)
        """
        if grad is None:
            grad = np.ones_like(self.data)
        
        self.grad = grad
    
    # ========================================================================
    # Arithmetic Operations
    # ========================================================================
    
    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        if isinstance(other, Tensor):
            return add(self, other)
        return add(self, Tensor(other))
    
    def __radd__(self, other: Union[Tensor, float]) -> Tensor:
        return self.__add__(other)
    
    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        if isinstance(other, Tensor):
            return sub(self, other)
        return sub(self, Tensor(other))
    
    def __rsub__(self, other: Union[Tensor, float]) -> Tensor:
        return sub(Tensor(other), self)
    
    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        if isinstance(other, Tensor):
            return mul(self, other)
        return mul(self, Tensor(other))
    
    def __rmul__(self, other: Union[Tensor, float]) -> Tensor:
        return self.__mul__(other)
    
    def __matmul__(self, other: Tensor) -> Tensor:
        return matmul(self, other)
    
    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        if isinstance(other, Tensor):
            return div(self, other)
        return div(self, Tensor(other))
    
    def __neg__(self) -> Tensor:
        return neg(self)
    
    # ========================================================================
    # Shape Operations
    # ========================================================================
    
    def reshape(self, *shape: int) -> Tensor:
        """Reshape the tensor."""
        return reshape(self, shape)
    
    def transpose(self, *axes: int) -> Tensor:
        """Transpose the tensor."""
        return transpose(self, axes)
    
    def squeeze(self, axis: Optional[int] = None) -> Tensor:
        """Remove axes of length one."""
        return squeeze(self, axis)
    
    def unsqueeze(self, axis: int) -> Tensor:
        """Add a dimension of length one."""
        return unsqueeze(self, axis)
    
    def flatten(self) -> Tensor:
        """Flatten to 1D."""
        return flatten(self)
    
    # ========================================================================
    # Activation Functions
    # ========================================================================
    
    def relu(self) -> Tensor:
        """ReLU activation."""
        return relu(self)
    
    def gelu(self) -> Tensor:
        """GELU activation (approximate)."""
        return gelu(self)
    
    def softmax(self, axis: int = -1) -> Tensor:
        """Softmax along axis."""
        return softmax(self, axis)
    
    def sigmoid(self) -> Tensor:
        """Sigmoid activation."""
        return sigmoid(self)


# ========================================================================
# Function Implementations
# ========================================================================

def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition."""
    result = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            if a.requires_grad:
                a_grad = grad
                while a_grad.ndim > a.ndim:
                    a_grad = np.sum(a_grad, axis=0)
                for i in range(a.ndim):
                    if a.shape[i] == 1 and a_grad.shape[i] > 1:
                        a_grad = np.sum(a_grad, axis=i, keepdims=True)
                a.grad += a_grad
            
            if b.requires_grad:
                b_grad = grad
                while b_grad.ndim > b.ndim:
                    b_grad = np.sum(b_grad, axis=0)
                for i in range(b.ndim):
                    if b.shape[i] == 1 and b_grad.shape[i] > 1:
                        b_grad = np.sum(b_grad, axis=i, keepdims=True)
                b.grad += b_grad
        
        result.backward = grad_fn
    
    return result


def sub(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise subtraction."""
    return a + (-b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication."""
    result = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            if a.requires_grad:
                a.grad += grad * b.data
            if b.requires_grad:
                b.grad += grad * a.data
        
        result.backward = grad_fn
    
    return result


def div(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division."""
    result = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            if a.requires_grad:
                a.grad += grad / b.data
            if b.requires_grad:
                b.grad += grad * (-a.data / (b.data ** 2))
        
        result.backward = grad_fn
    
    return result


def neg(a: Tensor) -> Tensor:
    """Negation."""
    result = Tensor(-a.data, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            a.grad += -grad
        
        result.backward = grad_fn
    
    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication."""
    result = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            if a.requires_grad:
                if b.ndim == 1:
                    a.grad += grad.reshape(-1, 1) @ b.data.reshape(1, -1)
                else:
                    a.grad += grad @ b.data.T
            
            if b.requires_grad:
                if a.ndim == 1:
                    a_data = a.data.reshape(-1, 1)
                    grad_reshaped = grad.reshape(1, -1)
                    b.grad += a_data @ grad_reshaped
                else:
                    b.grad += a.data.T @ grad
        
        result.backward = grad_fn
    
    return result


def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape the tensor."""
    result = Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)
    return result


def transpose(a: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Transpose the tensor."""
    result = Tensor(a.data.transpose(axes), requires_grad=a.requires_grad)
    return result


def squeeze(a: Tensor, axis: Optional[int] = None) -> Tensor:
    """Remove axes of length one."""
    if axis is None:
        result = Tensor(a.data.squeeze(), requires_grad=a.requires_grad)
    else:
        result = Tensor(a.data.squeeze(axis=axis), requires_grad=a.requires_grad)
    return result


def unsqueeze(a: Tensor, axis: int) -> Tensor:
    """Add a dimension of length one."""
    result = Tensor(np.expand_dims(a.data, axis=axis), requires_grad=a.requires_grad)
    return result


def flatten(a: Tensor) -> Tensor:
    """Flatten to 1D."""
    return reshape(a, (-1,))


def relu(a: Tensor) -> Tensor:
    """ReLU activation: max(0, x)."""
    result = Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            a.grad += grad * (a.data > 0)
        
        result.backward = grad_fn
    
    return result


def gelu(a: Tensor) -> Tensor:
    """GELU activation (approximate using tanh).
    
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_pi = np.sqrt(2 / np.pi)
    x = a.data
    approx = 0.5 * x * (1 + np.tanh(sqrt_2_pi * (x + 0.044715 * x ** 3)))
    result = Tensor(approx, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            tanh_arg = sqrt_2_pi * (x + 0.044715 * x ** 3)
            tanh_val = np.tanh(tanh_arg)
            sech_sq = 1 - tanh_val ** 2
            
            grad_input = 0.5 * (1 + tanh_val) + 0.5 * x * sech_sq * sqrt_2_pi * (1 + 0.134145 * x ** 2)
            a.grad += grad * grad_input
        
        result.backward = grad_fn
    
    return result


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation."""
    x = a.data
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    softmax_x = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    result = Tensor(softmax_x, requires_grad=a.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            s = result.data
            grad_input = s * (grad - np.sum(grad * s, axis=axis, keepdims=True))
            a.grad += grad_input
        
        result.backward = grad_fn
    
    return result


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    result = Tensor(1 / (1 + np.exp(-a.data)), requires_grad=a.requires_grad)
    
    if result.requires_grad:
        def grad_fn(grad):
            s = result.data
            a.grad += grad * s * (1 - s)
        
        result.backward = grad_fn
    
    return result


def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """Create a tensor of zeros."""
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """Create a tensor of ones."""
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """Create a tensor with random values from N(0, 1)."""
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def rand(shape: Tuple[int, ...], requires_grad: bool = False) -> Tensor:
    """Create a tensor with random values from U(0, 1)."""
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)


def random_init(shape: Tuple[int, ...], seed: int, scale: float = 1.0) -> np.ndarray:
    """Initialize weights with scaled random values.
    
    Args:
        shape: Output shape
        seed: Random seed
        scale: Scaling factor (typically 1/sqrt(fan_in) or similar)
    
    Returns:
        NumPy array with initialized values
    """
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32) * scale
