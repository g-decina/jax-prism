import jax.numpy as jnp

from jax_prism._typing import Array, Tuple

def build_rope_frequencies(
        dim: int,
        max_seq_len: int,
        base: float = 10_000.0
    ) -> Tuple[Array, Array]:
    """Precompute RoPE frequency tensor.
    
    Args:
        dim: Embedding dimension (must be even).
        max_seq_len: Maximum sequence length to support.
        base: Base for frequency computation (default 10000).
    
    Returns:
        Complex exponentials of shape (max_seq_len, dim // 2).
        Real part is cos, imaginary part is sin.
    """
    i = jnp.arange(dim // 2)
    theta = base ** (-2 * i / dim)
    
    positions = jnp.arange(max_seq_len) # (max_seq_len, )
    angles = positions[:, None] * theta[None, :] # (max_seq_len, dim // 2)
    
    return jnp.cos(angles), jnp.sin(angles)
    
def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """Apply rotary position embedding to input.
    
    Args:
        x: Input tensor of shape (..., seq_len, dim).
        cos: Cosine frequencies, shape (seq_len, dim // 2) or broadcastable.
        sin: Sine frequencies, shape (seq_len, dim // 2) or broadcastable.
    
    Returns:
        Rotated tensor, same shape as x.
    """
    dim = x.shape[-1] // 2
    x1 = x[..., :dim]
    x2 = x[..., dim:]
    
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    cos = jnp.concatenate([cos, cos], axis=-1)
    sin = jnp.concatenate([sin, sin], axis=-1)
    
    return x * cos + rotated * sin