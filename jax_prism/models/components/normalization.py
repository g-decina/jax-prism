import jax.numpy as jnp
from flax import linen as nn

from jax_prism._typing import Array

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Attributes:
        epsilon: Small constant for numerical stability.
    """
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        features = x.shape[-1]
        
        mean = jnp.mean(x ** 2, axis=-1, keepdims=True)
        rms = jnp.sqrt(mean + self.epsilon)
        
        scale = self.param("scale", nn.initializers.ones, (features, ))
        
        return (x / rms) * scale