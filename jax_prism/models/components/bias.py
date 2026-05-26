"""Learnable bias modules for output correction."""

from flax import linen as nn
import jax.numpy as jnp

from jax_prism._typing import Array


class OutputBias(nn.Module):
    """Learnable scalar bias added to output.

    A simple module that adds a learnable bias term to its input.
    Useful for output offset correction in multi-head architectures
    where the Dense layer's bias may not learn offsets effectively.

    The bias is broadcast across all dimensions except the last,
    making it suitable for sequence outputs of shape (B, T, D).

    Attributes:
        features: Number of output features. If None, inferred from input.
        initializer: Initializer for the bias. Default zeros.

    Example:
        >>> bias = OutputBias(features=1)
        >>> x = jnp.ones((2, 10, 1))  # (batch, time, features)
        >>> y = bias(x)  # Adds learned scalar bias
    """

    features: int | None = None
    initializer: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Add learnable bias to input.

        Args:
            x: Input array, shape (..., features).

        Returns:
            Output with bias added, same shape as input.
        """
        features = self.features or x.shape[-1]
        bias = self.param("bias", self.initializer, (features,))
        return x + bias


class LearnableScale(nn.Module):
    """Learnable scalar multiplier for output scaling.

    Multiplies input by a learnable scale factor, initialized to 1.
    Useful for output magnitude adjustment.

    Attributes:
        features: Number of output features. If None, inferred from input.
        initializer: Initializer for the scale. Default ones.
    """

    features: int | None = None
    initializer: nn.initializers.Initializer = nn.initializers.ones_init()

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Multiply input by learnable scale.

        Args:
            x: Input array, shape (..., features).

        Returns:
            Scaled output, same shape as input.
        """
        features = self.features or x.shape[-1]
        scale = self.param("scale", self.initializer, (features,))
        return x * scale
