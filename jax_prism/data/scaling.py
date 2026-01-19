"""Per-window scalers for time series normalization.

These scalers normalize each series independently using only
its own history, incurring zero DP cost.

All functions expect 3D input with shape (B, T, F) where:
- B: batch size
- T: time steps
- F: features (use F=1 for univariate series)

This convention ensures consistent axis semantics: time is always axis=-2.
"""
import jax.numpy as jnp

from typing import Tuple

from jax_prism._typing import Array

# Type alias for clarity
Scale = Array # Shape matches the batch dimension

def last_value_scale(x: Array) -> Tuple[Array, Scale]:
    """Scale by the last value in each series.

    Args:
        x: Input array, shape (B, T, F). Scaling uses the last time step.

    Returns:
        Tuple of (scaled_x, scale) where:
        - scaled_x: Same shape as x, normalized values.
        - scale: Shape (B, F), the scaling factors used.

    Notes:
        - If last value is 0, uses 1.0 to avoid division by zero.
        - Scale is always positive (uses absolute value).
    """
    last = jnp.take(x, -1, axis=-2) # Select only the last timestep
    abs_last = jnp.absolute(last)
    scale = jnp.where(abs_last == 0, 1.0, abs_last)
    
    scaled_x = x / jnp.expand_dims(scale, axis=-2)

    return scaled_x, scale


def median_scale(x: Array, k: int | None = None) -> Tuple[Array, Scale]:
    """Scale by the median of the last k values.

    Args:
        x: Input array, shape (B, T, F).
        k: Number of trailing time steps to use. If None, uses all.

    Returns:
        Tuple of (scaled_x, scale) where:
        - scaled_x: Same shape as x, normalized values.
        - scale: Shape (B, F), the scaling factors used.

    Notes:
        - More robust to outliers than last_value_scale.
        - If median is zero, uses 1.0 to avoid division by zero.
    """
    # 1. Slice last k timesteps along time axis (-2)
    if k is None:
        x_slice = x
    
    else:
        x_slice = jnp.take(x, jnp.arange(-k, 0), axis=-2)
    
    # 2. Compute median along time axis
    median = jnp.median(x_slice, axis=-2) # (B, F)
    
    # 3. Take absolute value
    med_abs = jnp.absolute(median)
    
    # 4. Replace zeros with 1.0
    scale = jnp.where(med_abs == 0, 1.0, med_abs)
    
    # 5. Broadcast and divide
    scaled_x = x / jnp.expand_dims(scale, axis=-2)

    return scaled_x, scale


def fixed_scale(x: Array, scale: Scale) -> Array:
    """Scale by a user-provided fixed value.

    Args:
        x: Input array, shape (B, T, F).
        scale: Fixed scale values, shape (B, F).

    Returns:
        Scaled array with same shape as x.

    Notes:
        - Use when you have prior knowledge of appropriate scale.
        - No scale is returned since it's provided as input.
    """
    return x / jnp.expand_dims(scale, axis=-2)


def inverse_scale(scaled_x: Array, scale: Scale) -> Array:
    """Reverse the scaling transformation.

    Args:
        scaled_x: Scaled array from any of the scale functions, shape (B, T, F).
        scale: The scale values returned from scaling, shape (B, F).

    Returns:
        Original-scale array with same shape as scaled_x.
    """
    return scaled_x * jnp.expand_dims(scale, axis=-2)