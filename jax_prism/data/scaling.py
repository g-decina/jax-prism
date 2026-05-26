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

def _validate_3d(x: Array) -> None:
    """Validates whether an Array is 3D or not."""
    if x.ndim != 3:
        raise ValueError(
            f"Series must be 3D (B, T, F), got {x.ndim}D with shape {x.shape}. "
            f"For univariate series, use x[..., None] to add a feature dimension."
        )
    
    return None

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
    _validate_3d(x)
    
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
    _validate_3d(x)
    
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
    _validate_3d(x)
    
    return x / jnp.expand_dims(scale, axis=-2)


def inverse_scale(scaled_x: Array, scale: Scale) -> Array:
    """Reverse the scaling transformation.

    Args:
        scaled_x: Scaled array, shape (B, T) or (B, T, F).
            2D arrays are handled gracefully (common after squeezing predictions).
        scale: The scale values returned from scaling.
            Shape (B,) or (B, 1) for 2D input, (B, F) for 3D input.

    Returns:
        Original-scale array with same shape as scaled_x.

    Raises:
        ValueError: If scaled_x is not 2D or 3D.
    """
    if scaled_x.ndim == 2:
        # Handle squeezed predictions: (B, T) with scale (B,) or (B, 1)
        # Flatten scale to (B,) then expand to (B, 1) for broadcasting
        scale_flat = scale.reshape(scale.shape[0])
        return scaled_x * jnp.expand_dims(scale_flat, axis=-1)
    elif scaled_x.ndim == 3:
        # Standard 3D case: (B, T, F) with scale (B, F)
        return scaled_x * jnp.expand_dims(scale, axis=-2)
    else:
        raise ValueError(
            f"scaled_x must be 2D (B, T) or 3D (B, T, F), "
            f"got {scaled_x.ndim}D with shape {scaled_x.shape}."
        )


def window_median_scale(
    past: Array,
    future: Array,
) -> Tuple[Array, Array, Scale]:
    """Scale past and future by the median of the combined window.

    Unlike median_scale which only uses past values, this function computes
    the scale from both past and future targets. This ensures both encoder
    inputs and decoder targets are centered around 1.0, eliminating
    systematic bias when past and future have different distributions
    (e.g., due to day-of-week effects).

    Args:
        past: Past targets, shape (B, T_enc, F).
        future: Future targets, shape (B, T_dec, F).

    Returns:
        Tuple of (scaled_past, scaled_future, scale) where:
        - scaled_past: Same shape as past, normalized values.
        - scaled_future: Same shape as future, normalized values.
        - scale: Shape (B, F), the scaling factors used.

    Notes:
        - Uses median for robustness to outliers.
        - If median is zero, uses 1.0 to avoid division by zero.
        - During inference, use the training scale or estimate from past only.
    """
    _validate_3d(past)
    _validate_3d(future)
    
    # Concatenate along time axis
    combined = jnp.concatenate([past, future], axis=-2)

    # Compute median along time axis
    median = jnp.median(combined, axis=-2)  # (B, F)

    # Take absolute value and handle zeros
    med_abs = jnp.absolute(median)
    scale = jnp.where(med_abs == 0, 1.0, med_abs)

    # Scale both arrays
    scale_expanded = jnp.expand_dims(scale, axis=-2)
    scaled_past = past / scale_expanded
    scaled_future = future / scale_expanded

    return scaled_past, scaled_future, scale


def window_mean_scale(
    past: Array,
    future: Array,
) -> Tuple[Array, Array, Scale]:
    """Scale past and future by the mean of the combined window.

    Similar to window_median_scale but uses mean instead of median.
    Less robust to outliers but provides exact centering.

    Args:
        past: Past targets, shape (B, T_enc, F).
        future: Future targets, shape (B, T_dec, F).

    Returns:
        Tuple of (scaled_past, scaled_future, scale) where:
        - scaled_past: Same shape as past, normalized values.
        - scaled_future: Same shape as future, normalized values.
        - scale: Shape (B, F), the scaling factors used.
    """
    _validate_3d(past)
    _validate_3d(future)
    
    # Concatenate along time axis
    combined = jnp.concatenate([past, future], axis=-2)

    # Compute mean along time axis
    mean = jnp.mean(combined, axis=-2)  # (B, F)

    # Take absolute value and handle zeros
    mean_abs = jnp.absolute(mean)
    scale = jnp.where(mean_abs == 0, 1.0, mean_abs)

    # Scale both arrays
    scale_expanded = jnp.expand_dims(scale, axis=-2)
    scaled_past = past / scale_expanded
    scaled_future = future / scale_expanded

    return scaled_past, scaled_future, scale