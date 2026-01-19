"""Probabilistic forecast metrics: quantile loss, coverage."""

import jax.numpy as jnp
from jax_prism._typing import Array


def quantile_loss(
    y_true: Array,
    y_pred: Array,
    quantiles: Array,
    mask: Array | None = None,
) -> Array:
    """Pinball loss for quantile forecasts.
    
    Args:
        y_true: Ground truth, shape (B, T, F).
        y_pred: Quantile predictions, shape (B, T, F, Q) where Q = len(quantiles).
        quantiles: Quantile levels, shape (Q,), values in (0, 1).
        mask: Optional mask, shape broadcastable to (B, T, F).
    
    Returns:
        Scalar mean quantile loss.
    
    Notes:
        For each quantile q:
        L_q(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)
        
        Equivalent to:
        L_q(y, ŷ) = (y - ŷ) * (q - 𝟙[y < ŷ])
    """
    # 1. Expand y_true to match y_pred: (B, T, F) -> (B, T, F, 1)
    y_true_expanded = jnp.expand_dims(y_true, -1)   # (B, T, F, 1)
    
    # 2. Compute errors
    errors = y_true_expanded - y_pred      # (B, T, F, Q)
    
    # 3. Compute indicator: errors < 0
    indicator = (errors < 0)
    
    # 4. Pinball: errors * (quantiles - indicator)
    pinball = errors * (quantiles - indicator)
    
    # 5. Apply mask, if any
    if mask is not None:
        mask_expanded = jnp.expand_dims(mask, -1)
        pinball = pinball * mask_expanded
        return pinball.sum() / (mask.sum() * quantiles.shape[0])
    
    # 6. Return mean
    return jnp.mean(pinball)


def coverage(
    y_true: Array,
    lower: Array,
    upper: Array,
    mask: Array | None = None,
) -> Array:
    """Empirical coverage of prediction intervals.
    
    Args:
        y_true: Ground truth, shape (B, T, F).
        lower: Lower bound of interval, same shape as y_true.
        upper: Upper bound of interval, same shape as y_true.
        mask: Optional mask.
    
    Returns:
        Scalar coverage rate in [0, 1].
    
    Notes:
        Coverage = mean(𝟙[lower ≤ y_true ≤ upper])
    """
    # 1. Compute indicator: (y_true >= lower) & (y_true <= upper)
    in_interval = (y_true >= lower) & (y_true <= upper)
    
    # 2. Convert to float
    in_interval = in_interval.astype(jnp.float32)
    
    # 3. Apply mask if provided
    if mask is not None:
        in_interval = in_interval * mask
        return in_interval.sum() / mask.sum()
    
    # 4. Return mean
    return jnp.mean(in_interval)
