"""Point forecast metrics: MAE, SMAPE, MASE."""

import jax.numpy as jnp
from jax_prism._typing import Array


def mae(y_true: Array, y_pred: Array, mask: Array | None = None) -> Array:
    """Mean Absolute Error.
    
    Args:
        y_true: Ground truth, shape (B, T, F) or any broadcastable shape.
        y_pred: Predictions, same shape as y_true.
        mask: Optional mask, 1=valid, 0=ignore. Broadcastable to y_true.
    
    Returns:
        Scalar MAE value.
    
    Notes:
        MAE = mean(|y_true - y_pred|)
    """
    # 1. Compute absolute errors
    ae = jnp.abs(y_true - y_pred)
    
    # 2. Apply mask, if provided
    if mask is not None:
        ae = ae * mask
        return ae.sum() / mask.sum()
    
    return ae.sum() / ae.size


def smape(y_true: Array, y_pred: Array, mask: Array | None = None) -> Array:
    """Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        mask: Optional mask.
    
    Returns:
        Scalar SMAPE value in range [0, 200].
    
    Notes:
        SMAPE = 100 * mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))
        
        When both y_true and y_pred are zero, that term contributes 0.
    """
    # 1. Compute numerator: |y_true - y_pred|
    ae = jnp.abs(y_true - y_pred)
    
    # 2. Compute denominator: (|y_true| + |y_pred|) / 2
    denom = (jnp.abs(y_true) + jnp.abs(y_pred)) / 2
    
    # 3. Handle zero denominator
    
    ratio = jnp.where(denom == 0, 0.0, ae / denom) * 100
    
    if mask is not None:
        ratio = ratio * mask
        return ratio.sum() / mask.sum()
    
    return ratio.sum() / ratio.size


def mase(
    y_true: Array,
    y_pred: Array,
    y_train: Array,
    seasonality: int = 1,
    mask: Array | None = None,
) -> Array:
    """Mean Absolute Scaled Error.
    
    Args:
        y_true: Ground truth for evaluation period.
        y_pred: Predictions for evaluation period.
        y_train: Training data used to compute naive forecast error.
        seasonality: Seasonal period for naive forecast. Default 1 (random walk).
        mask: Optional mask for y_true/y_pred.
    
    Returns:
        Scalar MASE value. Values < 1 beat the naive forecast.
    
    Notes:
        MASE = MAE(y_true, y_pred) / MAE_naive
        
        MAE_naive is computed on training data as:
        mean(|y_train[t] - y_train[t - seasonality]|) for t >= seasonality
    """
    # 1. Compute MAE of predictions
    mae_pred = mae(y_true, y_pred, mask)
    
    # 2. Compute naive forecast error on training data
    naive_errors = jnp.abs(y_train[:, seasonality:, :] - y_train[:, :-seasonality, :])
    mae_naive = jnp.mean(naive_errors)
    mae_naive = jnp.where(mae_naive == 0, 1e-6, mae_naive)
    
    return mae_pred / mae_naive