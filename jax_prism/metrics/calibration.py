"""Calibration metrics for probabilistic forecasts."""

import jax.numpy as jnp

from jax_prism._typing import Array


def quantile_calibration_error(
    q_values: Array,
    q_levels: Array,
    y_true: Array,
    mask: Array | None = None,
) -> Array:
    """Compute Quantile Calibration Error (QCE).

    Measures how well predicted quantiles match empirical coverage. A perfectly
    calibrated model has QCE = 0, meaning the q-th quantile prediction exceeds
    exactly q fraction of observations.

    Args:
        q_values: Predicted quantile values, shape (B, T, Q).
        q_levels: Quantile levels in (0, 1), shape (Q,). E.g., [0.1, 0.5, 0.9].
        y_true: Ground truth values, shape (B, T, 1).
        mask: Optional validity mask, shape (B, T, 1). 1=valid, 0=ignore.

    Returns:
        Scalar QCE value in [0, 1]. Lower is better.

    Example:
        >>> q_levels = jnp.array([0.1, 0.5, 0.9])
        >>> q_values = model.predict_quantiles(x)  # (B, T, 3)
        >>> qce = quantile_calibration_error(q_values, q_levels, y_true)
    """
    if q_values.shape[-1] != q_levels.shape[0]:
        raise ValueError(
            f"q_values last dim ({q_values.shape[-1]}) must match "
            f"q_levels length ({q_levels.shape[0]})."
        )

    # indicators[b, t, q] = 1 if y_true[b, t] < q_values[b, t, q]
    indicators = (y_true < q_values).astype(jnp.float32)

    if mask is not None:
        indicators = indicators * mask.astype(jnp.float32)
        N = mask.sum()
    else:
        N = q_values.shape[0] * q_values.shape[1]  # B * T

    # Empirical coverage per quantile level
    p_hat = indicators.sum(axis=(0, 1)) / N  # (Q,)

    # Mean absolute calibration error
    qce = jnp.abs(q_levels - p_hat).mean()
    return qce