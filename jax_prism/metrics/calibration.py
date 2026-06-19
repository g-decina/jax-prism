"""Calibration metrics for probabilistic forecasts."""

import jax.numpy as jnp

from jax_prism._typing import Array


def pit_histogram(
    pit_values: Array,
    num_bins: int = 10,
    mask: Array | None = None,
) -> tuple[Array, Array]:
    """Compute histogram of PIT values for calibration diagnostics.

    The Probability Integral Transform (PIT) maps observations through the
    predictive CDF: PIT = F(y_true). For a well-calibrated model, PIT values
    should be uniformly distributed on [0, 1], producing a flat histogram.

    Deviations from uniformity indicate miscalibration:
    - U-shaped: underdispersed (intervals too narrow)
    - Inverse-U: overdispersed (intervals too wide)
    - Skewed: biased location

    Args:
        pit_values: CDF values F(y_true), shape (...). Values in [0, 1].
            Compute via `distribution.cdf(params, y_true)` or
            `jax.scipy.stats.norm.cdf(y_true, loc=mu, scale=sigma)`.
        num_bins: Number of histogram bins. Default 10.
        mask: Optional mask, same shape as pit_values. 1=valid, 0=ignore.

    Returns:
        Tuple of (bin_edges, counts):
        - bin_edges: Array of shape (num_bins + 1,) with bin boundaries [0, 1]
        - counts: Array of shape (num_bins,) with normalized frequencies.
            Perfect calibration → all counts ≈ 1/num_bins.

    Example:
        >>> # For Gaussian predictions
        >>> params = gaussian_head.params_from_raw(predictions)
        >>> pit = gaussian_head.cdf(params, y_true)
        >>> edges, counts = pit_histogram(pit)
        >>> # Plot: plt.bar(edges[:-1], counts, width=1/num_bins)
    """
    # Flatten to 1D
    flat_pit = pit_values.ravel()

    # Bin edges uniformly spaced on [0, 1]
    bin_edges = jnp.linspace(0.0, 1.0, num_bins + 1)

    # Assign each value to a bin index in [0, num_bins-1]
    # Values at exactly 1.0 go to the last bin
    bin_idx = jnp.clip(
        jnp.floor(flat_pit * num_bins).astype(jnp.int32),
        0,
        num_bins - 1,
    )

    # Weights: 1 for valid, 0 for masked
    if mask is not None:
        weights = mask.ravel().astype(jnp.float32)
    else:
        weights = jnp.ones_like(flat_pit)

    # Accumulate weighted counts per bin
    counts = jnp.zeros(num_bins, dtype=jnp.float32)
    counts = counts.at[bin_idx].add(weights)

    # Normalize to frequencies (sum to 1)
    total = jnp.maximum(weights.sum(), 1.0)
    counts = counts / total

    return bin_edges, counts


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