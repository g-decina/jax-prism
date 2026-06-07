"""Continuous Ranked Probability Score (CRPS)."""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array


def crps_gaussian(
    y_true: Array,
    mu: Array,
    sigma: Array,
    mask: Array | None = None,
) -> Array:
    """CRPS for Gaussian predictive distribution.

    The Continuous Ranked Probability Score is a proper scoring rule that
    generalizes MAE to full distributions. It rewards both calibration
    (correct coverage) and sharpness (tight intervals).

    Closed-form for N(μ, σ²):
        CRPS = σ · [z·(2Φ(z) - 1) + 2φ(z) - 1/√π]

    where z = (y - μ)/σ, Φ = standard normal CDF, φ = standard normal PDF.

    Args:
        y_true: Ground truth values, shape (B, T, 1).
        mu: Predicted mean, same shape as y_true.
        sigma: Predicted standard deviation (positive), same shape as y_true.
        mask: Optional validity mask, shape (B, T, 1). 1=valid, 0=ignore.

    Returns:
        Scalar mean CRPS. Lower is better.
    """
    z = (y_true - mu) / sigma
    phi_z = jax.scipy.stats.norm.pdf(z)
    Phi_z = jax.scipy.stats.norm.cdf(z)

    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / jnp.sqrt(jnp.pi))

    if mask is not None:
        crps = crps * mask
        return crps.sum() / mask.sum()

    return crps.mean()