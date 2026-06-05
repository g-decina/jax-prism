"""Gaussian (Normal) distribution head."""

from typing import Dict

import jax
import jax.numpy as jnp

from jax_prism._typing import Array, PRNGKey, Shape


class GaussianHead:
    """Gaussian distribution head.

    Transforms raw model outputs into (μ, σ) parameters and provides
    log probability and sampling methods.

    The model outputs 2 values per prediction:
    - First value → μ (mean), passed through unchanged
    - Second value → log(σ), exponentiated to ensure σ > 0

    Attributes:
        min_scale: Minimum value for σ (numerical stability).
    """

    min_scale: float = 1e-6

    def __init__(self, min_scale: float = 1e-6):
        self.min_scale = min_scale

    @property
    def num_params(self) -> int:
        """Number of parameters per output (μ, σ)."""
        return 2

    def params_from_raw(self, raw: Array) -> Dict[str, Array]:
        """Transform raw network output to distribution parameters.

        Args:
            raw: Raw output, shape (..., 2).

        Returns:
            Dictionary with 'loc' (μ) and 'scale' (σ), each shape (..., 1).
        """
        loc = raw[..., 0:1]
        scale_raw = raw[..., 1:2]
        
        scale = jax.nn.softplus(scale_raw) + self.min_scale
        
        return {"loc": loc, "scale": scale}

    def log_prob(self, params: Dict[str, Array], targets: Array) -> Array:
        """Compute log probability of targets under Gaussian.

        Args:
            params: Dict with 'loc' and 'scale', each shape (..., 1).
            targets: Target values, shape (..., 1).

        Returns:
            Log probabilities, shape (..., 1).
        """
        # Targets: (B, T, 1), loc: (B, T, 1), scale: (B, T, 1)
        return jax.scipy.stats.norm.logpdf(
            targets, loc=params["loc"], scale=params["scale"]
        )

    def sample(
        self, params: Dict[str, Array], key: PRNGKey, sample_shape: Shape = ()
    ) -> Array:
        """Sample from the Gaussian distribution.

        Args:
            params: Dict with 'loc' and 'scale'.
            key: PRNG key.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape (*sample_shape, ...).
        """
        shape = sample_shape + params["loc"].shape
        raw_sample = jax.random.normal(key=key, shape=shape)
        
        return params["loc"] + params["scale"] * raw_sample

    def mean(self, params: Dict[str, Array]) -> Array:
        """Return the distribution mean (point prediction)."""
        return params["loc"]

    def cdf(self, params: Dict[str, Array], x: Array) -> Array:
        """Compute cumulative distribution function P(X <= x).

        Args:
            params: Dict with 'loc' and 'scale'.
            x: Values at which to evaluate CDF, shape (..., 1).

        Returns:
            CDF values in [0, 1], shape (..., 1).
        """
        return jax.scipy.stats.norm.cdf(x, loc=params["loc"], scale=params["scale"])

    def quantile(self, params: Dict[str, Array], q: Array) -> Array:
        """Compute quantile function (inverse CDF).

        Returns x such that P(X <= x) = q.

        Args:
            params: Dict with 'loc' and 'scale'.
            q: Quantile levels in (0, 1), shape (num_quantiles,).

        Returns:
            Quantile values, shape (..., num_quantiles).
        """
        standard_quantiles = jax.scipy.stats.norm.ppf(q)

        # loc/scale: (..., 1), standard_quantiles: (Q,) → broadcast to (..., Q)
        return params["loc"] + params["scale"] * standard_quantiles
