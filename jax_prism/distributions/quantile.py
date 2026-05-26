"""Quantile distribution head.

Unlike parametric distributions (Gaussian, Student-t), the quantile head
directly outputs quantile values without assuming a functional form.
This is more flexible but doesn't provide a proper density.
"""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array, PRNGKey, Shape


class QuantileHead:
    """Quantile-based distribution head.

    The model directly outputs quantile values (e.g., 10th, 50th, 90th percentiles).
    This is non-parametric: no assumed distribution family.

    Tradeoff vs Gaussian:
    - More flexible (can capture asymmetry, heavy tails)
    - No closed-form density (can't compute exact log_prob)
    - Quantiles may cross if not constrained

    Attributes:
        quantiles: Array of quantile levels, e.g., [0.1, 0.5, 0.9].
    """

    def __init__(self, quantiles: Array):
        """Initialize with target quantile levels.

        Args:
            quantiles: 1D array of quantile levels in (0, 1).
                    Should be sorted ascending.
        """
        self.quantiles = jnp.asarray(quantiles)

    @property
    def num_params(self) -> int:
        """Number of parameters = number of quantiles."""
        return len(self.quantiles)

    def params_from_raw(self, raw: Array) -> dict[str, Array]:
        """Transform raw output to quantile values.

        Args:
            raw: Raw output, shape (..., num_quantiles).

        Returns:
            Dictionary with 'quantile_values' and 'quantile_levels'.
        """
        return {
            "quantile_values": raw,
            "quantile_levels": self.quantiles,
        }

    def log_prob(self, params: dict[str, Array], targets: Array) -> Array:
        """Not well-defined for quantile regression.

        Quantile regression doesn't define a proper density, so log_prob
        is not directly available. For training, use QuantileLoss instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "QuantileHead doesn't define a density. Use QuantileLoss for training."
        )

    def sample(
        self, params: dict[str, Array], key: PRNGKey, sample_shape: Shape = ()
    ) -> Array:
        """Sample by interpolating between quantiles.

        Approximates the CDF by linear interpolation between quantile values,
        then inverts to sample.

        Args:
            params: Dict with 'quantile_values' and 'quantile_levels'.
            key: PRNG key.
            sample_shape: Shape of samples.

        Returns:
            Samples with shape (*sample_shape, ...).
        """
        quantile_values = params["quantile_values"]  # (..., Q)
        quantile_levels = params["quantile_levels"]  # (Q,)

        # Batch shape is everything except the last (quantile) dimension
        batch_shape = quantile_values.shape[:-1]
        num_quantiles = quantile_values.shape[-1]

        # Use Python math for shape computation (static at trace time)
        import math
        num_batch = math.prod(batch_shape) if batch_shape else 1
        num_samples = math.prod(sample_shape) if sample_shape else 1

        # Sample uniform values: (num_samples, num_batch)
        u = jax.random.uniform(key, (num_samples, num_batch))

        # Flatten quantile_values to (num_batch, Q)
        flat_qv = quantile_values.reshape(num_batch, num_quantiles)

        # Interpolate for each batch element
        def interp_single(u_col, qv_row):
            return jnp.interp(u_col, quantile_levels, qv_row)

        # vmap over batch dimension (axis 1 of u, axis 0 of flat_qv)
        # Result: (num_samples, num_batch)
        samples = jax.vmap(interp_single, in_axes=(1, 0), out_axes=1)(u, flat_qv)

        # Reshape to (*sample_shape, *batch_shape)
        output_shape = sample_shape + batch_shape
        return samples.reshape(output_shape)

    def median(self, params: dict[str, Array]) -> Array:
        """Return the median (0.5 quantile) if available, else interpolate.

        Args:
            params: Dict with 'quantile_values' and 'quantile_levels'.

        Returns:
            Median values, shape (..., 1).
        """
        quantile_values = params["quantile_values"]  # (..., Q)
        quantile_levels = params["quantile_levels"]  # (Q,)

        # Interpolate to find value at q=0.5
        # Need to handle arbitrary batch shapes
        # Use vmap to apply interp over batch dimensions
        def interp_median(qv):
            return jnp.interp(0.5, quantile_levels, qv)

        # Apply over all batch dimensions
        original_shape = quantile_values.shape[:-1]
        flat_qv = quantile_values.reshape(-1, len(quantile_levels))
        flat_median = jax.vmap(interp_median)(flat_qv)
        return flat_median.reshape(original_shape + (1,))  # Keep trailing dim

    def mean(self, params: dict[str, Array]) -> Array:
        """Return median as point prediction (mean not well-defined)."""
        return self.median(params)

    def prediction_interval(
        self, params: dict[str, Array], coverage: float = 0.8
    ) -> tuple[Array, Array]:
        """Return prediction interval for given coverage.

        Args:
            params: Dict with 'quantile_values' and 'quantile_levels'.
            coverage: Desired coverage, e.g., 0.8 for 80% interval.

        Returns:
            (lower, upper) bounds, each shape (..., 1).
        """
        quantile_values = params["quantile_values"]  # (..., Q)
        quantile_levels = params["quantile_levels"]  # (Q,)

        lower_q = (1 - coverage) / 2
        upper_q = (1 + coverage) / 2

        def interp_bounds(qv):
            lower = jnp.interp(lower_q, quantile_levels, qv)
            upper = jnp.interp(upper_q, quantile_levels, qv)
            return lower, upper

        original_shape = quantile_values.shape[:-1]
        flat_qv = quantile_values.reshape(-1, len(quantile_levels))
        flat_lower, flat_upper = jax.vmap(interp_bounds)(flat_qv)

        return (
            flat_lower.reshape(original_shape + (1,)),
            flat_upper.reshape(original_shape + (1,)),
        )