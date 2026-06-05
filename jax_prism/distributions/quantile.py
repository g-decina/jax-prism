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

    When enforce_monotonicity=True (default), raw outputs are interpreted as:
        [median, delta_lower_1, ..., delta_lower_k, delta_upper_1, ..., delta_upper_k]
    and transformed via cumulative softplus to guarantee q_i <= q_{i+1}.

    Tradeoff vs Gaussian:
    - More flexible (can capture asymmetry, heavy tails)
    - No closed-form density (can't compute exact log_prob)

    Attributes:
        quantiles: Array of quantile levels, e.g., [0.1, 0.5, 0.9].
        enforce_monotonicity: Whether to enforce quantile ordering.
    """

    def __init__(
        self, 
        quantiles: Array,
        enforce_monotonicity: bool = True,
    ):
        """Initialize with target quantile levels.

        Args:
            quantiles: 1D array of quantile levels in (0, 1).
                    Should be sorted ascending.
            enforce_monotonicity: Whether to enforce q_i <= q_{i+1}.
                    Defaults to True.
        """
        self.quantiles = jnp.asarray(quantiles)
        
        if enforce_monotonicity:
            if len(self.quantiles) % 2 != 1:
                raise ValueError(
                    f"An odd number of quantiles must be provided for monotonicity to be enforced."
                )
        
        self.enforce_monotonicity = enforce_monotonicity
        

    @property
    def num_params(self) -> int:
        """Number of parameters = number of quantiles."""
        return len(self.quantiles)

    def params_from_raw(self, raw: Array) -> dict[str, Array]:
        """Transform raw output to quantile values.

        When enforce_monotonicity=True, raw is interpreted as:
            [median, delta_lower_1, ..., delta_lower_k, delta_upper_1, ..., delta_upper_k]
        where deltas are passed through softplus and cumsum to enforce ordering.

        Args:
            raw: Raw output, shape (..., num_quantiles).

        Returns:
            Dictionary with 'quantile_values' and 'quantile_levels'.
        """
        if self.enforce_monotonicity:
            n_pairs = (len(self.quantiles) - 1) // 2 
            # First output is the median (unconstrained)
            median = raw[..., 0:1]
            # Lower deltas: indices 1 to n_pairs (inner to outer from median)
            lower_deltas = jax.nn.softplus(raw[..., 1 : n_pairs + 1])
            # Upper deltas: indices n_pairs+1 to end (inner to outer from median)
            upper_deltas = jax.nn.softplus(raw[..., n_pairs + 1 :])
            
            # Cumulative sums to enforce monotonicity
            lower_cumsum = jnp.cumsum(lower_deltas, axis=-1)
            upper_cumsum = jnp.cumsum(upper_deltas, axis=-1)

            # Lower quantiles: median - cumsum, flip so outermost (q10) comes first
            lower_quantiles = median - jnp.flip(lower_cumsum, axis=-1)

            # Upper quantiles: median + cumsum
            upper_quantiles = median + upper_cumsum

            vals = jnp.concatenate([lower_quantiles, median, upper_quantiles], axis=-1)
            return {
                "quantile_values": vals,
                "quantile_levels": self.quantiles,
            }
        
        else:
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

    def cdf(self, params: dict[str, Array], x: Array) -> Array:
        """Compute cumulative distribution function P(X <= x).

        Approximates CDF by linear interpolation between quantile values.

        Args:
            params: Dict with 'quantile_values' and 'quantile_levels'.
            x: Values at which to evaluate CDF, shape (..., 1).

        Returns:
            CDF values in [0, 1], shape (..., 1).
        """
        quantile_values = params["quantile_values"]  # (..., Q)
        quantile_levels = params["quantile_levels"]  # (Q,)

        # Interpolate: given x, find corresponding quantile level
        # jnp.interp(x, xp, fp) where xp=quantile_values, fp=quantile_levels
        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1)
        flat_qv = quantile_values.reshape(-1, len(quantile_levels))

        def interp_cdf(x_val, qv_row):
            return jnp.interp(x_val, qv_row, quantile_levels)

        flat_cdf = jax.vmap(interp_cdf)(flat_x, flat_qv)
        return flat_cdf.reshape(original_shape + (1,))

    def quantile(self, params: dict[str, Array], q: Array) -> Array:
        """Compute quantile function (inverse CDF).

        Returns x such that P(X <= x) = q, via linear interpolation.

        Args:
            params: Dict with 'quantile_values' and 'quantile_levels'.
            q: Quantile levels in (0, 1), shape (num_quantiles,).

        Returns:
            Quantile values, shape (..., num_quantiles).
        """
        quantile_values = params["quantile_values"]  # (..., Q)
        quantile_levels = params["quantile_levels"]  # (Q,)

        original_shape = quantile_values.shape[:-1]
        flat_qv = quantile_values.reshape(-1, len(quantile_levels))

        def interp_quantile(qv_row):
            return jnp.interp(q, quantile_levels, qv_row)

        # Result: (num_batch, num_q)
        flat_result = jax.vmap(interp_quantile)(flat_qv)
        return flat_result.reshape(original_shape + (len(q),))