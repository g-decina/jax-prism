"""Custom initializers for neural network parameters.

JAX/Flax initializers are callables with signature:
    (key: PRNGKey, shape: Shape, dtype: DType) -> Array

These initializers handle special cases like quantile regression where
certain outputs need specific initialization to avoid gradient issues.
"""

import jax.numpy as jnp
from flax.linen import initializers

from jax_prism._typing import Array, DType, PRNGKey, Shape


def quantile_bias_init(
    num_quantiles: int,
    delta_init: float = 2.0,
    median_init: float = 0.0,
) -> initializers.Initializer:
    """Create bias initializer for quantile regression output layer.

    For monotonic quantile parameterization with centered softplus deltas:
    - Output 0: median (unconstrained)
    - Outputs 1 to n_pairs: lower deltas (inner to outer from median)
    - Outputs n_pairs+1 to end: upper deltas (inner to outer from median)

    The delta outputs pass through softplus, which saturates for large
    negative inputs (gradient ≈ 0). Initializing deltas to positive values
    ensures softplus operates in a region with healthy gradients.

    Args:
        num_quantiles: Total number of quantile outputs (must be odd).
        delta_init: Initial value for delta outputs. Default 2.0 gives
            softplus(2.0) ≈ 2.1 with gradient sigmoid(2.0) ≈ 0.88.
        median_init: Initial value for median output. Default 0.0.

    Returns:
        Flax initializer function.

    Example:
        >>> from flax import linen as nn
        >>> from jax_prism.nn.initializers import quantile_bias_init
        >>>
        >>> # 5 quantiles: q10, q25, q50, q75, q90
        >>> output_layer = nn.Dense(
        ...     features=5,
        ...     bias_init=quantile_bias_init(num_quantiles=5, delta_init=2.0)
        ... )
    """
    if num_quantiles % 2 == 0:
        raise ValueError(
            f"num_quantiles must be odd (need a median), got {num_quantiles}"
        )

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Initialize bias with median=0 and deltas=delta_init."""
        del key  # Deterministic initialization

        if shape[-1] != num_quantiles:
            raise ValueError(
                f"Expected bias shape ending in {num_quantiles}, got {shape}"
            )

        # Build bias vector: [median, lower_deltas..., upper_deltas...]
        bias = jnp.full(shape, delta_init, dtype=dtype)
        # Set median (first output) to median_init
        bias = bias.at[..., 0].set(median_init)

        return bias

    return init


def softplus_bias_init(
    init_value: float = 2.0,
) -> initializers.Initializer:
    """Create bias initializer for outputs that pass through softplus.

    Softplus(x) = log(1 + exp(x)) saturates for x << 0, causing near-zero
    gradients. This initializer sets biases to positive values to ensure
    the operating point has healthy gradients.

    Args:
        init_value: Initial bias value. Default 2.0 gives:
            - softplus(2.0) ≈ 2.1 (output value)
            - sigmoid(2.0) ≈ 0.88 (gradient magnitude)

    Returns:
        Flax initializer function.

    Example:
        >>> # For a scale parameter that goes through softplus
        >>> scale_layer = nn.Dense(
        ...     features=1,
        ...     bias_init=softplus_bias_init(init_value=2.0)
        ... )
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Initialize all biases to init_value."""
        del key  # Deterministic initialization
        return jnp.full(shape, init_value, dtype=dtype)

    return init
