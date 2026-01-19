"""TimeSeriesBatch: The core data structure for JAX-Prism.

This module defines the pytree-compatible batch structure that flows
through the entire training and inference pipeline.
"""

from typing import Dict

from flax import struct

from jax_prism._typing import Array


@struct.dataclass
class TimeSeriesBatch:
    """Immutable, pytree-compatible batch of time series data.

    This is the canonical data structure passed to models during training
    and inference. All arrays have a leading batch dimension.

    The design separates inputs by type (static vs temporal, known vs observed)
    to support TFT's variable selection networks and enable flexible
    feature handling across different model architectures.

    Attributes:
        past_targets: Historical target values, shape (B, T_past, num_targets).
            Use num_targets=1 for univariate forecasting.
        future_targets: Future target values for training, shape (B, T_future, num_targets).
            None during inference.
        past_observed_covariates: Time-varying features known only in the past,
            shape (B, T_past, F_obs). Examples: sensor readings, prices.
        past_known_covariates: Time-varying features known in past and future,
            shape (B, T_past, F_known). Examples: day-of-week, holidays.
        future_known_covariates: Known future covariates for the forecast horizon,
            shape (B, T_future, F_known).
        static_covariates: Time-invariant features, shape (B, F_static).
            Examples: store ID embedding, product category.
        mask: Valid data indicator, shape (B,). 1.0 = valid, 0.0 = padding.

    Example:
        >>> batch = TimeSeriesBatch(
        ...     past_targets=jnp.ones((32, 168, 1)),   # 32 series, 168 past steps, univariate
        ...     future_targets=jnp.ones((32, 24, 1)),  # 24-step horizon
        ...     past_known_covariates=jnp.ones((32, 168, 3)),  # 3 known features
        ...     future_known_covariates=jnp.ones((32, 24, 3)),
        ...     mask=jnp.ones((32,)),
        ... )
    """

    # Targets
    past_targets: Array                             # (B, T_past, num_targets)
    future_targets: Array | None = None             # (B, T_future, num_targets) or None

    # Covariates
    past_observed_covariates: Array | None = None   # (B, T_past, F_obs)
    past_known_covariates: Array | None = None      # (B, T_past, F_known)
    future_known_covariates: Array | None = None    # (B, T_future, F_known)
    static_covariates: Array | None = None          # (B, F_static)

    # Masking
    mask: Array | None = None                       # (B,)

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self.past_targets.shape[0]

    @property
    def past_length(self) -> int:
        """Return the number of past time steps."""
        return self.past_targets.shape[1]

    @property
    def future_length(self) -> int | None:
        """Return the forecast horizon length, or None if no future targets."""
        if self.future_targets is None:
            return None
        return self.future_targets.shape[1]

    def get_encoder_inputs(self) -> Dict[str, Array]:
        """Assemble inputs for the encoder (past sequence).

        Returns a dictionary suitable for variable selection networks,
        containing all available past information.

        Returns:
            Dictionary with keys for each input type present.
        """
        inputs = {"targets": self.past_targets}

        if self.past_observed_covariates is not None:
            inputs["observed"] = self.past_observed_covariates

        if self.past_known_covariates is not None:
            inputs["known"] = self.past_known_covariates

        return inputs

    def get_decoder_inputs(self) -> Dict[str, Array]:
        """Assemble inputs for the decoder (future sequence).

        Returns a dictionary with known future information.
        Does not include future_targets (those are labels, not inputs).

        Returns:
            Dictionary with keys for each input type present.
        """
        inputs = {}

        if self.future_known_covariates is not None:
            inputs["known"] = self.future_known_covariates

        return inputs

    def replace(self, **kwargs) -> "TimeSeriesBatch":
        """Create a new batch with some fields replaced.

        This is a convenience wrapper around flax.struct.dataclass's
        built-in replace functionality.

        Args:
            **kwargs: Fields to replace.

        Returns:
            New TimeSeriesBatch with updated fields.
        """
        return self.replace(**kwargs)
