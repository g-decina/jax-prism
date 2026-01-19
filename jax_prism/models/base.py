"""Base protocol for forecasting models."""

from typing import Protocol, runtime_checkable

from jax_prism._typing import Array
from jax_prism.data.batch import TimeSeriesBatch


@runtime_checkable
class ForecastModel(Protocol):
    """Protocol for probabilistic forecasting models.

    All forecasting models (TFT, DeepAR, N-BEATS, etc.) should conform
    to this interface. This enables type-safe composition with trainers,
    loss functions, and evaluation utilities.

    The model takes a TimeSeriesBatch and returns raw output parameters
    that are interpreted by a DistributionHead.

    Example:
        >>> model: ForecastModel = TemporalFusionTransformer(config)
        >>> params = model.init(key, batch)
        >>> raw_output = model.apply(params, batch)  # (B, T_dec, num_params)
        >>> dist_params = head.params_from_raw(raw_output)
        >>> predictions = head.sample(dist_params, key)
    """

    def __call__(
        self,
        batch: TimeSeriesBatch,
        training: bool = False,
    ) -> Array:
        """Forward pass.

        Args:
            batch: TimeSeriesBatch containing all inputs.
            training: Whether in training mode (affects dropout, etc.).

        Returns:
            Raw output parameters, shape (B, decoder_length, num_output_params).
            These are passed to a DistributionHead for interpretation.
        """
        ...

    @property
    def num_output_params(self) -> int:
        """Number of output parameters per timestep.

        This should match the DistributionHead used for training:
        - Gaussian: 2 (loc, scale)
        - Quantile: num_quantiles
        - StudentT: 3 (loc, scale, df)
        """
        ...

    @property
    def decoder_length(self) -> int:
        """Forecast horizon length."""
        ...
