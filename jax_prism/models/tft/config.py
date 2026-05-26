"""TFT configuration."""

from dataclasses import dataclass
from typing import Any, Callable

# Type alias for Flax initializers
Initializer = Callable[[Any, tuple[int, ...], Any], Any]


@dataclass(frozen=True)
class ParamHeadConfig:
    """Configuration for a single parameter output head.

    Used with TFTConfig.param_head_configs to create separate output heads
    for each distribution parameter (e.g., μ and σ for Gaussian).

    Attributes:
        output_dim: Number of outputs for this head.
        dropout_rate: Dropout probability for this head's GRN.
            None inherits from TFTConfig.dropout_rate.
        hidden_size: Hidden dimension for this head's GRN.
            None inherits from TFTConfig.hidden_size.
        use_output_bias: Whether to add an explicit OutputBias module
            after the projection. Helps with offset correction during
            phased training. Default True.
    """
    output_dim: int = 1
    dropout_rate: float | None = None
    hidden_size: int | None = None
    use_output_bias: bool = True


@dataclass(frozen=True)
class TFTConfig:
    """Configuration for Temporal Fusion Transformer.

    All dimensions and hyperparameters for the TFT model.
    Frozen to prevent accidental mutation.

    Attributes:
        hidden_size: Hidden dimension throughout the model.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads for GQA. Defaults to num_heads (MHA).
            Set to 1 for MQA, or divisor of num_heads for GQA.
        num_lstm_layers: Number of stacked LSTM layers in encoder/decoder.
        dropout_rate: Dropout probability.
        attention_dropout_rate: Dropout probability for attention weights.
        num_static_features: Number of static input features (0 if none).
        num_known_features: Number of known future covariates.
        num_observed_features: Number of observed-only covariates.
        num_output_params: Number of output parameters (e.g., 2 for Gaussian, Q for quantiles).
        encoder_length: Number of past time steps.
        decoder_length: Number of future time steps to predict.
        rope_base: Base for RoPE frequencies.
    """

    # Model dimensions
    hidden_size: int = 64
    num_heads: int = 4
    num_kv_heads: int | None = None  # None = MHA (num_kv_heads = num_heads)
    num_lstm_layers: int = 1

    # Regularization
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0

    # Input dimensions (must be set based on data)
    num_static_features: int = 0
    num_known_features: int = 0
    num_observed_features: int = 0

    # Output
    num_output_params: int = 2  # e.g., 2 for Gaussian (loc, scale)

    # Sequence lengths
    encoder_length: int = 168  # e.g., 7 days at hourly
    decoder_length: int = 24  # e.g., 1 day ahead

    # Positional encoding
    rope_base: float = 10000.0

    # Per-parameter output heads (optional)
    param_head_configs: tuple[ParamHeadConfig, ...] | None = None

    # Output layer initialization (for single-head path only)
    # Use jax_prism.nn.quantile_bias_init for quantile regression
    output_bias_init: Initializer | None = None

    # Whether to add a separate OutputBias module after output projection.
    # This is crucial for quantile regression where initial delta outputs
    # need to be positive to avoid softplus saturation.
    # When True, use output_bias_init to set the initial bias values.
    use_output_bias: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.num_kv_heads is None:
            # Default to MHA
            object.__setattr__(self, "num_kv_heads", self.num_heads)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        if self.param_head_configs is not None:
            total_outputs = sum(h.output_dim for h in self.param_head_configs)
            if total_outputs != self.num_output_params:
                raise ValueError(
                    f"Sum of param_head_configs output_dims ({total_outputs}) "
                    f"must match num_output_params ({self.num_output_params})."
                )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads

    @property
    def total_sequence_length(self) -> int:
        """Total sequence length (encoder + decoder)."""
        return self.encoder_length + self.decoder_length

    @property
    def num_encoder_features(self) -> int:
        """Number of features in encoder input (target + known + observed)."""
        # 1 for target
        return 1 + self.num_known_features + self.num_observed_features

    @property
    def num_decoder_features(self) -> int:
        """Number of features in decoder input (known only)."""
        return self.num_known_features
