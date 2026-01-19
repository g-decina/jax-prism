"""JAX-Prism: Privacy-preserving probabilistic forecasting.

A library combining differentially private training (DP-SGD) with
Bayesian uncertainty quantification for time series forecasting.
"""

__version__ = "0.1.0-dev"

# Data structures
from jax_prism.data import TimeSeriesBatch
from jax_prism.data import last_value_scale, median_scale, fixed_scale, inverse_scale

# Model
from jax_prism.models.tft import TFTConfig, TemporalFusionTransformer

# Distributions
from jax_prism.distributions import GaussianHead, QuantileHead

# Losses
from jax_prism.losses import NLLLoss, QuantileLoss

# Privacy
from jax_prism.privacy import (
    RDPAccountant,
    dp_gradients,
    compute_per_sample_gradients,
    clip_gradients,
    add_noise,
)

# Metrics
from jax_prism.metrics import mae, smape, mase, quantile_loss, coverage

__all__ = [
    # Version
    "__version__",
    # Data
    "TimeSeriesBatch",
    "last_value_scale",
    "median_scale",
    "fixed_scale",
    "inverse_scale",
    # Model
    "TFTConfig",
    "TemporalFusionTransformer",
    # Distributions
    "GaussianHead",
    "QuantileHead",
    # Losses
    "NLLLoss",
    "QuantileLoss",
    # Privacy
    "RDPAccountant",
    "dp_gradients",
    "compute_per_sample_gradients",
    "clip_gradients",
    "add_noise",
    # Metrics
    "mae",
    "smape",
    "mase",
    "quantile_loss",
    "coverage",
]
