"""Loss functions for training."""

from jax_prism.losses.mse import MSELoss
from jax_prism.losses.nll import NLLLoss
from jax_prism.losses.quantile import QuantileLoss

__all__ = [
    "NLLLoss",
    "QuantileLoss",
    "MSELoss"
]
