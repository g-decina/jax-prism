"""Distribution heads for probabilistic forecasting."""

from jax_prism.distributions.gaussian import GaussianHead
from jax_prism.distributions.quantile import QuantileHead

__all__ = [
    "GaussianHead",
    "QuantileHead",
]
