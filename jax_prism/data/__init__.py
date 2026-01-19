"""Data handling for JAX-Prism."""

from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.data.scaling import (
    fixed_scale,
    inverse_scale,
    last_value_scale,
    median_scale,
)

__all__ = [
    "TimeSeriesBatch",
    # Scaling functions
    "last_value_scale",
    "median_scale",
    "fixed_scale",
    "inverse_scale",
]
