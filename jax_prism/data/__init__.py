"""Data handling for JAX-Prism."""

from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.data.dataset import TimeSeriesDataset
from jax_prism.data.scaling import (
    fixed_scale,
    inverse_scale,
    last_value_scale,
    median_scale,
    window_mean_scale,
    window_median_scale,
)
from jax_prism.data.windowing import create_sliding_windows

__all__ = [
    "TimeSeriesBatch",
    "TimeSeriesDataset",
    # Windowing
    "create_sliding_windows",
    # Scaling functions
    "last_value_scale",
    "median_scale",
    "window_median_scale",
    "window_mean_scale",
    "fixed_scale",
    "inverse_scale",
]
