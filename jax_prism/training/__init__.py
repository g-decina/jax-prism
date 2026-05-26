"""Training module."""

from jax_prism.training.calibration import calibrate_output_bias
from jax_prism.training.freezing import freeze_params_by_pattern
from jax_prism.training.masking import (
    freeze_output_indices,
    no_weight_decay_on_bias,
    no_weight_decay_on_pattern,
)
from jax_prism.training.reinit import reinitialize_param_head
from jax_prism.training.schedules import warmup_cosine_schedule, warmup_plateau_cosine_schedule

__all__ = [
    # Freezing utilities
    "freeze_params_by_pattern",
    # Reinitialization utilities
    "reinitialize_param_head",
    # Scheduling creation functions
    "warmup_cosine_schedule",
    "warmup_plateau_cosine_schedule",
    # Calibration utilities
    "calibrate_output_bias",
    # Masking utilities
    "freeze_output_indices",
    "no_weight_decay_on_bias",
    "no_weight_decay_on_pattern",
]