"""Differential privacy machinery for DP-SGD training."""

from jax_prism.privacy.accountants import RDPAccountant
from jax_prism.privacy.clipping import (
    clip_gradients,
    clip_single_gradient,
    compute_global_norm,
)
from jax_prism.privacy.gradients import (
    compute_per_sample_gradients,
    dp_gradients,
)
from jax_prism.privacy.noise import add_noise, generate_noise_tree
from jax_prism.privacy.training import (
    TrainStepOutput,
    make_dp_train_step,
    make_train_step,
)

__all__ = [
    # Accountants
    "RDPAccountant",
    # High-level API
    "make_dp_train_step",
    "make_train_step",
    "TrainStepOutput",
    # Functional API
    "dp_gradients",
    "compute_per_sample_gradients",
    # Clipping
    "clip_gradients",
    "clip_single_gradient",
    "compute_global_norm",
    # Noise
    "add_noise",
    "generate_noise_tree",
]
