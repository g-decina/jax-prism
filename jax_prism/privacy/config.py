"""Differential privacy configuration."""

from dataclasses import dataclass


@dataclass
class DPConfig:
    """Configuration for differential privacy training.

    Args:
        clip_norm: Maximum L2 norm for per-sample gradients.
        noise_multiplier: σ, ratio of noise std to clip_norm.
        target_epsilon: Target privacy budget (ε). Optional, for reference.
        target_delta: Target privacy parameter (δ). Optional, for reference.

    Example:
        >>> dp_config = DPConfig(
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ...     target_epsilon=3.0,
        ...     target_delta=1e-5,
        ... )
    """

    clip_norm: float
    noise_multiplier: float
    target_epsilon: float | None = None
    target_delta: float | None = None