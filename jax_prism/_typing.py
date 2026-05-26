"""Core type definitions for JAX-Prism.

This module defines the foundational types used throughout the library.
All type aliases and protocols are exported from here.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from flax import struct

# =============================================================================
# JAX Array Types
# =============================================================================

Array = jax.Array
"""JAX array type. Prefer this over jnp.ndarray for type hints."""

PRNGKey = jax.Array
"""PRNG key type. Semantically distinct from Array for clarity."""

# =============================================================================
# Scalar Types
# =============================================================================

Scalar = Union[float, int, Array]
"""A scalar value, either Python numeric or 0-d JAX array."""

# =============================================================================
# Shape Types
# =============================================================================

Shape = Tuple[int, ...]
"""Static shape tuple."""

DType = jnp.dtype
"""JAX dtype."""

# =============================================================================
# PyTree Types
# =============================================================================

PyTree = Any
"""Generic PyTree type. JAX's tree utilities work on any nested structure."""

Params = Mapping[str, Any]
"""Model parameters as a frozen dict-like structure."""

T = TypeVar("T")
PyTreeOf = Any  # PyTree[T] not yet expressible in Python's type system

# =============================================================================
# Function Types
# =============================================================================

LossFn = Callable[[Array, Array], Array]
"""Loss function signature: (predictions, targets) -> scalar loss."""

GradFn = Callable[[Params, Array, Array], PyTree]
"""Gradient function signature: (params, inputs, targets) -> param gradients."""

# =============================================================================
# Privacy Types
# =============================================================================

Epsilon = float
"""Differential privacy epsilon parameter."""

Delta = float
"""Differential privacy delta parameter."""

AlphaOrder = float
"""RDP alpha (Rényi order) parameter."""

ClipNorm = float
"""L2 norm bound for gradient clipping."""

NoiseMultiplier = float
"""Ratio of noise standard deviation to clipping norm."""


@struct.dataclass
class PrivacyBudget:
    """Represents a (ε, δ)-differential privacy guarantee."""

    epsilon: float
    delta: float

    def __repr__(self) -> str:
        return f"PrivacyBudget(ε={self.epsilon:.4f}, δ={self.delta:.2e})"


# =============================================================================
# Time Series Types
# =============================================================================

TimeSteps = int
"""Number of time steps in a sequence."""

HorizonLength = int
"""Forecast horizon length."""

NumFeatures = int
"""Number of input features."""

BatchSize = int
"""Batch size."""


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DistributionHead(Protocol):
    """Protocol for distribution output heads.

    A DistributionHead transforms raw model outputs into distribution
    parameters and provides methods for computing log probabilities
    and sampling.

    Shape Convention:
        All methods preserve a trailing feature dimension of 1 for univariate
        forecasting. This ensures consistent shapes throughout the pipeline:

        - Raw model output: (B, T, num_params)
        - Distribution params (e.g., loc, scale): (B, T, 1)
        - Targets: (B, T, 1)
        - Log probabilities: (B, T, 1)
        - Point predictions (mean/median): (B, T, 1)

        For multivariate forecasting with F features, shapes become (B, T, F).
    """

    @property
    def num_params(self) -> int:
        """Number of parameters this distribution requires per output."""
        ...

    def params_from_raw(self, raw: Array) -> Dict[str, Array]:
        """Transform raw network output to distribution parameters.

        Args:
            raw: Raw output from model, shape (..., num_params).

        Returns:
            Dictionary mapping parameter names to arrays, each shape (..., 1).
        """
        ...

    def log_prob(self, params: Dict[str, Array], targets: Array) -> Array:
        """Compute log probability of targets under the distribution.

        Args:
            params: Distribution parameters from params_from_raw, each (..., 1).
            targets: Target values, shape (..., 1).

        Returns:
            Log probabilities, shape (..., 1).
        """
        ...

    def sample(
        self, params: Dict[str, Array], key: PRNGKey, sample_shape: Shape = ()
    ) -> Array:
        """Sample from the distribution.

        Args:
            params: Distribution parameters from params_from_raw.
            key: PRNG key for sampling.
            sample_shape: Shape of samples to draw.

        Returns:
            Samples with shape (*sample_shape, ..., 1).
        """
        ...


@runtime_checkable
class Loss(Protocol):
    """Protocol for loss functions.

    Loss functions compute scalar losses from predictions and targets,
    optionally using a distribution head for probabilistic losses.

    Shape Convention:
        - predictions: (B, T, num_params) — raw model output
        - targets: (B, T, 1) — ground truth with trailing feature dim
        - mask: (B, T, 1) — optional mask with same shape as targets
    """

    def __call__(
        self,
        predictions: Array,
        targets: Array,
        mask: Array | None = None,
    ) -> Array:
        """Compute the loss.

        Args:
            predictions: Model predictions, shape (B, T, num_params).
            targets: Ground truth targets, shape (B, T, 1).
            mask: Optional mask, shape (B, T, 1). 1 = valid, 0 = ignore.

        Returns:
            Scalar loss value.
        """
        ...


@runtime_checkable
class PrivacyAccountant(Protocol):
    """Protocol for privacy accountants.

    Privacy accountants track cumulative privacy expenditure across
    multiple training steps using composition theorems.
    """

    def step(
        self,
        noise_multiplier: NoiseMultiplier,
        sample_rate: float,
        num_steps: int = 1,
    ) -> "PrivacyAccountant":
        """Record privacy expenditure for training step(s).

        Args:
            noise_multiplier: σ/C ratio for the step.
            sample_rate: Probability of including each example (Poisson sampling).
            num_steps: Number of identical steps to account for.

        Returns:
            Updated accountant (immutable—returns new instance).
        """
        ...

    def get_privacy_spent(self, delta: Delta) -> PrivacyBudget:
        """Compute total privacy budget spent.

        Args:
            delta: Target delta for (ε, δ)-DP conversion.

        Returns:
            PrivacyBudget with computed epsilon and the given delta.
        """
        ...


@runtime_checkable
class ForecastModel(Protocol):
    """Protocol for forecast models.

    Defines the interface that all forecasting models must implement.
    """

    def __call__(
        self,
        inputs: Array,
        training: bool = False,
    ) -> Array:
        """Forward pass.

        Args:
            inputs: Input tensor, typically shape (B, T, F).
            training: Whether in training mode (affects dropout, etc.).

        Returns:
            Raw output parameters for the distribution head.
        """
        ...
