"""Rényi Differential Privacy (RDP) accountant.

Tracks privacy expenditure using RDP composition, which gives tighter
bounds than naive (ε,δ)-DP composition for iterative mechanisms like DP-SGD.

References:
    Mironov, "Rényi Differential Privacy", CSF 2017.
    Wang, Balle, Kasiviswanathan, "Subsampled Rényi Differential Privacy
        and Analytical Moments Accountant", AISTATS 2019.
"""

import jax
import jax.numpy as jnp
from flax import struct

from jax_prism._typing import Array, PrivacyBudget
from jax_prism.privacy.accountants.base import (
    DEFAULT_RDP_ORDERS,
    MAX_EPSILON,
    MIN_SAMPLE_RATE,
)
from jax_prism.utils.math import log_binom


@struct.dataclass
class RDPAccountant:
    """Rényi Differential Privacy accountant.

    Tracks cumulative RDP budget across multiple α orders and converts
    to (ε, δ)-DP on demand using the optimal order.

    Immutable: step() returns a new accountant with updated state.

    Attributes:
        alphas: Rényi orders to track, shape (num_orders,).
        rdp_epsilons: Accumulated RDP ε for each α, shape (num_orders,).

    Example:
        >>> accountant = RDPAccountant.create()
        >>> # After each training step:
        >>> accountant = accountant.step(noise_multiplier=1.0, sample_rate=0.01)
        >>> # Query current privacy:
        >>> budget = accountant.get_privacy_spent(delta=1e-5)
        >>> print(f"ε = {budget.epsilon:.2f}")
    """

    alphas: Array  # (num_orders,)
    rdp_epsilons: Array  # (num_orders,)

    @classmethod
    def create(cls, alphas: Array | None = None) -> "RDPAccountant":
        """Create a fresh accountant with zero budget spent.

        Args:
            alphas: Optional custom Rényi orders. Defaults to DEFAULT_RDP_ORDERS.

        Returns:
            New RDPAccountant with zero accumulated RDP.
        """
        if alphas is None:
            alphas = DEFAULT_RDP_ORDERS
        alphas = jnp.asarray(alphas)
        return cls(alphas=alphas, rdp_epsilons=jnp.zeros_like(alphas))

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> "RDPAccountant":
        """Record privacy expenditure for training step(s).

        Args:
            noise_multiplier: σ = noise_std / clip_norm.
            sample_rate: q = batch_size / dataset_size, in (0, 1].
            num_steps: Number of identical steps to account for.

        Returns:
            New accountant with updated RDP budget.

        Raises:
            ValueError: If noise_multiplier <= 0 or sample_rate not in (0, 1].
        """
        if noise_multiplier <= 0:
            raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
        if not (0 < sample_rate <= 1):
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

        # Compute RDP for one step at each α
        step_rdp = compute_rdp(self.alphas, noise_multiplier, sample_rate)

        # Accumulate (RDP composes additively)
        new_rdp = self.rdp_epsilons + num_steps * step_rdp

        return self.replace(rdp_epsilons=new_rdp)

    def get_privacy_spent(self, delta: float) -> PrivacyBudget:
        """Convert accumulated RDP to (ε, δ)-DP.

        Uses optimal α selection: ε = min_α [rdp(α) + log(1/δ)/(α-1)]

        Args:
            delta: Target δ for the (ε, δ)-DP guarantee.

        Returns:
            PrivacyBudget with computed ε and the given δ.

        Raises:
            ValueError: If delta <= 0 or delta >= 1.
        """
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        epsilon = rdp_to_epsilon(self.rdp_epsilons, self.alphas, delta)
        return PrivacyBudget(epsilon=float(epsilon), delta=delta)


# =============================================================================
# Core RDP Computations — YOU IMPLEMENT THESE
# =============================================================================


def compute_rdp(
    alphas: Array,
    noise_multiplier: float,
    sample_rate: float,
) -> Array:
    """Compute RDP of the subsampled Gaussian mechanism.

    For each Rényi order α, compute the RDP guarantee ε(α) for one step
    of the Gaussian mechanism with Poisson subsampling.

    Args:
        alphas: Rényi orders, shape (num_orders,). Must be > 1.
        noise_multiplier: σ = noise_std / sensitivity.
        sample_rate: q = probability each example is included (Poisson sampling).

    Returns:
        RDP ε values for each α, shape (num_orders,).

    Mathematical Background:
        For the Gaussian mechanism without subsampling (q=1):
            ε(α) = α / (2σ²)

        For subsampled Gaussian (q < 1), the privacy amplifies.
        The exact formula involves a log-sum-exp over binomial terms.

        For v0.1.0, implement:
        1. If sample_rate >= 1 - 1e-6: use the no-subsampling formula
        2. Otherwise: use the analytical moments accountant formula

    Reference:
        Equation (5) in Mironov 2017 for Gaussian RDP.
        Theorem 9 in Wang et al. 2019 for subsampled RDP.
    """
    if sample_rate >= 1.0 - 1e-6:
        return alphas / (2 * noise_multiplier ** 2)

    # Subsampling case: use analytical moments accountant
    # We need static array sizes for JAX tracing, so we use the max alpha
    # and mask out invalid entries with -inf (contributes 0 to logsumexp)
    q = sample_rate
    sigma_sq = noise_multiplier ** 2

    # Max alpha determines array size (ceiling + 1 for k=0..alpha)
    max_alpha_int = int(jnp.ceil(jnp.max(alphas))) + 1
    k = jnp.arange(max_alpha_int)  # Fixed size: [0, 1, 2, ..., max_alpha]

    def rdp_single_alpha(alpha):
        """Compute RDP for a single alpha using masked logsumexp."""
        alpha_int = jnp.ceil(alpha).astype(jnp.int32)

        # Compute log terms for all k (some will be masked out)
        log_terms = (
            log_binom(alpha_int, k)
            + (alpha_int - k) * jnp.log(jnp.maximum(1 - q, 1e-10))
            + k * jnp.log(jnp.maximum(q, 1e-10))
            + (k ** 2 - k) / (2 * sigma_sq)
        )

        # Mask: only k <= alpha_int are valid
        # Invalid entries get -inf so they contribute 0 to logsumexp
        mask = k <= alpha_int
        log_terms = jnp.where(mask, log_terms, -jnp.inf)

        log_sum = jax.scipy.special.logsumexp(log_terms)

        return log_sum / (alpha - 1)

    return jax.vmap(rdp_single_alpha)(alphas)


def rdp_to_epsilon(
    rdp_epsilons: Array,
    alphas: Array,
    delta: float,
) -> float:
    """Convert RDP guarantee to (ε, δ)-DP.

    Finds the optimal α that minimizes ε for the given δ.

    Args:
        rdp_epsilons: Accumulated RDP values for each α, shape (num_orders,).
        alphas: Rényi orders, shape (num_orders,).
        delta: Target δ for the (ε, δ)-DP guarantee.

    Returns:
        Optimal ε such that the mechanism is (ε, δ)-DP.

    Mathematical Background:
        The conversion formula is:
            ε = rdp(α) + log(1/δ) / (α - 1)

        We compute this for all α in our grid and return the minimum.
        This is tight when the optimal α is in our grid.

    Reference:
        Proposition 3 in Mironov 2017.
    """
    if delta <= 0:
        raise ValueError("Delta must be positive.")
    
    eps_candidates = rdp_epsilons + jnp.log(1 / delta) / (alphas - 1)
    
    return float(jnp.min(eps_candidates))


# =============================================================================
# Helper Functions (for subsampling amplification)
# =============================================================================


def _compute_rdp_no_subsampling(alpha: float, noise_multiplier: float) -> float:
    """RDP of Gaussian mechanism without subsampling.

    ε(α) = α / (2σ²)

    Args:
        alpha: Rényi order (> 1).
        noise_multiplier: σ = noise_std / sensitivity.

    Returns:
        RDP ε at order α.
    """
    return alpha / (2.0 * noise_multiplier ** 2)


def _compute_rdp_subsampled(
    alpha: float,
    noise_multiplier: float,
    sample_rate: float,
) -> float:
    """RDP of subsampled Gaussian mechanism (single α).

    Uses the analytical moments accountant formula from Wang et al. 2019.

    For integer α, this is:
        ε(α) = (1/(α-1)) * log( Σ_{k=0}^{α} C(α,k) * (1-q)^{α-k} * q^k * exp((k²-k)/(2σ²)) )

    For non-integer α, we use an upper bound or interpolation.

    Args:
        alpha: Rényi order (> 1).
        noise_multiplier: σ = noise_std / sensitivity.
        sample_rate: q = subsampling probability.

    Returns:
        RDP ε at order α.
    """
    # TODO: Implement if needed for exact computation
    # For v0.1.0, you can use a simpler bound or call an external library
    raise NotImplementedError("Subsampled RDP computation")
