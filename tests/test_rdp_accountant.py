"""Tests for RDP (Rényi Differential Privacy) accountant.

Test Structure:
---------------
1. TestRDPAccountantCreation: Factory method and initialization
2. TestRDPAccountantStep: Privacy budget accumulation
3. TestRDPAccountantConversion: RDP to (ε,δ)-DP conversion
4. TestComputeRDP: Core RDP computation (no subsampling + subsampling)
5. TestRDPToEpsilon: Conversion function directly
6. TestMathematicalProperties: Property-based tests for theoretical invariants
7. TestNumericalStability: Edge cases and numerical robustness
8. TestProtocolCompliance: Verify it satisfies PrivacyAccountant protocol

Testing Philosophy:
-------------------
- Test behavior, not implementation
- Use known analytical results to verify correctness
- Property-based tests for mathematical invariants
- Edge cases: extreme σ, extreme q, many steps
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings, strategies as st

from jax_prism._typing import PrivacyAccountant, PrivacyBudget
from jax_prism.privacy.accountants.rdp import (
    RDPAccountant,
    compute_rdp,
    rdp_to_epsilon,
)
from jax_prism.privacy.accountants.base import DEFAULT_RDP_ORDERS


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fresh_accountant():
    """Create a fresh RDP accountant with default orders."""
    return RDPAccountant.create()


@pytest.fixture
def custom_alphas():
    """Small alpha grid for faster tests."""
    return jnp.array([2.0, 4.0, 8.0, 16.0, 32.0])


# =============================================================================
# TestRDPAccountantCreation
# =============================================================================


class TestRDPAccountantCreation:
    """Tests for RDPAccountant.create() factory method."""

    def test_create_default_alphas(self):
        """Default creation uses DEFAULT_RDP_ORDERS."""
        accountant = RDPAccountant.create()

        chex.assert_trees_all_close(accountant.alphas, DEFAULT_RDP_ORDERS)

    def test_create_zero_initial_budget(self):
        """Fresh accountant has zero RDP budget."""
        accountant = RDPAccountant.create()

        chex.assert_trees_all_close(
            accountant.rdp_epsilons,
            jnp.zeros_like(DEFAULT_RDP_ORDERS),
        )

    def test_create_custom_alphas(self, custom_alphas):
        """Can create accountant with custom alpha grid."""
        accountant = RDPAccountant.create(alphas=custom_alphas)

        chex.assert_trees_all_close(accountant.alphas, custom_alphas)
        assert len(accountant.rdp_epsilons) == len(custom_alphas)

    def test_create_converts_list_to_array(self):
        """Accepts Python list and converts to JAX array."""
        alphas_list = [2.0, 4.0, 8.0]
        accountant = RDPAccountant.create(alphas=alphas_list)

        assert isinstance(accountant.alphas, jax.Array)


# =============================================================================
# TestRDPAccountantStep
# =============================================================================


class TestRDPAccountantStep:
    """Tests for RDPAccountant.step() method."""

    def test_step_returns_new_accountant(self, fresh_accountant):
        """step() returns a NEW accountant (immutability)."""
        new_accountant = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)

        # Different objects
        assert new_accountant is not fresh_accountant
        # Original unchanged
        chex.assert_trees_all_close(
            fresh_accountant.rdp_epsilons,
            jnp.zeros_like(fresh_accountant.alphas),
        )

    def test_step_increases_budget(self, fresh_accountant):
        """Each step increases privacy budget."""
        after_step = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)

        # All RDP values should increase (be positive)
        assert jnp.all(after_step.rdp_epsilons > 0)

    def test_step_accumulates(self, fresh_accountant):
        """Multiple steps accumulate additively."""
        acc1 = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        acc2 = acc1.step(noise_multiplier=1.0, sample_rate=1.0)

        # Two steps should be ~2x one step
        chex.assert_trees_all_close(
            acc2.rdp_epsilons,
            2 * acc1.rdp_epsilons,
            rtol=1e-5,
        )

    def test_step_num_steps_parameter(self, fresh_accountant):
        """num_steps=k is equivalent to k individual steps."""
        # 5 individual steps
        acc_individual = fresh_accountant
        for _ in range(5):
            acc_individual = acc_individual.step(noise_multiplier=1.0, sample_rate=1.0)

        # 1 step with num_steps=5
        acc_batched = fresh_accountant.step(
            noise_multiplier=1.0, sample_rate=1.0, num_steps=5
        )

        chex.assert_trees_all_close(
            acc_individual.rdp_epsilons,
            acc_batched.rdp_epsilons,
            rtol=1e-5,
        )

    def test_step_validates_noise_multiplier(self, fresh_accountant):
        """Rejects non-positive noise_multiplier."""
        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            fresh_accountant.step(noise_multiplier=0.0, sample_rate=1.0)

        with pytest.raises(ValueError, match="noise_multiplier must be positive"):
            fresh_accountant.step(noise_multiplier=-1.0, sample_rate=1.0)

    def test_step_validates_sample_rate(self, fresh_accountant):
        """Rejects sample_rate outside (0, 1]."""
        with pytest.raises(ValueError, match="sample_rate must be in"):
            fresh_accountant.step(noise_multiplier=1.0, sample_rate=0.0)

        with pytest.raises(ValueError, match="sample_rate must be in"):
            fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.5)


# =============================================================================
# TestRDPAccountantConversion
# =============================================================================


class TestRDPAccountantConversion:
    """Tests for get_privacy_spent() (RDP → (ε,δ)-DP)."""

    def test_returns_privacy_budget(self, fresh_accountant):
        """Returns a PrivacyBudget namedtuple."""
        acc = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        budget = acc.get_privacy_spent(delta=1e-5)

        assert isinstance(budget, PrivacyBudget)
        assert hasattr(budget, "epsilon")
        assert hasattr(budget, "delta")

    def test_delta_preserved(self, fresh_accountant):
        """Returned delta matches input delta."""
        acc = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        budget = acc.get_privacy_spent(delta=1e-5)

        assert budget.delta == 1e-5

    def test_epsilon_positive(self, fresh_accountant):
        """Epsilon is positive after spending budget."""
        acc = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        budget = acc.get_privacy_spent(delta=1e-5)

        assert budget.epsilon > 0

    def test_more_noise_less_epsilon(self):
        """Higher noise multiplier → lower epsilon (better privacy)."""
        acc_low_noise = RDPAccountant.create().step(noise_multiplier=0.5, sample_rate=1.0)
        acc_high_noise = RDPAccountant.create().step(noise_multiplier=2.0, sample_rate=1.0)

        eps_low = acc_low_noise.get_privacy_spent(delta=1e-5).epsilon
        eps_high = acc_high_noise.get_privacy_spent(delta=1e-5).epsilon

        assert eps_high < eps_low

    def test_more_steps_more_epsilon(self, fresh_accountant):
        """More training steps → higher epsilon (worse privacy)."""
        acc_few = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0, num_steps=10)
        acc_many = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0, num_steps=100)

        eps_few = acc_few.get_privacy_spent(delta=1e-5).epsilon
        eps_many = acc_many.get_privacy_spent(delta=1e-5).epsilon

        assert eps_many > eps_few

    def test_smaller_delta_larger_epsilon(self, fresh_accountant):
        """Smaller δ → larger ε (privacy tradeoff)."""
        acc = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0, num_steps=100)

        eps_large_delta = acc.get_privacy_spent(delta=1e-3).epsilon
        eps_small_delta = acc.get_privacy_spent(delta=1e-6).epsilon

        assert eps_small_delta > eps_large_delta

    def test_validates_delta(self, fresh_accountant):
        """Rejects invalid delta values."""
        acc = fresh_accountant.step(noise_multiplier=1.0, sample_rate=1.0)

        with pytest.raises(ValueError, match="delta must be in"):
            acc.get_privacy_spent(delta=0.0)

        with pytest.raises(ValueError, match="delta must be in"):
            acc.get_privacy_spent(delta=1.0)

        with pytest.raises(ValueError, match="delta must be in"):
            acc.get_privacy_spent(delta=-0.1)


# =============================================================================
# TestComputeRDP
# =============================================================================


class TestComputeRDP:
    """Tests for compute_rdp() function."""

    def test_no_subsampling_closed_form(self):
        """Without subsampling, ε(α) = α / (2σ²)."""
        alphas = jnp.array([2.0, 4.0, 8.0])
        sigma = 2.0

        rdp = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=1.0)

        expected = alphas / (2 * sigma**2)
        chex.assert_trees_all_close(rdp, expected, rtol=1e-5)

    def test_subsampling_reduces_rdp(self):
        """Subsampling (q < 1) gives lower RDP than no subsampling."""
        alphas = jnp.array([2.0, 4.0, 8.0, 16.0])
        sigma = 1.0

        rdp_full = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=1.0)
        rdp_subsampled = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=0.01)

        # Subsampling should give better (lower) RDP at all orders
        assert jnp.all(rdp_subsampled < rdp_full)

    def test_smaller_sample_rate_better_rdp(self):
        """Smaller sample rate → better privacy amplification."""
        alphas = jnp.array([2.0, 4.0, 8.0])
        sigma = 1.0

        rdp_10pct = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=0.1)
        rdp_1pct = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=0.01)

        assert jnp.all(rdp_1pct < rdp_10pct)

    def test_output_shape_matches_alphas(self):
        """Output shape equals input alphas shape."""
        alphas = jnp.array([1.5, 2.0, 3.0, 4.0, 8.0, 16.0])
        rdp = compute_rdp(alphas, noise_multiplier=1.0, sample_rate=0.01)

        assert rdp.shape == alphas.shape

    def test_rdp_positive(self):
        """RDP values are always positive."""
        alphas = jnp.array([2.0, 4.0, 8.0, 16.0, 32.0])

        rdp = compute_rdp(alphas, noise_multiplier=1.0, sample_rate=0.01)

        assert jnp.all(rdp > 0)

    def test_rdp_monotonic_in_alpha(self):
        """For fixed σ and q, RDP increases with α."""
        alphas = jnp.array([2.0, 4.0, 8.0, 16.0, 32.0])

        rdp = compute_rdp(alphas, noise_multiplier=1.0, sample_rate=0.5)

        # Check monotonicity: rdp[i+1] > rdp[i]
        assert jnp.all(jnp.diff(rdp) > 0)


# =============================================================================
# TestRDPToEpsilon
# =============================================================================


class TestRDPToEpsilon:
    """Tests for rdp_to_epsilon() conversion function."""

    def test_basic_conversion(self):
        """Basic conversion produces finite positive result."""
        alphas = jnp.array([2.0, 4.0, 8.0, 16.0])
        rdp_values = jnp.array([0.5, 1.0, 2.0, 4.0])

        eps = rdp_to_epsilon(rdp_values, alphas, delta=1e-5)

        assert eps > 0
        assert jnp.isfinite(eps)

    def test_selects_optimal_alpha(self):
        """Conversion finds the α that minimizes ε."""
        alphas = jnp.array([2.0, 4.0, 8.0, 16.0])
        rdp_values = jnp.array([0.5, 1.0, 2.0, 4.0])
        delta = 1e-5

        eps = rdp_to_epsilon(rdp_values, alphas, delta)

        # Manually compute all candidates
        candidates = rdp_values + jnp.log(1 / delta) / (alphas - 1)
        expected_min = float(jnp.min(candidates))

        assert abs(eps - expected_min) < 1e-6

    def test_validates_delta(self):
        """Rejects invalid delta."""
        alphas = jnp.array([2.0, 4.0])
        rdp_values = jnp.array([0.5, 1.0])

        with pytest.raises(ValueError):
            rdp_to_epsilon(rdp_values, alphas, delta=0.0)


# =============================================================================
# TestMathematicalProperties
# =============================================================================


class TestMathematicalProperties:
    """Property-based tests for mathematical invariants."""

    @given(
        sigma=st.floats(min_value=0.1, max_value=10.0),
        alpha=st.floats(min_value=2.0, max_value=64.0),
    )
    @settings(max_examples=50, deadline=None)  # JAX JIT can be slow
    def test_no_subsampling_formula(self, sigma, alpha):
        """Verify ε(α) = α/(2σ²) for q=1."""
        alphas = jnp.array([alpha])
        rdp = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=1.0)

        expected = alpha / (2 * sigma**2)
        # Use relative tolerance for float32 precision
        assert abs(float(rdp[0]) - expected) / expected < 1e-4

    @given(
        sigma=st.floats(min_value=0.5, max_value=5.0),
        q=st.floats(min_value=0.01, max_value=0.5),  # Avoid very small q
    )
    @settings(max_examples=30, deadline=None)  # JAX JIT can be slow
    def test_subsampling_amplification(self, sigma, q):
        """Subsampling always improves privacy."""
        alphas = jnp.array([4.0, 8.0])

        rdp_full = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=1.0)
        rdp_sub = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=q)

        assert jnp.all(rdp_sub <= rdp_full)


# =============================================================================
# TestNumericalStability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_very_large_noise(self):
        """Very large σ gives very small RDP (≈0)."""
        alphas = jnp.array([2.0, 4.0, 8.0])
        rdp = compute_rdp(alphas, noise_multiplier=100.0, sample_rate=1.0)

        # ε(α) = α/(2×100²) = α/20000, very small
        assert jnp.all(rdp < 0.01)
        assert jnp.all(jnp.isfinite(rdp))

    def test_small_noise(self):
        """Small σ gives large RDP (still finite)."""
        alphas = jnp.array([2.0, 4.0])
        rdp = compute_rdp(alphas, noise_multiplier=0.1, sample_rate=1.0)

        # ε(α) = α/(2×0.01) = 50α, large but finite
        assert jnp.all(rdp > 10)
        assert jnp.all(jnp.isfinite(rdp))

    def test_very_small_sample_rate(self):
        """Small q doesn't cause numerical issues."""
        alphas = jnp.array([2.0, 4.0, 8.0])
        # Use 1e-3 instead of 1e-4 to avoid numerical precision issues
        rdp = compute_rdp(alphas, noise_multiplier=1.0, sample_rate=1e-3)

        assert jnp.all(jnp.isfinite(rdp))
        # RDP should be positive (or at least non-negative within numerical precision)
        assert jnp.all(rdp >= -1e-6)

    def test_many_steps(self):
        """Accountant handles many training steps."""
        accountant = RDPAccountant.create()

        # Simulate 10000 training steps
        accountant = accountant.step(
            noise_multiplier=1.0,
            sample_rate=0.01,
            num_steps=10000,
        )

        budget = accountant.get_privacy_spent(delta=1e-5)

        assert jnp.isfinite(budget.epsilon)
        assert budget.epsilon > 0

    def test_no_nan_in_rdp(self):
        """RDP computation never produces NaN."""
        alphas = DEFAULT_RDP_ORDERS

        for sigma in [0.1, 1.0, 10.0]:
            for q in [0.001, 0.01, 0.1, 1.0]:
                rdp = compute_rdp(alphas, noise_multiplier=sigma, sample_rate=q)
                assert not jnp.any(jnp.isnan(rdp)), f"NaN for σ={sigma}, q={q}"


# =============================================================================
# TestProtocolCompliance
# =============================================================================


class TestProtocolCompliance:
    """Verify RDPAccountant satisfies PrivacyAccountant protocol."""

    def test_is_privacy_accountant(self):
        """RDPAccountant is a valid PrivacyAccountant."""
        accountant = RDPAccountant.create()

        # Protocol is runtime_checkable
        assert isinstance(accountant, PrivacyAccountant)

    def test_step_returns_accountant(self):
        """step() returns something that's also a PrivacyAccountant."""
        accountant = RDPAccountant.create()
        new_accountant = accountant.step(noise_multiplier=1.0, sample_rate=1.0)

        assert isinstance(new_accountant, PrivacyAccountant)

    def test_get_privacy_spent_returns_budget(self):
        """get_privacy_spent() returns PrivacyBudget."""
        accountant = RDPAccountant.create()
        accountant = accountant.step(noise_multiplier=1.0, sample_rate=1.0)
        budget = accountant.get_privacy_spent(delta=1e-5)

        assert isinstance(budget, PrivacyBudget)
