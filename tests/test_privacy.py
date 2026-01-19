"""Tests for differential privacy components: clipping, noise, and dp_gradients."""

import chex
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jax_prism.privacy import (
    add_noise,
    clip_gradients,
    clip_single_gradient,
    compute_global_norm,
    compute_per_sample_gradients,
    dp_gradients,
    generate_noise_tree,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_grads():
    """Simple gradient pytree for testing."""
    return {"w": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}


@pytest.fixture
def batch_grads():
    """Per-sample gradients with batch dimension."""
    return {
        "w": jnp.array([[3.0, 4.0], [6.0, 8.0], [1.0, 0.0]]),  # (3, 2)
        "b": jnp.array([[0.0], [0.0], [1.0]]),  # (3, 1)
    }


@pytest.fixture
def key():
    """PRNG key for tests."""
    return jax.random.key(42)


# =============================================================================
# Clipping Tests
# =============================================================================


class TestComputeGlobalNorm:
    """Tests for compute_global_norm."""

    def test_simple_vector(self):
        """Norm of [3, 4] is 5."""
        grads = {"w": jnp.array([3.0, 4.0])}
        norm = compute_global_norm(grads)
        chex.assert_trees_all_close(norm, 5.0)

    def test_pytree_norm(self, simple_grads):
        """Global norm across multiple leaves."""
        # [3, 4, 0] has norm sqrt(9 + 16 + 0) = 5
        norm = compute_global_norm(simple_grads)
        chex.assert_trees_all_close(norm, 5.0)

    def test_nested_pytree(self):
        """Works with nested structures."""
        grads = {
            "layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([2.0])},
            "layer2": {"w": jnp.array([0.0])},
        }
        # sqrt(1 + 4 + 4 + 0) = 3
        norm = compute_global_norm(grads)
        chex.assert_trees_all_close(norm, 3.0)

    def test_zero_norm(self):
        """Zero gradients have zero norm."""
        grads = {"w": jnp.zeros(10), "b": jnp.zeros(5)}
        norm = compute_global_norm(grads)
        chex.assert_trees_all_close(norm, 0.0)


class TestClipSingleGradient:
    """Tests for clip_single_gradient."""

    def test_no_clipping_needed(self, simple_grads):
        """Gradients within norm are unchanged."""
        # Norm is 5, clip at 10 -> no change
        clipped = clip_single_gradient(simple_grads, clip_norm=10.0)
        chex.assert_trees_all_close(clipped, simple_grads)

    def test_clipping_applied(self, simple_grads):
        """Gradients exceeding norm are scaled down."""
        # Norm is 5, clip at 1 -> scale by 1/5
        clipped = clip_single_gradient(simple_grads, clip_norm=1.0)

        expected = {"w": jnp.array([0.6, 0.8]), "b": jnp.array([0.0])}
        chex.assert_trees_all_close(clipped, expected)

        # Verify norm is now 1
        new_norm = compute_global_norm(clipped)
        chex.assert_trees_all_close(new_norm, 1.0)

    def test_exact_clip_norm(self, simple_grads):
        """Gradients exactly at clip norm are unchanged."""
        clipped = clip_single_gradient(simple_grads, clip_norm=5.0)
        chex.assert_trees_all_close(clipped, simple_grads)

    def test_preserves_direction(self):
        """Clipping preserves gradient direction."""
        grads = {"w": jnp.array([6.0, 8.0])}  # norm = 10
        clipped = clip_single_gradient(grads, clip_norm=5.0)

        # Direction should be same: [0.6, 0.8]
        original_dir = grads["w"] / compute_global_norm(grads)
        clipped_dir = clipped["w"] / compute_global_norm(clipped)
        chex.assert_trees_all_close(original_dir, clipped_dir)

    def test_zero_gradients(self):
        """Zero gradients stay zero."""
        grads = {"w": jnp.zeros(5)}
        clipped = clip_single_gradient(grads, clip_norm=1.0)
        chex.assert_trees_all_close(clipped, grads)


class TestClipGradients:
    """Tests for clip_gradients (batched version)."""

    def test_batch_clipping(self, batch_grads):
        """Per-sample clipping and averaging."""
        # Sample 0: norm = 5, Sample 1: norm = 10, Sample 2: norm = sqrt(2)
        clipped = clip_gradients(batch_grads, clip_norm=1.0)

        # Should remove batch dimension
        assert clipped["w"].shape == (2,)
        assert clipped["b"].shape == (1,)

    def test_output_shape(self, batch_grads):
        """Batch dimension is removed after aggregation."""
        clipped = clip_gradients(batch_grads, clip_norm=1.0)

        # Input: (3, 2), (3, 1) -> Output: (2,), (1,)
        chex.assert_shape(clipped["w"], (2,))
        chex.assert_shape(clipped["b"], (1,))

    def test_bounded_contribution(self, batch_grads):
        """Each sample's contribution is bounded."""
        # With clip_norm=1.0, each clipped sample has norm <= 1
        # So the average should have norm <= 1
        clipped = clip_gradients(batch_grads, clip_norm=1.0)
        norm = compute_global_norm(clipped)
        assert norm <= 1.0 + 1e-6

    def test_single_sample_batch(self):
        """Works with batch size 1."""
        grads = {"w": jnp.array([[3.0, 4.0]])}  # (1, 2)
        clipped = clip_gradients(grads, clip_norm=1.0)

        chex.assert_shape(clipped["w"], (2,))
        norm = compute_global_norm(clipped)
        chex.assert_trees_all_close(norm, 1.0)


# =============================================================================
# Noise Tests
# =============================================================================


class TestAddNoise:
    """Tests for add_noise."""

    def test_output_shape(self, simple_grads, key):
        """Noise doesn't change shape."""
        noisy = add_noise(simple_grads, noise_multiplier=1.0, clip_norm=1.0, key=key)
        chex.assert_trees_all_equal_shapes(noisy, simple_grads)

    def test_noise_is_added(self, simple_grads, key):
        """Output differs from input (unless noise_multiplier=0)."""
        noisy = add_noise(simple_grads, noise_multiplier=1.0, clip_norm=1.0, key=key)

        # Should not be equal (with high probability)
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(noisy, simple_grads)

    def test_zero_noise_multiplier(self, simple_grads, key):
        """noise_multiplier=0 means no noise."""
        noisy = add_noise(simple_grads, noise_multiplier=0.0, clip_norm=1.0, key=key)
        chex.assert_trees_all_close(noisy, simple_grads)

    def test_noise_scale(self, key):
        """Noise std is noise_multiplier * clip_norm."""
        grads = {"w": jnp.zeros((10000,))}  # Large array for statistics
        noisy = add_noise(grads, noise_multiplier=2.0, clip_norm=3.0, key=key)

        # Expected std = 2.0 * 3.0 = 6.0
        empirical_std = jnp.std(noisy["w"])
        assert abs(float(empirical_std) - 6.0) < 0.5  # Allow some variance

    def test_reproducibility(self, simple_grads, key):
        """Same key gives same noise."""
        noisy1 = add_noise(simple_grads, noise_multiplier=1.0, clip_norm=1.0, key=key)
        noisy2 = add_noise(simple_grads, noise_multiplier=1.0, clip_norm=1.0, key=key)
        chex.assert_trees_all_close(noisy1, noisy2)

    def test_different_keys_different_noise(self, simple_grads, key):
        """Different keys give different noise."""
        key1, key2 = jax.random.split(key)
        noisy1 = add_noise(simple_grads, noise_multiplier=1.0, clip_norm=1.0, key=key1)
        noisy2 = add_noise(simple_grads, noise_multiplier=1.0, clip_norm=1.0, key=key2)

        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(noisy1, noisy2)


class TestGenerateNoiseTree:
    """Tests for generate_noise_tree."""

    def test_output_structure(self, simple_grads, key):
        """Generated noise has same structure as template."""
        noise = generate_noise_tree(simple_grads, noise_std=1.0, key=key)
        chex.assert_trees_all_equal_shapes(noise, simple_grads)

    def test_noise_statistics(self, key):
        """Generated noise has correct statistics."""
        template = {"w": jnp.zeros((10000,))}
        noise = generate_noise_tree(template, noise_std=2.5, key=key)

        # Mean should be ~0, std should be ~2.5
        assert abs(float(jnp.mean(noise["w"]))) < 0.1
        assert abs(float(jnp.std(noise["w"])) - 2.5) < 0.2

    def test_zero_std(self, simple_grads, key):
        """noise_std=0 gives zeros."""
        noise = generate_noise_tree(simple_grads, noise_std=0.0, key=key)
        expected = jax.tree_util.tree_map(jnp.zeros_like, simple_grads)
        chex.assert_trees_all_close(noise, expected)


# =============================================================================
# DP Gradients Integration Tests
# =============================================================================


class TestDPGradients:
    """Integration tests for dp_gradients."""

    def test_full_pipeline(self, batch_grads, key):
        """dp_gradients clips and adds noise correctly."""
        dp_grads = dp_gradients(
            batch_grads,
            clip_norm=1.0,
            noise_multiplier=0.5,
            key=key,
        )

        # Output should have batch dimension removed
        chex.assert_shape(dp_grads["w"], (2,))
        chex.assert_shape(dp_grads["b"], (1,))

    def test_zero_noise(self, batch_grads, key):
        """With zero noise, dp_gradients = clip_gradients."""
        dp_grads = dp_gradients(
            batch_grads,
            clip_norm=1.0,
            noise_multiplier=0.0,
            key=key,
        )

        expected = clip_gradients(batch_grads, clip_norm=1.0)
        chex.assert_trees_all_close(dp_grads, expected)

    def test_jit_compatible(self, batch_grads, key):
        """dp_gradients can be JIT compiled."""
        dp_fn = jax.jit(
            lambda g, k: dp_gradients(g, clip_norm=1.0, noise_multiplier=1.0, key=k)
        )
        result = dp_fn(batch_grads, key)
        assert result["w"].shape == (2,)


class TestComputePerSampleGradients:
    """Tests for compute_per_sample_gradients."""

    def test_simple_loss(self):
        """Per-sample gradients from a simple loss."""
        # Simple linear model: loss = (w * x - y)^2
        def loss_fn(params, example):
            x, y = example
            pred = params["w"] * x
            return jnp.mean((pred - y) ** 2)

        params = {"w": jnp.array(1.0)}
        batch = (jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 4.0, 6.0]))

        per_sample_grads = compute_per_sample_gradients(loss_fn, params, batch)

        # Should have batch dimension
        assert per_sample_grads["w"].shape == (3,)

    def test_mlp_loss(self):
        """Per-sample gradients for an MLP-style loss."""
        def loss_fn(params, example):
            x, y = example
            h = jnp.tanh(params["w1"] @ x + params["b1"])
            pred = params["w2"] @ h + params["b2"]
            return jnp.sum((pred - y) ** 2)

        # Small MLP: 2 -> 3 -> 1
        params = {
            "w1": jnp.ones((3, 2)),
            "b1": jnp.zeros(3),
            "w2": jnp.ones((1, 3)),
            "b2": jnp.zeros(1),
        }

        # Batch of 4 examples
        batch = (
            jnp.ones((4, 2)),  # inputs: (4, 2)
            jnp.ones((4, 1)),  # targets: (4, 1)
        )

        per_sample_grads = compute_per_sample_gradients(loss_fn, params, batch)

        # Each gradient should have batch dimension
        assert per_sample_grads["w1"].shape == (4, 3, 2)
        assert per_sample_grads["b1"].shape == (4, 3)
        assert per_sample_grads["w2"].shape == (4, 1, 3)
        assert per_sample_grads["b2"].shape == (4, 1)


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestMathematicalProperties:
    """Property-based tests for mathematical invariants."""

    @given(
        clip_norm=st.floats(min_value=0.1, max_value=10.0),
        scale=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_clipping_bounds_norm(self, clip_norm, scale):
        """Clipped gradients always have norm <= clip_norm."""
        grads = {"w": jnp.array([3.0 * scale, 4.0 * scale])}
        clipped = clip_single_gradient(grads, clip_norm=clip_norm)
        norm = compute_global_norm(clipped)
        assert float(norm) <= clip_norm + 1e-5

    @given(
        noise_mult=st.floats(min_value=0.1, max_value=5.0),
        clip_norm=st.floats(min_value=0.1, max_value=5.0),
    )
    @settings(max_examples=30, deadline=None)
    def test_noise_scale_property(self, noise_mult, clip_norm):
        """Noise standard deviation is noise_mult * clip_norm."""
        key = jax.random.key(0)
        grads = {"w": jnp.zeros((5000,))}

        noisy = add_noise(grads, noise_mult, clip_norm, key)
        empirical_std = float(jnp.std(noisy["w"]))
        expected_std = noise_mult * clip_norm

        # Allow 20% relative tolerance for statistical variance
        assert abs(empirical_std - expected_std) / expected_std < 0.2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_leaves(self):
        """Handle pytrees with empty arrays."""
        grads = {"w": jnp.array([1.0, 2.0]), "empty": jnp.array([])}
        norm = compute_global_norm(grads)
        # Empty array contributes 0 to norm
        chex.assert_trees_all_close(norm, jnp.sqrt(5.0))

    def test_scalar_leaves(self):
        """Handle pytrees with scalar leaves."""
        grads = {"a": jnp.array(3.0), "b": jnp.array(4.0)}
        norm = compute_global_norm(grads)
        chex.assert_trees_all_close(norm, 5.0)

    def test_deeply_nested(self, key):
        """Handle deeply nested pytrees."""
        grads = {
            "encoder": {
                "layer1": {"w": jnp.ones((2, 2)), "b": jnp.zeros(2)},
                "layer2": {"w": jnp.ones((2, 2)), "b": jnp.zeros(2)},
            },
            "decoder": {"w": jnp.ones((2, 2))},
        }

        # Should work without error
        norm = compute_global_norm(grads)
        clipped = clip_single_gradient(grads, clip_norm=1.0)
        noisy = add_noise(clipped, noise_multiplier=1.0, clip_norm=1.0, key=key)

        # Structure preserved
        assert "encoder" in noisy
        assert "layer1" in noisy["encoder"]

    def test_large_batch(self, key):
        """Handle large batch sizes."""
        batch_grads = {"w": jnp.ones((1000, 10))}
        dp_grads = dp_gradients(
            batch_grads,
            clip_norm=1.0,
            noise_multiplier=1.0,
            key=key,
        )
        assert dp_grads["w"].shape == (10,)
