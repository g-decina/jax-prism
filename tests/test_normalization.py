"""Tests for normalization components."""

import chex
import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from jax_prism.models.components.normalization import RMSNorm


class TestRMSNorm:
    """Tests for RMSNorm module."""

    def test_output_shape_matches_input(self, rng_key):
        """Output shape should equal input shape."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 8, 64))  # (batch, seq, features)

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, x.shape)

    def test_output_shape_2d(self, rng_key):
        """Works with 2D input (batch, features)."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (4, 32))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (4, 32))

    def test_output_shape_4d(self, rng_key):
        """Works with 4D input (batch, height, width, features)."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 4, 4, 16))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 4, 4, 16))

    def test_rms_normalized_to_one(self, rng_key):
        """After normalization (before scaling), RMS should be ~1.

        With scale initialized to ones, output RMS should be approximately 1.
        """
        model = RMSNorm()
        x = jax.random.normal(rng_key, (8, 64)) * 10 + 5  # shifted and scaled

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        # Compute RMS of each row
        rms_per_row = jnp.sqrt(jnp.mean(y ** 2, axis=-1))

        # Should be close to 1 (scale is initialized to ones)
        chex.assert_trees_all_close(rms_per_row, jnp.ones(8), atol=1e-5)

    def test_scale_parameter_shape(self, rng_key):
        """Scale parameter should have shape (features,)."""
        model = RMSNorm()
        features = 64
        x = jax.random.normal(rng_key, (2, 8, features))

        params = model.init(rng_key, x)

        assert "scale" in params["params"]
        chex.assert_shape(params["params"]["scale"], (features,))

    def test_scale_initialized_to_ones(self, rng_key):
        """Scale parameter should be initialized to ones."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 32))

        params = model.init(rng_key, x)

        chex.assert_trees_all_close(
            params["params"]["scale"],
            jnp.ones(32),
            atol=1e-7,
        )

    def test_scale_affects_output(self, rng_key):
        """Modifying scale should proportionally affect output."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 16))

        params = model.init(rng_key, x)
        y1 = model.apply(params, x)

        # Double the scale
        from flax.core import freeze, unfreeze
        mutable_params = unfreeze(params)
        mutable_params["params"]["scale"] = params["params"]["scale"] * 2
        scaled_params = freeze(mutable_params)
        y2 = model.apply(scaled_params, x)

        chex.assert_trees_all_close(y2, y1 * 2, atol=1e-6)

    def test_numerical_stability_small_values(self, rng_key):
        """Should handle very small input values without NaN."""
        model = RMSNorm(epsilon=1e-6)
        x = jax.random.normal(rng_key, (2, 32)) * 1e-10

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_numerical_stability_large_values(self, rng_key):
        """Should handle large input values without overflow."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 32)) * 1e6

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_numerical_stability_zero_input(self, rng_key):
        """Should handle all-zero input gracefully (epsilon prevents div by zero)."""
        model = RMSNorm(epsilon=1e-6)
        x = jnp.zeros((2, 32))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        # Output should be zeros (0 / epsilon * scale = 0)
        chex.assert_trees_all_close(y, jnp.zeros((2, 32)), atol=1e-7)

    def test_custom_epsilon(self, rng_key):
        """Should respect custom epsilon value."""
        model = RMSNorm(epsilon=1e-3)
        x = jnp.zeros((2, 32))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        # With zeros input and epsilon=1e-3, RMS = sqrt(0 + 1e-3)
        # Just verify it runs without error and produces zeros
        chex.assert_trees_all_close(y, jnp.zeros((2, 32)), atol=1e-7)

    def test_jit_compatible(self, rng_key):
        """RMSNorm should be JIT-compilable."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 32))

        params = model.init(rng_key, x)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x)

        y = forward(params, x)
        chex.assert_shape(y, (2, 32))

    def test_vmap_compatible(self, rng_key):
        """RMSNorm should work with vmap over batch dimension."""
        model = RMSNorm()

        # Single example
        x_single = jax.random.normal(rng_key, (16,))
        params = model.init(rng_key, x_single)

        # Batched via vmap
        x_batch = jax.random.normal(rng_key, (4, 16))

        vmapped_apply = jax.vmap(lambda x: model.apply(params, x))
        y_vmap = vmapped_apply(x_batch)

        # Direct batched application
        y_direct = model.apply(params, x_batch)

        chex.assert_trees_all_close(y_vmap, y_direct, atol=1e-6)

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through RMSNorm."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 32))

        params = model.init(rng_key, x)

        def loss_fn(params):
            y = model.apply(params, x)
            return jnp.mean(y ** 2)

        grads = jax.grad(loss_fn)(params)

        # Scale gradient should exist and not be zero
        assert "scale" in grads["params"]
        assert not jnp.allclose(grads["params"]["scale"], 0)

    def test_deterministic(self, rng_key):
        """Same input should produce same output (no stochasticity)."""
        model = RMSNorm()
        x = jax.random.normal(rng_key, (2, 32))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x)
        y2 = model.apply(params, x)

        chex.assert_trees_all_close(y1, y2, atol=0)
