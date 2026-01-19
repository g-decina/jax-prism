"""Tests for SwiGLU feedforward module."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.models.components.swiglu import SwiGLU


class TestSwiGLUShapes:
    """Tests for output shapes."""

    def test_output_shape_matches_input(self, rng_key):
        """Output should have same shape as input."""
        model = SwiGLU(hidden_dim=256)
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 128))

    def test_output_shape_2d(self, rng_key):
        """Works with 2D input (batch, features)."""
        model = SwiGLU(hidden_dim=64)
        x = jax.random.normal(rng_key, (4, 32))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (4, 32))

    def test_output_shape_4d(self, rng_key):
        """Works with 4D input."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 4, 8, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 4, 8, 64))

    def test_different_hidden_dims(self, rng_key):
        """Various hidden_dim values work correctly."""
        x = jax.random.normal(rng_key, (2, 16, 64))

        for hidden_dim in [32, 64, 128, 256]:
            model = SwiGLU(hidden_dim=hidden_dim)
            params = model.init(rng_key, x)
            y = model.apply(params, x)
            chex.assert_shape(y, (2, 16, 64))


class TestSwiGLUParameters:
    """Tests for parameter structure."""

    def test_has_three_projections(self, rng_key):
        """Should have gate_proj, up_proj, and down_proj."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        assert "gate_proj" in params["params"]
        assert "up_proj" in params["params"]
        assert "down_proj" in params["params"]

    def test_projection_shapes(self, rng_key):
        """Projection kernels have correct shapes."""
        d_model = 64
        hidden_dim = 128
        model = SwiGLU(hidden_dim=hidden_dim)
        x = jax.random.normal(rng_key, (2, 16, d_model))

        params = model.init(rng_key, x)

        # gate and up: (d_model, hidden_dim)
        chex.assert_shape(params["params"]["gate_proj"]["kernel"], (d_model, hidden_dim))
        chex.assert_shape(params["params"]["up_proj"]["kernel"], (d_model, hidden_dim))
        # down: (hidden_dim, d_model)
        chex.assert_shape(params["params"]["down_proj"]["kernel"], (hidden_dim, d_model))

    def test_no_bias_by_default(self, rng_key):
        """use_bias=False should produce no bias parameters."""
        model = SwiGLU(hidden_dim=128, use_bias=False)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        assert "bias" not in params["params"]["gate_proj"]
        assert "bias" not in params["params"]["up_proj"]
        assert "bias" not in params["params"]["down_proj"]

    def test_with_bias(self, rng_key):
        """use_bias=True should produce bias parameters."""
        model = SwiGLU(hidden_dim=128, use_bias=True)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        assert "bias" in params["params"]["gate_proj"]
        assert "bias" in params["params"]["up_proj"]
        assert "bias" in params["params"]["down_proj"]


class TestSwiGLUBehavior:
    """Tests for correct SwiGLU behavior."""

    def test_gating_effect(self, rng_key):
        """Gate should modulate the signal (output != simple linear transform)."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        # Output should be different from input (non-trivial transform)
        assert not jnp.allclose(y, x)

    def test_nonzero_output(self, rng_key):
        """Should produce non-zero output for random input."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.allclose(y, 0)

    def test_zero_input_produces_zero_output(self, rng_key):
        """Zero input should produce zero output (no bias, silu(0)=0)."""
        model = SwiGLU(hidden_dim=128, use_bias=False)
        x = jnp.zeros((2, 16, 64))

        params = model.init(rng_key, jax.random.normal(rng_key, x.shape))
        y = model.apply(params, x)

        chex.assert_trees_all_close(y, jnp.zeros_like(y), atol=1e-7)


class TestJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """SwiGLU should be JIT-compilable."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x)

        y = forward(params, x)
        chex.assert_shape(y, (2, 16, 64))

    def test_vmap_compatible(self, rng_key):
        """SwiGLU should work with vmap."""
        model = SwiGLU(hidden_dim=128)
        x_single = jax.random.normal(rng_key, (1, 16, 64))
        x_batch = jax.random.normal(rng_key, (4, 16, 64))

        params = model.init(rng_key, x_single)

        # Direct batch
        y_direct = model.apply(params, x_batch)

        # Via vmap
        vmapped = jax.vmap(lambda x: model.apply(params, x[None])[0])
        y_vmap = vmapped(x_batch)

        chex.assert_trees_all_close(y_direct, y_vmap, atol=1e-5)

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through SwiGLU."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        def loss_fn(params):
            y = model.apply(params, x)
            return jnp.mean(y ** 2)

        grads = jax.grad(loss_fn)(params)

        # All projection gradients should exist and be non-zero
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            assert proj in grads["params"]
            assert not jnp.allclose(grads["params"][proj]["kernel"], 0)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_normal_input(self, rng_key):
        """Normal input should not produce NaN."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 32, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_no_nan_large_input(self, rng_key):
        """Large input values should not cause overflow."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64)) * 100

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_no_nan_small_input(self, rng_key):
        """Small input values should not cause issues."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64)) * 1e-6

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, rng_key):
        """Same input should produce identical output."""
        model = SwiGLU(hidden_dim=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x)
        y2 = model.apply(params, x)

        chex.assert_trees_all_close(y1, y2, atol=0)