"""Tests for TFT components: GRN and VariableSelectionNetwork."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.models.tft.components import GRN, VSNOutput, VariableSelectionNetwork


class TestGRNShapes:
    """Tests for GRN output shapes."""

    def test_output_shape_same_dim(self, rng_key):
        """Output shape when input_dim == hidden_size."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 64))

    def test_output_shape_different_dim(self, rng_key):
        """Output shape when input_dim != hidden_size (skip projection)."""
        model = GRN(hidden_size=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 128))

    def test_output_shape_2d(self, rng_key):
        """Works with 2D input (batch, features)."""
        model = GRN(hidden_size=32)
        x = jax.random.normal(rng_key, (4, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (4, 32))


class TestGRNContext:
    """Tests for GRN context injection."""

    def test_with_context(self, rng_key):
        """GRN should accept and use context."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))
        context = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, 16, 32))

        params = model.init(rng_key, x, context=context)
        y = model.apply(params, x, context=context)

        chex.assert_shape(y, (2, 16, 64))

    def test_context_affects_output(self, rng_key):
        """Different contexts should produce different outputs."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))
        context1 = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, 16, 32))
        context2 = jax.random.normal(jax.random.fold_in(rng_key, 2), (2, 16, 32))

        params = model.init(rng_key, x, context=context1)
        y1 = model.apply(params, x, context=context1)
        y2 = model.apply(params, x, context=context2)

        assert not jnp.allclose(y1, y2)

    def test_without_context(self, rng_key):
        """GRN should work without context."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))


class TestGRNParameters:
    """Tests for GRN parameter structure."""

    def test_has_skip_projection_when_needed(self, rng_key):
        """Skip projection exists when input_dim != hidden_size."""
        model = GRN(hidden_size=128)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        assert "skip_proj" in params["params"]

    def test_no_skip_projection_when_same_dim(self, rng_key):
        """No skip projection when input_dim == hidden_size."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        assert "skip_proj" not in params["params"]

    def test_has_context_projection_when_used(self, rng_key):
        """Context projection exists when context is provided."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))
        context = jax.random.normal(rng_key, (2, 16, 32))

        params = model.init(rng_key, x, context=context)

        assert "context_proj" in params["params"]


class TestGRNDropout:
    """Tests for GRN dropout behavior."""

    def test_deterministic_mode(self, rng_key):
        """deterministic=True should be deterministic."""
        model = GRN(hidden_size=64, dropout_rate=0.5)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x, deterministic=True)
        y2 = model.apply(params, x, deterministic=True)

        chex.assert_trees_all_close(y1, y2, atol=0)

    def test_training_mode_differs(self, rng_key):
        """deterministic=False with different RNG should produce different outputs."""
        model = GRN(hidden_size=64, dropout_rate=0.5)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x, deterministic=False, rngs={"dropout": jax.random.key(0)})
        y2 = model.apply(params, x, deterministic=False, rngs={"dropout": jax.random.key(1)})

        assert not jnp.allclose(y1, y2)


class TestGRNJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """GRN should be JIT-compilable."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x)

        y = forward(params, x)
        chex.assert_shape(y, (2, 16, 64))

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through GRN."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)

        def loss_fn(params):
            y = model.apply(params, x)
            return jnp.mean(y**2)

        grads = jax.grad(loss_fn)(params)

        assert "fc1" in grads["params"]
        assert not jnp.allclose(grads["params"]["fc1"]["kernel"], 0)


# ============================================================================
# VSN Tests
# ============================================================================


class TestVSNShapes:
    """Tests for VariableSelectionNetwork output shapes."""

    def test_output_shapes(self, rng_key):
        """VSN should return correct shapes."""
        num_features = 5
        hidden_size = 64
        model = VariableSelectionNetwork(hidden_size=hidden_size)
        features = jax.random.normal(rng_key, (2, 16, num_features, 32))

        params = model.init(rng_key, features)
        output = model.apply(params, features)

        assert isinstance(output, VSNOutput)
        chex.assert_shape(output.selected, (2, 16, hidden_size))
        chex.assert_shape(output.weights, (2, 16, num_features))

    def test_different_num_features(self, rng_key):
        """Works with different numbers of features."""
        hidden_size = 64

        for num_features in [2, 5, 10]:
            model = VariableSelectionNetwork(hidden_size=hidden_size)
            features = jax.random.normal(rng_key, (2, 16, num_features, 32))

            params = model.init(rng_key, features)
            output = model.apply(params, features)

            chex.assert_shape(output.weights, (2, 16, num_features))


class TestVSNWeights:
    """Tests for VSN weight properties."""

    def test_weights_sum_to_one(self, rng_key):
        """Feature weights should sum to 1 (softmax)."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))

        params = model.init(rng_key, features)
        output = model.apply(params, features)

        weight_sums = jnp.sum(output.weights, axis=-1)
        chex.assert_trees_all_close(weight_sums, jnp.ones((2, 16)), atol=1e-5)

    def test_weights_non_negative(self, rng_key):
        """Feature weights should be non-negative (softmax)."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))

        params = model.init(rng_key, features)
        output = model.apply(params, features)

        assert jnp.all(output.weights >= 0)

    def test_weights_change_with_input(self, rng_key):
        """Different inputs should produce different weights."""
        model = VariableSelectionNetwork(hidden_size=64)
        features1 = jax.random.normal(rng_key, (2, 16, 5, 32))
        features2 = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, 16, 5, 32))

        params = model.init(rng_key, features1)
        output1 = model.apply(params, features1)
        output2 = model.apply(params, features2)

        assert not jnp.allclose(output1.weights, output2.weights)


class TestVSNContext:
    """Tests for VSN context injection."""

    def test_with_context(self, rng_key):
        """VSN should accept and use context."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))
        context = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, 16, 48))

        params = model.init(rng_key, features, context=context)
        output = model.apply(params, features, context=context)

        assert isinstance(output, VSNOutput)
        chex.assert_shape(output.selected, (2, 16, 64))

    def test_context_affects_output(self, rng_key):
        """Different contexts should produce different outputs."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))
        context1 = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, 16, 48))
        context2 = jax.random.normal(jax.random.fold_in(rng_key, 2), (2, 16, 48))

        params = model.init(rng_key, features, context=context1)
        output1 = model.apply(params, features, context=context1)
        output2 = model.apply(params, features, context=context2)

        assert not jnp.allclose(output1.selected, output2.selected)


class TestVSNJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """VSN should be JIT-compilable."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))

        params = model.init(rng_key, features)

        @jax.jit
        def forward(params, features):
            return model.apply(params, features)

        output = forward(params, features)
        chex.assert_shape(output.selected, (2, 16, 64))

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through VSN."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))

        params = model.init(rng_key, features)

        def loss_fn(params):
            output = model.apply(params, features)
            return jnp.mean(output.selected**2)

        grads = jax.grad(loss_fn)(params)

        assert "input_proj" in grads["params"]
        assert not jnp.allclose(grads["params"]["input_proj"]["kernel"], 0)


class TestVSNOutput:
    """Tests for VSNOutput dataclass."""

    def test_is_pytree(self, rng_key):
        """VSNOutput should be a valid pytree."""
        output = VSNOutput(
            selected=jnp.ones((2, 16, 64)), weights=jnp.ones((2, 16, 5)) / 5
        )

        # Should work with tree_map
        doubled = jax.tree_util.tree_map(lambda x: x * 2, output)
        chex.assert_trees_all_close(doubled.selected, jnp.ones((2, 16, 64)) * 2)

    def test_fields_accessible(self, rng_key):
        """VSNOutput fields should be accessible."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))

        params = model.init(rng_key, features)
        output = model.apply(params, features)

        # Both fields should be accessible
        _ = output.selected
        _ = output.weights


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_grn_no_nan(self, rng_key):
        """GRN should not produce NaN."""
        model = GRN(hidden_size=64)
        x = jax.random.normal(rng_key, (2, 16, 64))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_vsn_no_nan(self, rng_key):
        """VSN should not produce NaN."""
        model = VariableSelectionNetwork(hidden_size=64)
        features = jax.random.normal(rng_key, (2, 16, 5, 32))

        params = model.init(rng_key, features)
        output = model.apply(params, features)

        assert not jnp.any(jnp.isnan(output.selected))
        assert not jnp.any(jnp.isnan(output.weights))
