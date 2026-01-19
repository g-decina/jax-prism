"""Tests for positional encoding components (RoPE)."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.models.components.positional import apply_rope, build_rope_frequencies


class TestBuildRopeFrequencies:
    """Tests for build_rope_frequencies function."""

    def test_output_shapes(self):
        """Returns cos and sin with correct shapes."""
        dim, max_seq_len = 64, 128
        cos, sin = build_rope_frequencies(dim, max_seq_len)

        chex.assert_shape(cos, (max_seq_len, dim // 2))
        chex.assert_shape(sin, (max_seq_len, dim // 2))

    def test_returns_tuple(self):
        """Returns a tuple of two arrays."""
        result = build_rope_frequencies(32, 64)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cos_sin_bounds(self):
        """Cos and sin values should be in [-1, 1]."""
        cos, sin = build_rope_frequencies(64, 128)

        assert jnp.all(cos >= -1.0)
        assert jnp.all(cos <= 1.0)
        assert jnp.all(sin >= -1.0)
        assert jnp.all(sin <= 1.0)

    def test_pythagorean_identity(self):
        """cos^2 + sin^2 should equal 1."""
        cos, sin = build_rope_frequencies(64, 128)

        identity = cos ** 2 + sin ** 2
        chex.assert_trees_all_close(identity, jnp.ones_like(identity), atol=1e-6)

    def test_position_zero_is_identity(self):
        """At position 0, cos=1 and sin=0 (no rotation)."""
        cos, sin = build_rope_frequencies(64, 128)

        # Position 0 (first row)
        chex.assert_trees_all_close(cos[0], jnp.ones(32), atol=1e-6)
        chex.assert_trees_all_close(sin[0], jnp.zeros(32), atol=1e-6)

    def test_frequencies_decrease_with_dimension(self):
        """Higher dimension indices should have lower frequencies."""
        cos, sin = build_rope_frequencies(64, 128)

        # At a fixed position (e.g., position 10), the angle = pos * theta_i
        # theta_i decreases with i, so later dimensions rotate slower
        # This means cos values for later dims should be closer to 1 at low positions
        pos = 10
        # For dim 0 (high freq) vs dim 31 (low freq)
        # The low freq dimension should have cos closer to 1
        assert jnp.abs(cos[pos, -1] - 1.0) < jnp.abs(cos[pos, 0] - 1.0)

    def test_custom_base(self):
        """Custom base affects frequency computation."""
        cos1, sin1 = build_rope_frequencies(64, 128, base=10_000.0)
        cos2, sin2 = build_rope_frequencies(64, 128, base=1_000.0)

        # Different bases should produce different frequencies
        assert not jnp.allclose(cos1, cos2)
        assert not jnp.allclose(sin1, sin2)

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        cos1, sin1 = build_rope_frequencies(64, 128)
        cos2, sin2 = build_rope_frequencies(64, 128)

        chex.assert_trees_all_close(cos1, cos2, atol=0)
        chex.assert_trees_all_close(sin1, sin2, atol=0)


class TestApplyRope:
    """Tests for apply_rope function."""

    def test_output_shape_matches_input(self, rng_key):
        """Output shape should equal input shape."""
        seq_len, dim = 32, 64
        x = jax.random.normal(rng_key, (4, seq_len, dim))
        cos, sin = build_rope_frequencies(dim, seq_len)

        y = apply_rope(x, cos, sin)

        chex.assert_shape(y, x.shape)

    def test_output_shape_4d(self, rng_key):
        """Works with 4D input (batch, heads, seq, dim)."""
        batch, heads, seq_len, head_dim = 2, 8, 32, 64
        x = jax.random.normal(rng_key, (batch, heads, seq_len, head_dim))
        cos, sin = build_rope_frequencies(head_dim, seq_len)

        y = apply_rope(x, cos, sin)

        chex.assert_shape(y, (batch, heads, seq_len, head_dim))

    def test_position_zero_identity(self, rng_key):
        """At position 0, rotation should be identity (cos=1, sin=0)."""
        seq_len, dim = 32, 64
        x = jax.random.normal(rng_key, (1, seq_len, dim))
        cos, sin = build_rope_frequencies(dim, seq_len)

        y = apply_rope(x, cos, sin)

        # At position 0, output should equal input
        chex.assert_trees_all_close(y[0, 0, :], x[0, 0, :], atol=1e-5)

    def test_rotation_preserves_norm(self, rng_key):
        """Rotation should preserve vector norms."""
        seq_len, dim = 32, 64
        x = jax.random.normal(rng_key, (4, seq_len, dim))
        cos, sin = build_rope_frequencies(dim, seq_len)

        y = apply_rope(x, cos, sin)

        # Norm of each vector should be preserved
        x_norms = jnp.linalg.norm(x, axis=-1)
        y_norms = jnp.linalg.norm(y, axis=-1)

        chex.assert_trees_all_close(x_norms, y_norms, atol=1e-5)

    def test_relative_position_property(self, rng_key):
        """Dot product between rotated vectors depends on relative position.

        For positions m and n: <R(m)q, R(n)k> = <R(m-n)q, k>
        This is the key property that makes RoPE work.
        """
        dim = 64
        max_seq_len = 128
        cos, sin = build_rope_frequencies(dim, max_seq_len)

        # Create two vectors
        q = jax.random.normal(rng_key, (dim,))
        k = jax.random.normal(jax.random.fold_in(rng_key, 1), (dim,))

        # Positions to test
        m, n = 10, 15
        relative = m - n  # = -5, but we use absolute value for lookup

        # Rotate q at position m, k at position n
        q_m = apply_rope(q[None, None, :], cos[m:m+1], sin[m:m+1])[0, 0]
        k_n = apply_rope(k[None, None, :], cos[n:n+1], sin[n:n+1])[0, 0]

        # Dot product of rotated vectors
        dot_rotated = jnp.dot(q_m, k_n)

        # Now rotate q at position (m-n) relative to origin, keep k at 0
        # For negative relative position, we need to think about this differently
        # The property is: the dot product only depends on (m-n)
        # Let's verify by shifting both positions by the same amount
        offset = 20
        q_m2 = apply_rope(q[None, None, :], cos[m+offset:m+offset+1], sin[m+offset:m+offset+1])[0, 0]
        k_n2 = apply_rope(k[None, None, :], cos[n+offset:n+offset+1], sin[n+offset:n+offset+1])[0, 0]
        dot_shifted = jnp.dot(q_m2, k_n2)

        # Both should be equal (same relative position)
        chex.assert_trees_all_close(dot_rotated, dot_shifted, atol=1e-5)

    def test_jit_compatible(self, rng_key):
        """apply_rope should be JIT-compilable."""
        seq_len, dim = 32, 64
        x = jax.random.normal(rng_key, (4, seq_len, dim))
        cos, sin = build_rope_frequencies(dim, seq_len)

        @jax.jit
        def apply(x, cos, sin):
            return apply_rope(x, cos, sin)

        y = apply(x, cos, sin)
        chex.assert_shape(y, x.shape)

    def test_vmap_compatible(self, rng_key):
        """apply_rope should work with vmap."""
        seq_len, dim = 32, 64
        cos, sin = build_rope_frequencies(dim, seq_len)

        # Single sequence
        x_single = jax.random.normal(rng_key, (seq_len, dim))

        # Batched via vmap
        x_batch = jax.random.normal(rng_key, (4, seq_len, dim))

        vmapped_apply = jax.vmap(lambda x: apply_rope(x, cos, sin))
        y_vmap = vmapped_apply(x_batch)

        # Direct batched application
        y_direct = apply_rope(x_batch, cos, sin)

        chex.assert_trees_all_close(y_vmap, y_direct, atol=1e-6)

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through apply_rope."""
        seq_len, dim = 32, 64
        x = jax.random.normal(rng_key, (4, seq_len, dim))
        cos, sin = build_rope_frequencies(dim, seq_len)

        def loss_fn(x):
            y = apply_rope(x, cos, sin)
            return jnp.mean(y ** 2)

        grads = jax.grad(loss_fn)(x)

        # Gradients should exist and not be zero
        assert not jnp.allclose(grads, 0)
        chex.assert_shape(grads, x.shape)

    def test_different_sequence_lengths(self, rng_key):
        """Should work with different sequence lengths using sliced frequencies."""
        dim = 64
        max_seq_len = 128
        cos, sin = build_rope_frequencies(dim, max_seq_len)

        # Use only first 32 positions
        actual_seq_len = 32
        x = jax.random.normal(rng_key, (4, actual_seq_len, dim))

        y = apply_rope(x, cos[:actual_seq_len], sin[:actual_seq_len])

        chex.assert_shape(y, (4, actual_seq_len, dim))

    def test_numerical_stability(self, rng_key):
        """Should handle various input magnitudes without NaN/Inf."""
        seq_len, dim = 32, 64
        cos, sin = build_rope_frequencies(dim, seq_len)

        # Large values
        x_large = jax.random.normal(rng_key, (4, seq_len, dim)) * 1e6
        y_large = apply_rope(x_large, cos, sin)
        assert not jnp.any(jnp.isnan(y_large))
        assert not jnp.any(jnp.isinf(y_large))

        # Small values
        x_small = jax.random.normal(rng_key, (4, seq_len, dim)) * 1e-6
        y_small = apply_rope(x_small, cos, sin)
        assert not jnp.any(jnp.isnan(y_small))
        assert not jnp.any(jnp.isinf(y_small))

    def test_deterministic(self, rng_key):
        """Same inputs produce same outputs."""
        seq_len, dim = 32, 64
        x = jax.random.normal(rng_key, (4, seq_len, dim))
        cos, sin = build_rope_frequencies(dim, seq_len)

        y1 = apply_rope(x, cos, sin)
        y2 = apply_rope(x, cos, sin)

        chex.assert_trees_all_close(y1, y2, atol=0)
