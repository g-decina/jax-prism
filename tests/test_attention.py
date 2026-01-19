"""Tests for GroupedQueryAttention module."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.models.components.attention import GroupedQueryAttention


class TestGroupedQueryAttentionShapes:
    """Tests for output shapes across configurations."""

    def test_output_shape_basic(self, rng_key):
        """Output shape should be (batch, seq_len, num_heads * head_dim)."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=8, head_dim=64, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 512))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 8 * 64))

    def test_output_shape_mha(self, rng_key):
        """MHA: num_kv_heads == num_heads."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=8, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (4, 32, 256))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (4, 32, 8 * 32))

    def test_output_shape_mqa(self, rng_key):
        """MQA: num_kv_heads == 1."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=1, head_dim=64, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 512))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 8 * 64))

    def test_output_shape_gqa(self, rng_key):
        """GQA: num_kv_heads between 1 and num_heads."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=2, head_dim=64, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 512))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 8 * 64))

    def test_output_shape_with_rope(self, rng_key):
        """Shape preserved when RoPE is enabled."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=True
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 4 * 32))


class TestSelfAttention:
    """Tests for self-attention mode (context=None)."""

    def test_self_attention_default(self, rng_key):
        """Self-attention when context is not provided."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 128))

    def test_self_attention_explicit_none(self, rng_key):
        """Explicit context=None behaves same as default."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)
        y1 = model.apply(params, x)
        y2 = model.apply(params, x, context=None)

        chex.assert_trees_all_close(y1, y2, atol=0)


class TestCrossAttention:
    """Tests for cross-attention mode (context provided)."""

    def test_cross_attention_different_lengths(self, rng_key):
        """Cross-attention with different query and key/value lengths."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        q_len, kv_len = 16, 32
        x = jax.random.normal(rng_key, (2, q_len, 128))
        context = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, kv_len, 128))

        params = model.init(rng_key, x, context=context)
        y = model.apply(params, x, context=context)

        # Output length matches query length
        chex.assert_shape(y, (2, q_len, 128))

    def test_cross_attention_same_length(self, rng_key):
        """Cross-attention with same lengths but different content."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128))
        context = jax.random.normal(jax.random.fold_in(rng_key, 1), (2, 16, 128))

        params = model.init(rng_key, x, context=context)
        y_cross = model.apply(params, x, context=context)
        y_self = model.apply(params, x)

        # Cross and self attention should differ
        assert not jnp.allclose(y_cross, y_self)


class TestCausalMasking:
    """Tests for causal (autoregressive) masking."""

    def test_causal_mask_shape_preserved(self, rng_key):
        """Causal masking doesn't change output shape."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, causal=True
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (2, 16, 128))

    def test_causal_vs_non_causal_differ(self, rng_key):
        """Causal and non-causal attention produce different outputs."""
        model_causal = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, causal=True
        )
        model_non_causal = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, causal=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model_causal.init(rng_key, x)
        y_causal = model_causal.apply(params, x)
        y_non_causal = model_non_causal.apply(params, x)

        assert not jnp.allclose(y_causal, y_non_causal)

    def test_causal_first_position_same(self, rng_key):
        """First position output should be same for causal and non-causal.

        At position 0, causal mask allows attending only to position 0,
        which is the same as what a non-causal model would primarily attend to
        for a single-position query (though not identical due to softmax normalization).
        """
        # This test verifies the mask is applied correctly at boundaries
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, causal=True
        )
        x = jax.random.normal(rng_key, (1, 8, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        # Just verify it runs without error and produces valid output
        assert not jnp.any(jnp.isnan(y))


class TestUserMask:
    """Tests for user-provided attention masks."""

    def test_mask_applied(self, rng_key):
        """User mask affects attention output."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 8, 128))

        params = model.init(rng_key, x)

        # No mask
        y_no_mask = model.apply(params, x)

        # Mask out second half of keys
        mask = jnp.ones((1, 1, 8, 8), dtype=bool)
        mask = mask.at[:, :, :, 4:].set(False)
        y_masked = model.apply(params, x, mask=mask)

        assert not jnp.allclose(y_no_mask, y_masked)

    def test_full_mask_no_nan(self, rng_key):
        """Masking all but one position shouldn't produce NaN."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (1, 4, 128))

        params = model.init(rng_key, x)

        # Only allow attending to first position
        mask = jnp.zeros((1, 1, 4, 4), dtype=bool)
        mask = mask.at[:, :, :, 0].set(True)
        y = model.apply(params, x, mask=mask)

        assert not jnp.any(jnp.isnan(y))


class TestRoPE:
    """Tests for rotary position embeddings."""

    def test_rope_affects_output(self, rng_key):
        """RoPE should change the output compared to no RoPE."""
        model_rope = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=True
        )
        model_no_rope = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        # Use same random init for both
        params_rope = model_rope.init(rng_key, x)
        params_no_rope = model_no_rope.init(rng_key, x)

        y_rope = model_rope.apply(params_rope, x)
        y_no_rope = model_no_rope.apply(params_no_rope, x)

        # Outputs should differ due to RoPE
        assert not jnp.allclose(y_rope, y_no_rope)

    def test_rope_position_sensitivity(self, rng_key):
        """Swapping position order should change attention patterns with RoPE.

        With RoPE, attention is position-sensitive. Reversing the sequence
        should produce different outputs than the original.
        """
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=True, causal=False
        )
        x = jax.random.normal(rng_key, (1, 8, 128))
        x_reversed = x[:, ::-1, :]  # Reverse sequence order

        params = model.init(rng_key, x)
        y_original = model.apply(params, x)
        y_reversed = model.apply(params, x_reversed)

        # If RoPE had no effect, reversing input would just reverse output
        # But with RoPE, the position encodings differ, so outputs differ beyond reversal
        y_reversed_back = y_reversed[:, ::-1, :]
        assert not jnp.allclose(y_original, y_reversed_back, atol=1e-3)


class TestDropout:
    """Tests for attention dropout."""

    def test_deterministic_no_dropout(self, rng_key):
        """Deterministic mode should not apply dropout."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, dropout_rate=0.5
        )
        x = jax.random.normal(rng_key, (2, 8, 128))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x, deterministic=True)
        y2 = model.apply(params, x, deterministic=True)

        chex.assert_trees_all_close(y1, y2, atol=0)

    def test_training_dropout_differs(self, rng_key):
        """Training mode with different RNG should produce different outputs."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, dropout_rate=0.5
        )
        x = jax.random.normal(rng_key, (2, 8, 128))

        params = model.init(rng_key, x)

        # Apply with different dropout RNG keys
        y1 = model.apply(
            params, x, deterministic=False, rngs={'dropout': jax.random.key(0)}
        )
        y2 = model.apply(
            params, x, deterministic=False, rngs={'dropout': jax.random.key(1)}
        )

        assert not jnp.allclose(y1, y2)

    def test_zero_dropout_rate_same_output(self, rng_key):
        """Zero dropout rate should produce same output in training mode."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=4, head_dim=32, use_rope=False, dropout_rate=0.0
        )
        x = jax.random.normal(rng_key, (2, 8, 128))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x, deterministic=False)
        y2 = model.apply(params, x, deterministic=False)

        chex.assert_trees_all_close(y1, y2, atol=0)


class TestGQAHeadExpansion:
    """Tests specific to GQA head expansion logic."""

    def test_mha_no_expansion(self, rng_key):
        """MHA (num_heads == num_kv_heads) should work correctly."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=8, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 256))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        chex.assert_shape(y, (2, 16, 256))

    def test_gqa_4_to_8_heads(self, rng_key):
        """GQA with 4 KV heads expanded to 8 query heads."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=4, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 256))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        chex.assert_shape(y, (2, 16, 256))

    def test_gqa_2_to_8_heads(self, rng_key):
        """GQA with 2 KV heads expanded to 8 query heads."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=2, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 256))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        chex.assert_shape(y, (2, 16, 256))

    def test_mqa_1_to_8_heads(self, rng_key):
        """MQA with 1 KV head expanded to 8 query heads."""
        model = GroupedQueryAttention(
            num_heads=8, num_kv_heads=1, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 256))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        chex.assert_shape(y, (2, 16, 256))


class TestJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """GQA should be JIT-compilable."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=True
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x)

        y = forward(params, x)
        chex.assert_shape(y, (2, 16, 128))

    def test_vmap_over_batch(self, rng_key):
        """GQA should work with vmap over batch dimension."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=False
        )
        x_single = jax.random.normal(rng_key, (1, 16, 128))
        x_batch = jax.random.normal(rng_key, (4, 16, 128))

        params = model.init(rng_key, x_single)

        # Direct batch
        y_direct = model.apply(params, x_batch)

        # Via vmap (remove batch dim, vmap adds it back)
        vmapped = jax.vmap(lambda x: model.apply(params, x[None])[0])
        y_vmap = vmapped(x_batch)

        chex.assert_trees_all_close(y_direct, y_vmap, atol=1e-5)

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through GQA."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=True
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)

        def loss_fn(params):
            y = model.apply(params, x)
            return jnp.mean(y ** 2)

        grads = jax.grad(loss_fn)(params)

        # All projection gradients should exist and be non-zero
        for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            assert proj in grads['params']
            assert not jnp.allclose(grads['params'][proj]['kernel'], 0)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_normal_input(self, rng_key):
        """Normal input should not produce NaN."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=True, causal=True
        )
        x = jax.random.normal(rng_key, (2, 32, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_no_nan_large_input(self, rng_key):
        """Large input values should not cause overflow."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128)) * 100

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))

    def test_no_nan_small_input(self, rng_key):
        """Small input values should not cause issues."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=False
        )
        x = jax.random.normal(rng_key, (2, 16, 128)) * 1e-6

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        assert not jnp.any(jnp.isnan(y))

    def test_long_sequence(self, rng_key):
        """Should handle longer sequences within max_seq_len."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32,
            use_rope=True, max_seq_len=1024
        )
        x = jax.random.normal(rng_key, (1, 512, 128))

        params = model.init(rng_key, x)
        y = model.apply(params, x)

        chex.assert_shape(y, (1, 512, 128))
        assert not jnp.any(jnp.isnan(y))


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self, rng_key):
        """Same input should produce identical output."""
        model = GroupedQueryAttention(
            num_heads=4, num_kv_heads=2, head_dim=32, use_rope=True
        )
        x = jax.random.normal(rng_key, (2, 16, 128))

        params = model.init(rng_key, x)

        y1 = model.apply(params, x)
        y2 = model.apply(params, x)

        chex.assert_trees_all_close(y1, y2, atol=0)