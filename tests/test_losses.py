"""Tests for loss functions."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.distributions.gaussian import GaussianHead
from jax_prism.losses.nll import NLLLoss
from jax_prism.losses.quantile import QuantileLoss


class TestNLLLossBasic:
    """Basic tests for NLLLoss."""

    def test_output_is_scalar(self, rng_key):
        """Loss should return a scalar."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_loss_is_positive(self, rng_key):
        """NLL loss can be any real value, but typically positive for reasonable predictions."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        # For random predictions and targets, loss should be finite
        assert jnp.isfinite(loss)

    def test_no_nan(self, rng_key):
        """Should not produce NaN."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        assert not jnp.isnan(loss)


class TestNLLLossBehavior:
    """Tests for expected NLL loss behavior."""

    def test_lower_loss_when_target_at_mean(self, rng_key):
        """Loss should be lower when target equals predicted mean."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        # Predictions: loc=0, scale=1 (after softplus)
        predictions = jnp.zeros((10, 2))

        # Targets at the mean (0) vs away from mean
        targets_at_mean = jnp.zeros(10)
        targets_away = jnp.ones(10) * 3

        loss_at_mean = loss_fn(predictions, targets_at_mean)
        loss_away = loss_fn(predictions, targets_away)

        assert loss_at_mean < loss_away

    def test_lower_loss_with_smaller_scale(self):
        """Lower scale should give lower loss when target is at mean."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        # Target at loc=0
        targets = jnp.zeros(10)

        # Small scale (more confident, lower loss at mean)
        predictions_small_scale = jnp.stack([jnp.zeros(10), -5 * jnp.ones(10)], axis=-1)
        # Large scale (less confident)
        predictions_large_scale = jnp.stack([jnp.zeros(10), 5 * jnp.ones(10)], axis=-1)

        loss_small = loss_fn(predictions_small_scale, targets)
        loss_large = loss_fn(predictions_large_scale, targets)

        assert loss_small < loss_large

    def test_loss_decreases_as_prediction_improves(self, rng_key):
        """Loss should decrease as loc approaches target."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        targets = jnp.ones(10) * 5

        # Prediction far from target
        pred_far = jnp.stack([jnp.zeros(10), jnp.zeros(10)], axis=-1)
        # Prediction close to target
        pred_close = jnp.stack([jnp.ones(10) * 5, jnp.zeros(10)], axis=-1)

        loss_far = loss_fn(pred_far, targets)
        loss_close = loss_fn(pred_close, targets)

        assert loss_close < loss_far


class TestNLLLossMasking:
    """Tests for NLL loss masking."""

    def test_mask_zeros_out_contribution(self, rng_key):
        """Masked positions should not contribute to loss."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 2))
        targets = jax.random.normal(rng_key, (4,))

        # Full mask (all valid)
        mask_full = jnp.ones(4)
        loss_full = loss_fn(predictions, targets, mask=mask_full)

        # Half mask
        mask_half = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss_half = loss_fn(predictions, targets, mask=mask_half)

        # Losses should differ
        assert not jnp.allclose(loss_full, loss_half)

    def test_empty_mask_returns_zero(self, rng_key):
        """All-zero mask should return 0 (or handle gracefully)."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 2))
        targets = jax.random.normal(rng_key, (4,))
        mask = jnp.zeros(4)

        loss = loss_fn(predictions, targets, mask=mask)

        # Should be 0 (sum of nothing / max(0, 1) = 0)
        chex.assert_trees_all_close(loss, 0.0, atol=1e-7)

    def test_mask_normalization(self, rng_key):
        """Loss should be normalized by sum of mask, not total count."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        # Create predictions where all have same NLL
        predictions = jnp.zeros((4, 2))  # loc=0, scale=softplus(0)+eps
        targets = jnp.zeros(4)  # targets at mean

        # Full mask
        mask_full = jnp.ones(4)
        loss_full = loss_fn(predictions, targets, mask=mask_full)

        # Partial mask (2 valid)
        mask_partial = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss_partial = loss_fn(predictions, targets, mask=mask_partial)

        # Since all NLLs are identical, mean should be same regardless of mask size
        chex.assert_trees_all_close(loss_full, loss_partial, atol=1e-5)


class TestNLLLossJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """NLLLoss should be JIT-compilable."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        targets = jax.random.normal(rng_key, (4, 8))

        @jax.jit
        def compute_loss(preds, targs):
            return loss_fn(preds, targs)

        loss = compute_loss(predictions, targets)

        assert jnp.isfinite(loss)

    def test_jit_with_mask(self, rng_key):
        """JIT should work with masking."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        targets = jax.random.normal(rng_key, (4, 8))
        mask = jax.random.bernoulli(rng_key, 0.8, (4, 8)).astype(jnp.float32)

        @jax.jit
        def compute_loss(preds, targs, m):
            return loss_fn(preds, targs, mask=m)

        loss = compute_loss(predictions, targets, mask)

        assert jnp.isfinite(loss)

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through the loss."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        targets = jax.random.normal(rng_key, (4, 8))

        def loss_wrapper(predictions):
            return loss_fn(predictions, targets)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        grads = jax.grad(loss_wrapper)(predictions)

        assert not jnp.any(jnp.isnan(grads))
        assert not jnp.allclose(grads, 0)

    def test_gradient_with_mask(self, rng_key):
        """Gradients should flow through masked loss."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        targets = jax.random.normal(rng_key, (4, 8))
        mask = jnp.ones((4, 8))
        mask = mask.at[:, 4:].set(0)  # Mask out second half

        def loss_wrapper(predictions):
            return loss_fn(predictions, targets, mask=mask)

        predictions = jax.random.normal(rng_key, (4, 8, 2))
        grads = jax.grad(loss_wrapper)(predictions)

        # Gradients for masked positions should be zero
        chex.assert_trees_all_close(grads[:, 4:], jnp.zeros((4, 4, 2)), atol=1e-6)

        # Gradients for unmasked positions should be non-zero
        assert not jnp.allclose(grads[:, :4], 0)

    def test_vmap_over_batch(self, rng_key):
        """Should work with vmap."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        # Batch of prediction-target pairs
        predictions = jax.random.normal(rng_key, (8, 16, 2))
        targets = jax.random.normal(rng_key, (8, 16))

        # vmap over first dimension
        vmapped_loss = jax.vmap(loss_fn)
        losses = vmapped_loss(predictions, targets)

        chex.assert_shape(losses, (8,))


class TestNLLLossWithDifferentShapes:
    """Tests for various input shapes."""

    def test_1d_input(self, rng_key):
        """Works with 1D inputs."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (10, 2))
        targets = jax.random.normal(rng_key, (10,))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_2d_input(self, rng_key):
        """Works with 2D inputs (batch, time)."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (4, 16, 2))
        targets = jax.random.normal(rng_key, (4, 16))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_3d_input(self, rng_key):
        """Works with 3D inputs (batch, time, features)."""
        head = GaussianHead()
        loss_fn = NLLLoss(distribution=head)

        predictions = jax.random.normal(rng_key, (2, 8, 4, 2))
        targets = jax.random.normal(rng_key, (2, 8, 4))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())


# ============================================================================
# QuantileLoss Tests
# ============================================================================


class TestQuantileLossBasic:
    """Basic tests for QuantileLoss."""

    def test_output_is_scalar(self, rng_key):
        """Loss should return a scalar."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_loss_is_non_negative(self, rng_key):
        """Pinball loss is always non-negative."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        assert loss >= 0

    def test_no_nan(self, rng_key):
        """Should not produce NaN."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        assert jnp.isfinite(loss)


class TestQuantileLossBehavior:
    """Tests for expected quantile loss behavior."""

    def test_zero_loss_when_perfect(self):
        """Loss should be zero when predictions equal targets at all quantiles."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        targets = jnp.array([1.0, 2.0, 3.0])
        # Predictions: each target repeated for all quantiles
        predictions = jnp.stack([targets, targets, targets], axis=-1)

        loss = loss_fn(predictions, targets)

        chex.assert_trees_all_close(loss, 0.0, atol=1e-7)

    def test_asymmetric_penalty_lower_quantile(self):
        """Lower quantile (q<0.5) should penalize underprediction more."""
        quantiles = jnp.array([0.1])  # Low quantile
        loss_fn = QuantileLoss(quantiles)

        target = jnp.array([0.0])

        # Underprediction: pred < target (error < 0)
        pred_under = jnp.array([[-1.0]])  # error = -1 - 0 = -1
        # Overprediction: pred > target (error > 0)
        pred_over = jnp.array([[1.0]])  # error = 1 - 0 = 1

        loss_under = loss_fn(pred_under, target)
        loss_over = loss_fn(pred_over, target)

        # For q=0.1: underprediction penalized by (1-q)=0.9, overprediction by q=0.1
        # So overprediction should have lower loss
        assert loss_over < loss_under

    def test_asymmetric_penalty_upper_quantile(self):
        """Upper quantile (q>0.5) should penalize overprediction more."""
        quantiles = jnp.array([0.9])  # High quantile
        loss_fn = QuantileLoss(quantiles)

        target = jnp.array([0.0])

        pred_under = jnp.array([[-1.0]])
        pred_over = jnp.array([[1.0]])

        loss_under = loss_fn(pred_under, target)
        loss_over = loss_fn(pred_over, target)

        # For q=0.9: overprediction penalized by q=0.9, underprediction by (1-q)=0.1
        # So underprediction should have lower loss
        assert loss_under < loss_over

    def test_median_symmetric(self):
        """At q=0.5, penalty should be symmetric."""
        quantiles = jnp.array([0.5])
        loss_fn = QuantileLoss(quantiles)

        target = jnp.array([0.0])

        pred_under = jnp.array([[-1.0]])
        pred_over = jnp.array([[1.0]])

        loss_under = loss_fn(pred_under, target)
        loss_over = loss_fn(pred_over, target)

        # For q=0.5: both directions penalized by 0.5
        chex.assert_trees_all_close(loss_under, loss_over, atol=1e-7)

    def test_loss_increases_with_error(self):
        """Loss should increase as prediction error increases."""
        quantiles = jnp.array([0.5])
        loss_fn = QuantileLoss(quantiles)

        target = jnp.array([0.0])

        pred_close = jnp.array([[0.1]])
        pred_far = jnp.array([[1.0]])

        loss_close = loss_fn(pred_close, target)
        loss_far = loss_fn(pred_far, target)

        assert loss_close < loss_far


class TestQuantileLossMasking:
    """Tests for quantile loss masking."""

    def test_mask_zeros_out_contribution(self, rng_key):
        """Masked positions should not contribute to loss."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 3))
        targets = jax.random.normal(rng_key, (4,))

        mask_full = jnp.ones(4)
        loss_full = loss_fn(predictions, targets, mask=mask_full)

        mask_half = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss_half = loss_fn(predictions, targets, mask=mask_half)

        assert not jnp.allclose(loss_full, loss_half)

    def test_empty_mask_returns_zero(self, rng_key):
        """All-zero mask should return 0."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 3))
        targets = jax.random.normal(rng_key, (4,))
        mask = jnp.zeros(4)

        loss = loss_fn(predictions, targets, mask=mask)

        chex.assert_trees_all_close(loss, 0.0, atol=1e-7)

    def test_mask_normalization(self, rng_key):
        """Loss should be normalized by sum of mask."""
        quantiles = jnp.array([0.5])
        loss_fn = QuantileLoss(quantiles)

        # Create identical predictions for all points
        predictions = jnp.ones((4, 1))
        targets = jnp.zeros(4)  # Error of 1 everywhere

        mask_full = jnp.ones(4)
        loss_full = loss_fn(predictions, targets, mask=mask_full)

        mask_partial = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss_partial = loss_fn(predictions, targets, mask=mask_partial)

        # Mean should be same since all losses are identical
        chex.assert_trees_all_close(loss_full, loss_partial, atol=1e-5)


class TestQuantileLossJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """QuantileLoss should be JIT-compilable."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        targets = jax.random.normal(rng_key, (4, 8))

        @jax.jit
        def compute_loss(preds, targs):
            return loss_fn(preds, targs)

        loss = compute_loss(predictions, targets)

        assert jnp.isfinite(loss)

    def test_jit_with_mask(self, rng_key):
        """JIT should work with masking."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        targets = jax.random.normal(rng_key, (4, 8))
        mask = jax.random.bernoulli(rng_key, 0.8, (4, 8)).astype(jnp.float32)

        @jax.jit
        def compute_loss(preds, targs, m):
            return loss_fn(preds, targs, mask=m)

        loss = compute_loss(predictions, targets, mask)

        assert jnp.isfinite(loss)

    def test_gradient_flow(self, rng_key):
        """Gradients should flow through the loss."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        targets = jax.random.normal(rng_key, (4, 8))

        def loss_wrapper(predictions):
            return loss_fn(predictions, targets)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        grads = jax.grad(loss_wrapper)(predictions)

        assert not jnp.any(jnp.isnan(grads))
        assert not jnp.allclose(grads, 0)

    def test_gradient_with_mask(self, rng_key):
        """Gradients should respect mask."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        targets = jax.random.normal(rng_key, (4, 8))
        mask = jnp.ones((4, 8))
        mask = mask.at[:, 4:].set(0)  # Mask out second half

        def loss_wrapper(predictions):
            return loss_fn(predictions, targets, mask=mask)

        predictions = jax.random.normal(rng_key, (4, 8, 3))
        grads = jax.grad(loss_wrapper)(predictions)

        # Gradients for masked positions should be zero
        chex.assert_trees_all_close(grads[:, 4:], jnp.zeros((4, 4, 3)), atol=1e-6)

        # Gradients for unmasked positions should be non-zero
        assert not jnp.allclose(grads[:, :4], 0)

    def test_vmap_over_batch(self, rng_key):
        """Should work with vmap."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (8, 16, 3))
        targets = jax.random.normal(rng_key, (8, 16))

        vmapped_loss = jax.vmap(loss_fn)
        losses = vmapped_loss(predictions, targets)

        chex.assert_shape(losses, (8,))


class TestQuantileLossWithDifferentShapes:
    """Tests for various input shapes."""

    def test_1d_input(self, rng_key):
        """Works with 1D inputs."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (10, 3))
        targets = jax.random.normal(rng_key, (10,))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_2d_input(self, rng_key):
        """Works with 2D inputs (batch, time)."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 16, 3))
        targets = jax.random.normal(rng_key, (4, 16))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_single_quantile(self, rng_key):
        """Works with single quantile."""
        quantiles = jnp.array([0.5])
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 1))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())

    def test_many_quantiles(self, rng_key):
        """Works with many quantiles."""
        quantiles = jnp.linspace(0.05, 0.95, 19)  # 19 quantiles
        loss_fn = QuantileLoss(quantiles)

        predictions = jax.random.normal(rng_key, (4, 8, 19))
        targets = jax.random.normal(rng_key, (4, 8))

        loss = loss_fn(predictions, targets)

        chex.assert_shape(loss, ())
