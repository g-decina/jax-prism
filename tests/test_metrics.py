"""Tests for metrics/point.py and metrics/probabilistic.py."""

import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

from jax_prism.metrics.point import mae, mase, smape
from jax_prism.metrics.probabilistic import coverage, quantile_loss


class TestMAE:
    """Tests for Mean Absolute Error."""

    def test_perfect_prediction(self):
        """MAE is zero when predictions equal targets."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])
        y_pred = jnp.array([[[1.0], [2.0], [3.0]]])
        assert_trees_all_close(mae(y_true, y_pred), 0.0)

    def test_constant_error(self):
        """MAE equals constant when all errors are the same."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])
        y_pred = jnp.array([[[2.0], [3.0], [4.0]]])  # all errors = 1
        assert_trees_all_close(mae(y_true, y_pred), 1.0)

    def test_symmetric(self):
        """MAE is symmetric (over/under prediction treated equally)."""
        y_true = jnp.array([[[0.0]]])
        y_pred_over = jnp.array([[[1.0]]])
        y_pred_under = jnp.array([[[-1.0]]])
        assert_trees_all_close(
            mae(y_true, y_pred_over),
            mae(y_true, y_pred_under),
        )

    def test_with_mask(self):
        """Mask excludes elements from calculation."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])
        y_pred = jnp.array([[[1.0], [2.0], [100.0]]])  # last element way off
        mask = jnp.array([[[1.0], [1.0], [0.0]]])  # ignore last
        assert_trees_all_close(mae(y_true, y_pred, mask), 0.0)

    def test_batch_aggregation(self):
        """MAE aggregates across batch dimension."""
        y_true = jnp.array([
            [[0.0]],  # batch 0
            [[0.0]],  # batch 1
        ])
        y_pred = jnp.array([
            [[1.0]],  # error = 1
            [[3.0]],  # error = 3
        ])
        # Mean of [1, 3] = 2
        assert_trees_all_close(mae(y_true, y_pred), 2.0)


class TestSMAPE:
    """Tests for Symmetric Mean Absolute Percentage Error."""

    def test_perfect_prediction(self):
        """SMAPE is zero when predictions equal targets."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])
        y_pred = jnp.array([[[1.0], [2.0], [3.0]]])
        assert_trees_all_close(smape(y_true, y_pred), 0.0)

    def test_symmetric_same_average(self):
        """SMAPE is symmetric when average stays constant."""
        # For true symmetry, we need |y_true| + |y_pred| to be the same
        # y_true=100, y_pred=100±δ doesn't give this
        # Instead, test that swapping y_true and y_pred gives same result
        y_a = jnp.array([[[100.0]]])
        y_b = jnp.array([[[120.0]]])
        # SMAPE(a, b) should equal SMAPE(b, a)
        assert_trees_all_close(
            smape(y_a, y_b),
            smape(y_b, y_a),
        )

    def test_both_zero(self):
        """When both y_true and y_pred are zero, contributes 0."""
        y_true = jnp.array([[[0.0], [1.0]]])
        y_pred = jnp.array([[[0.0], [1.0]]])
        # First element: 0/0 -> 0, second element: 0/1 -> 0
        assert_trees_all_close(smape(y_true, y_pred), 0.0)

    def test_range(self):
        """SMAPE is in range [0, 200]."""
        y_true = jnp.array([[[1.0]]])
        y_pred = jnp.array([[[1000.0]]])  # very wrong
        result = smape(y_true, y_pred)
        assert result >= 0.0
        assert result <= 200.0

    def test_with_mask(self):
        """Mask excludes elements from calculation."""
        y_true = jnp.array([[[1.0], [1.0]]])
        y_pred = jnp.array([[[1.0], [1000.0]]])  # second way off
        mask = jnp.array([[[1.0], [0.0]]])  # ignore second
        assert_trees_all_close(smape(y_true, y_pred, mask), 0.0)


class TestMASE:
    """Tests for Mean Absolute Scaled Error."""

    def test_beats_naive(self):
        """MASE < 1 means prediction beats naive forecast."""
        # Training data: constant series (naive forecast is perfect)
        # Actually, naive forecast error would be 0, causing division issues
        # Use trending series instead
        y_train = jnp.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])  # trend
        # Naive forecast (lag 1): predicts [1,2,3,4] for [2,3,4,5]
        # Naive errors: [1,1,1,1], MAE_naive = 1

        y_true = jnp.array([[[6.0], [7.0]]])  # continuing trend
        y_pred = jnp.array([[[6.0], [7.0]]])  # perfect prediction

        result = mase(y_true, y_pred, y_train, seasonality=1)
        assert result < 1.0  # beats naive

    def test_worse_than_naive(self):
        """MASE > 1 means prediction is worse than naive forecast."""
        y_train = jnp.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
        # MAE_naive = 1 (constant differences)

        y_true = jnp.array([[[6.0], [7.0]]])
        y_pred = jnp.array([[[10.0], [10.0]]])  # bad predictions, errors = [4, 3]
        # MAE_pred = 3.5

        result = mase(y_true, y_pred, y_train, seasonality=1)
        assert result > 1.0  # worse than naive

    def test_seasonality(self):
        """Seasonality parameter affects naive forecast."""
        # Seasonal pattern with period 2: [1, 10, 1, 10, 1, 10]
        y_train = jnp.array([[[1.0], [10.0], [1.0], [10.0], [1.0], [10.0]]])

        # With seasonality=2, naive predicts perfectly (errors = 0)
        # This would cause division by zero, so we check it uses epsilon
        y_true = jnp.array([[[1.0], [10.0]]])
        y_pred = jnp.array([[[1.0], [10.0]]])

        result = mase(y_true, y_pred, y_train, seasonality=2)
        # Should not be NaN/inf due to zero-handling
        assert jnp.isfinite(result)


class TestQuantileLoss:
    """Tests for pinball/quantile loss."""

    def test_perfect_prediction(self):
        """Loss is zero when prediction equals target for all quantiles."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        quantiles = jnp.array([0.1, 0.5, 0.9])
        # y_pred shape: (1, 3, 1, 3) - same value for all quantiles
        y_pred = jnp.array([[[[1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0]]]])

        result = quantile_loss(y_true, y_pred, quantiles)
        assert_trees_all_close(result, 0.0)

    def test_asymmetric_penalty(self):
        """Higher quantiles penalize under-prediction more."""
        y_true = jnp.array([[[10.0]]])  # (1, 1, 1)

        # Under-prediction: pred < true
        y_pred_under = jnp.array([[[[5.0]]]])  # (1, 1, 1, 1)

        # Over-prediction: pred > true
        y_pred_over = jnp.array([[[[15.0]]]])  # (1, 1, 1, 1)

        # High quantile (0.9) should penalize under-prediction more
        q_high = jnp.array([0.9])
        loss_under_high = quantile_loss(y_true, y_pred_under, q_high)
        loss_over_high = quantile_loss(y_true, y_pred_over, q_high)
        assert loss_under_high > loss_over_high

        # Low quantile (0.1) should penalize over-prediction more
        q_low = jnp.array([0.1])
        loss_under_low = quantile_loss(y_true, y_pred_under, q_low)
        loss_over_low = quantile_loss(y_true, y_pred_over, q_low)
        assert loss_over_low > loss_under_low

    def test_median_symmetric(self):
        """At q=0.5 (median), over and under prediction have equal penalty."""
        y_true = jnp.array([[[10.0]]])
        quantiles = jnp.array([0.5])

        y_pred_under = jnp.array([[[[5.0]]]])   # error = 5
        y_pred_over = jnp.array([[[[15.0]]]])   # error = 5

        loss_under = quantile_loss(y_true, y_pred_under, quantiles)
        loss_over = quantile_loss(y_true, y_pred_over, quantiles)
        assert_trees_all_close(loss_under, loss_over)

    def test_with_mask(self):
        """Mask excludes elements from calculation."""
        y_true = jnp.array([[[1.0], [2.0]]])  # (1, 2, 1)
        quantiles = jnp.array([0.5])
        y_pred = jnp.array([[[[1.0]], [[100.0]]]])  # (1, 2, 1, 1), second way off
        mask = jnp.array([[[1.0], [0.0]]])  # ignore second

        result = quantile_loss(y_true, y_pred, quantiles, mask)
        assert_trees_all_close(result, 0.0)

    def test_multiple_quantiles(self):
        """Works with multiple quantiles."""
        y_true = jnp.array([[[5.0]]])  # (1, 1, 1)
        quantiles = jnp.array([0.1, 0.5, 0.9])
        # Predictions for each quantile: [3, 5, 7]
        y_pred = jnp.array([[[[3.0, 5.0, 7.0]]]])  # (1, 1, 1, 3)

        result = quantile_loss(y_true, y_pred, quantiles)
        assert jnp.isfinite(result)
        assert result > 0  # not perfect since q10 and q90 are off


class TestCoverage:
    """Tests for prediction interval coverage."""

    def test_full_coverage(self):
        """Coverage is 1.0 when all points are within interval."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])
        lower = jnp.array([[[0.0], [1.0], [2.0]]])
        upper = jnp.array([[[2.0], [3.0], [4.0]]])

        assert_trees_all_close(coverage(y_true, lower, upper), 1.0)

    def test_zero_coverage(self):
        """Coverage is 0.0 when no points are within interval."""
        y_true = jnp.array([[[10.0], [20.0], [30.0]]])
        lower = jnp.array([[[0.0], [0.0], [0.0]]])
        upper = jnp.array([[[1.0], [1.0], [1.0]]])

        assert_trees_all_close(coverage(y_true, lower, upper), 0.0)

    def test_partial_coverage(self):
        """Coverage correctly computes fraction within interval."""
        y_true = jnp.array([[[1.0], [10.0]]])  # first in, second out
        lower = jnp.array([[[0.0], [0.0]]])
        upper = jnp.array([[[2.0], [2.0]]])

        assert_trees_all_close(coverage(y_true, lower, upper), 0.5)

    def test_boundary_inclusive(self):
        """Points exactly on boundary are counted as covered."""
        y_true = jnp.array([[[1.0], [2.0]]])  # exactly at bounds
        lower = jnp.array([[[1.0], [0.0]]])   # y_true[0] == lower[0]
        upper = jnp.array([[[3.0], [2.0]]])   # y_true[1] == upper[1]

        assert_trees_all_close(coverage(y_true, lower, upper), 1.0)

    def test_with_mask(self):
        """Mask excludes elements from calculation."""
        y_true = jnp.array([[[1.0], [100.0]]])  # second way outside
        lower = jnp.array([[[0.0], [0.0]]])
        upper = jnp.array([[[2.0], [2.0]]])
        mask = jnp.array([[[1.0], [0.0]]])  # ignore second

        assert_trees_all_close(coverage(y_true, lower, upper, mask), 1.0)


class TestMetricsNumericalStability:
    """Test numerical stability of metrics."""

    def test_mae_large_values(self):
        """MAE handles large values without overflow."""
        y_true = jnp.array([[[1e10]]])
        y_pred = jnp.array([[[1e10 + 1]]])
        result = mae(y_true, y_pred)
        assert jnp.isfinite(result)

    def test_smape_small_values(self):
        """SMAPE handles small values gracefully."""
        y_true = jnp.array([[[1e-10]]])
        y_pred = jnp.array([[[2e-10]]])
        result = smape(y_true, y_pred)
        assert jnp.isfinite(result)

    def test_quantile_loss_no_nan(self):
        """Quantile loss doesn't produce NaN."""
        y_true = jnp.array([[[0.0], [1.0], [-1.0]]])
        quantiles = jnp.array([0.1, 0.5, 0.9])
        y_pred = jnp.zeros((1, 3, 1, 3))
        result = quantile_loss(y_true, y_pred, quantiles)
        assert jnp.isfinite(result)
