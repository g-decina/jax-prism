"""Tests for metrics/point.py, metrics/probabilistic.py, metrics/calibration.py, and metrics/crps.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import assert_trees_all_close

from jax_prism._typing import UncertaintyOutput
from jax_prism.metrics.calibration import quantile_calibration_error
from jax_prism.metrics.crps import crps_gaussian
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


class TestQuantileCalibrationError:
    """Tests for Quantile Calibration Error."""

    def test_perfect_calibration(self):
        """QCE is zero when empirical coverage matches quantile levels."""
        # 10 observations, quantiles [0.1, 0.5, 0.9]
        # For perfect calibration: 1 below q10, 5 below q50, 9 below q90
        q_levels = jnp.array([0.1, 0.5, 0.9])

        # y_true: values 1-10
        y_true = jnp.arange(1, 11).reshape(10, 1, 1).astype(jnp.float32)  # (10, 1, 1)

        # q_values: set thresholds so exactly right fractions fall below
        # q10 = 1.5 (1 value below = 10%), q50 = 5.5 (5 below = 50%), q90 = 9.5 (9 below = 90%)
        q_values = jnp.array([[[1.5, 5.5, 9.5]]]).repeat(10, axis=0)  # (10, 1, 3)

        result = quantile_calibration_error(q_values, q_levels, y_true)
        assert_trees_all_close(result, 0.0, atol=1e-5)

    def test_overconfident_model(self):
        """QCE > 0 when model is overconfident (intervals too narrow)."""
        q_levels = jnp.array([0.1, 0.9])

        # All y_true values are 5.0
        y_true = jnp.full((10, 1, 1), 5.0)

        # Overconfident: q10=4.9, q90=5.1 (very tight interval)
        # All 10 values are >= 4.9, so p_hat[0] = 0.0 (should be 0.1)
        # All 10 values are < 5.1, so p_hat[1] = 1.0 (should be 0.9)
        q_values = jnp.array([[[4.9, 5.1]]]).repeat(10, axis=0)  # (10, 1, 2)

        result = quantile_calibration_error(q_values, q_levels, y_true)
        # Error: |0.1 - 0.0| + |0.9 - 1.0| = 0.1 + 0.1 = 0.2, mean = 0.1
        assert_trees_all_close(result, 0.1, atol=1e-5)

    def test_underconfident_model(self):
        """QCE > 0 when model is underconfident (intervals too wide)."""
        q_levels = jnp.array([0.1, 0.9])

        # Values uniformly from 1 to 10
        y_true = jnp.arange(1, 11).reshape(10, 1, 1).astype(jnp.float32)

        # Underconfident: q10=0.0 (all above), q90=100.0 (all below)
        # p_hat[0] = 0.0 (should be 0.1), p_hat[1] = 1.0 (should be 0.9)
        q_values = jnp.array([[[0.0, 100.0]]]).repeat(10, axis=0)

        result = quantile_calibration_error(q_values, q_levels, y_true)
        # Same error as overconfident case
        assert_trees_all_close(result, 0.1, atol=1e-5)

    def test_with_mask(self):
        """Mask excludes elements from QCE calculation."""
        q_levels = jnp.array([0.5])

        # 4 observations: [1, 2, 3, 100] but mask out the outlier
        y_true = jnp.array([[[[1.0]], [[2.0]], [[3.0]], [[100.0]]]])  # (1, 4, 1)
        y_true = y_true.reshape(1, 4, 1)

        # q50 = 2.5: 2 of 4 below (50%) - but with mask, 2 of 3 below (66%)
        q_values = jnp.array([[[2.5]]]).repeat(4, axis=1)  # (1, 4, 1)

        mask = jnp.array([[[1.0], [1.0], [1.0], [0.0]]])  # ignore last

        result = quantile_calibration_error(q_values, q_levels, y_true, mask)
        # p_hat = 2/3 ≈ 0.667, error = |0.5 - 0.667| ≈ 0.167
        assert_trees_all_close(result, 1/6, atol=1e-5)

    def test_shape_validation(self):
        """Raises error when q_values and q_levels dimensions mismatch."""
        q_levels = jnp.array([0.1, 0.5, 0.9])  # 3 quantiles
        q_values = jnp.zeros((4, 8, 2))  # only 2 quantiles
        y_true = jnp.zeros((4, 8, 1))

        with pytest.raises(ValueError):
            quantile_calibration_error(q_values, q_levels, y_true)

    def test_output_range(self):
        """QCE is bounded in [0, 1]."""
        q_levels = jnp.array([0.1, 0.5, 0.9])
        y_true = jnp.zeros((10, 5, 1))
        q_values = jnp.ones((10, 5, 3)) * 100  # all predictions way above

        result = quantile_calibration_error(q_values, q_levels, y_true)
        assert result >= 0.0
        assert result <= 1.0

    def test_batch_and_time_aggregation(self):
        """QCE correctly aggregates across batch and time dimensions."""
        q_levels = jnp.array([0.5])

        # 2 batches, 2 timesteps each = 4 total observations
        # Values: [[1, 2], [3, 4]] -> flattened: [1, 2, 3, 4]
        y_true = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])  # (2, 2, 1)

        # q50 = 2.5: 2 of 4 below = 50% -> perfect calibration
        q_values = jnp.full((2, 2, 1), 2.5)

        result = quantile_calibration_error(q_values, q_levels, y_true)
        assert_trees_all_close(result, 0.0, atol=1e-5)


class TestCRPSGaussian:
    """Tests for Continuous Ranked Probability Score (Gaussian)."""

    def test_perfect_prediction_zero_variance(self):
        """CRPS approaches 0 as sigma -> 0 when mu = y_true."""
        y_true = jnp.array([[[5.0]]])
        mu = jnp.array([[[5.0]]])
        sigma = jnp.array([[[1e-6]]])  # very small sigma

        result = crps_gaussian(y_true, mu, sigma)
        assert_trees_all_close(result, 0.0, atol=1e-4)

    def test_crps_positive(self):
        """CRPS is always non-negative."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])
        mu = jnp.array([[[0.0], [5.0], [3.0]]])
        sigma = jnp.array([[[1.0], [2.0], [0.5]]])

        result = crps_gaussian(y_true, mu, sigma)
        assert result >= 0.0

    def test_crps_increases_with_error(self):
        """CRPS increases when prediction is further from truth."""
        y_true = jnp.array([[[0.0]]])
        sigma = jnp.array([[[1.0]]])

        mu_close = jnp.array([[[0.5]]])
        mu_far = jnp.array([[[5.0]]])

        crps_close = crps_gaussian(y_true, mu_close, sigma)
        crps_far = crps_gaussian(y_true, mu_far, sigma)

        assert crps_far > crps_close

    def test_crps_increases_with_uncertainty(self):
        """CRPS increases with larger sigma when mu = y_true.

        This tests the sharpness penalty: wider distributions are penalized
        even when centered correctly.
        """
        y_true = jnp.array([[[0.0]]])
        mu = jnp.array([[[0.0]]])  # perfect mean

        sigma_narrow = jnp.array([[[0.5]]])
        sigma_wide = jnp.array([[[2.0]]])

        crps_narrow = crps_gaussian(y_true, mu, sigma_narrow)
        crps_wide = crps_gaussian(y_true, mu, sigma_wide)

        assert crps_wide > crps_narrow

    def test_known_value(self):
        """CRPS matches known analytical value.

        For N(0, 1) evaluated at y=0:
        CRPS = σ · [z·(2Φ(z) - 1) + 2φ(z) - 1/√π]
             = 1 · [0·(2·0.5 - 1) + 2·(1/√(2π)) - 1/√π]
             = 2/√(2π) - 1/√π
             = √(2/π) - 1/√π
             = (√2 - 1)/√π
             ≈ 0.2338
        """
        y_true = jnp.array([[[0.0]]])
        mu = jnp.array([[[0.0]]])
        sigma = jnp.array([[[1.0]]])

        result = crps_gaussian(y_true, mu, sigma)
        expected = (jnp.sqrt(2.0) - 1.0) / jnp.sqrt(jnp.pi)
        assert_trees_all_close(result, expected, atol=1e-5)

    def test_reduces_to_mae_for_point_mass(self):
        """As sigma -> 0, CRPS -> MAE.

        For a point mass distribution, CRPS equals the absolute error.
        """
        y_true = jnp.array([[[3.0]]])
        mu = jnp.array([[[5.0]]])
        sigma = jnp.array([[[1e-8]]])  # effectively a point mass

        result = crps_gaussian(y_true, mu, sigma)
        expected_mae = jnp.abs(y_true - mu).mean()

        assert_trees_all_close(result, expected_mae, atol=1e-4)

    def test_with_mask(self):
        """Mask excludes elements from CRPS calculation."""
        y_true = jnp.array([[[1.0], [1000.0]]])  # second is outlier
        mu = jnp.array([[[1.0], [1.0]]])
        sigma = jnp.array([[[1.0], [1.0]]])
        mask = jnp.array([[[1.0], [0.0]]])  # ignore second

        result = crps_gaussian(y_true, mu, sigma, mask)
        # Should only consider first element where mu = y_true
        expected = (jnp.sqrt(2.0) - 1.0) / jnp.sqrt(jnp.pi)  # CRPS for z=0
        assert_trees_all_close(result, expected, atol=1e-5)

    def test_batch_aggregation(self):
        """CRPS correctly aggregates across batch dimension."""
        # Two batches with same setup
        y_true = jnp.array([[[0.0]], [[0.0]]])  # (2, 1, 1)
        mu = jnp.array([[[0.0]], [[0.0]]])
        sigma = jnp.array([[[1.0]], [[1.0]]])

        result = crps_gaussian(y_true, mu, sigma)
        expected = (jnp.sqrt(2.0) - 1.0) / jnp.sqrt(jnp.pi)
        assert_trees_all_close(result, expected, atol=1e-5)

    def test_numerical_stability_large_z(self):
        """CRPS handles large standardized residuals without overflow."""
        y_true = jnp.array([[[100.0]]])
        mu = jnp.array([[[0.0]]])
        sigma = jnp.array([[[1.0]]])  # z = 100

        result = crps_gaussian(y_true, mu, sigma)
        assert jnp.isfinite(result)
        # For large z, CRPS ≈ |y - mu| = 100
        assert_trees_all_close(result, 100.0, atol=1.0)

    def test_numerical_stability_small_sigma(self):
        """CRPS handles small sigma without numerical issues."""
        y_true = jnp.array([[[1.0]]])
        mu = jnp.array([[[1.0]]])
        sigma = jnp.array([[[1e-10]]])

        result = crps_gaussian(y_true, mu, sigma)
        assert jnp.isfinite(result)
        assert result >= 0.0


class TestUncertaintyOutput:
    """Tests for UncertaintyOutput dataclass."""

    def test_basic_construction(self):
        """UncertaintyOutput can be constructed with required fields."""
        point = jnp.array([[[1.0], [2.0]]])
        lower = jnp.array([[[0.5], [1.5]]])
        upper = jnp.array([[[1.5], [2.5]]])
        levels = (0.8,)

        output = UncertaintyOutput(
            point=point,
            lower=lower,
            upper=upper,
            levels=levels,
        )

        assert output.point.shape == (1, 2, 1)
        assert output.lower.shape == (1, 2, 1)
        assert output.upper.shape == (1, 2, 1)
        assert output.levels == (0.8,)

    def test_optional_fields_default_none(self):
        """Optional fields default to None."""
        output = UncertaintyOutput(
            point=jnp.zeros((1, 2, 1)),
            lower=jnp.zeros((1, 2, 1)),
            upper=jnp.zeros((1, 2, 1)),
            levels=(0.9,),
        )

        assert output.mu is None
        assert output.sigma is None
        assert output.samples is None

    def test_with_gaussian_fields(self):
        """UncertaintyOutput can store Gaussian distribution parameters."""
        mu = jnp.array([[[5.0]]])
        sigma = jnp.array([[[1.0]]])

        output = UncertaintyOutput(
            point=mu,
            lower=mu - 1.96 * sigma,
            upper=mu + 1.96 * sigma,
            levels=(0.95,),
            mu=mu,
            sigma=sigma,
        )

        assert_trees_all_close(output.mu, mu)
        assert_trees_all_close(output.sigma, sigma)
        # Can use with CRPS
        crps = crps_gaussian(mu, output.mu, output.sigma)
        assert jnp.isfinite(crps)

    def test_with_samples(self):
        """UncertaintyOutput can store posterior samples."""
        num_samples = 100
        samples = jnp.ones((num_samples, 2, 4, 1))
        point = samples.mean(axis=0)
        lower = jnp.percentile(samples, 10, axis=0)
        upper = jnp.percentile(samples, 90, axis=0)

        output = UncertaintyOutput(
            point=point,
            lower=lower,
            upper=upper,
            levels=(0.8,),
            samples=samples,
        )

        assert output.samples.shape == (100, 2, 4, 1)

    def test_multiple_intervals(self):
        """UncertaintyOutput supports multiple confidence levels."""
        mu = jnp.array([[[5.0], [10.0]]])  # (1, 2, 1)
        sigma = jnp.array([[[1.0], [2.0]]])

        # 50% and 90% intervals
        z_50 = 0.674  # ~50% CI
        z_90 = 1.645  # ~90% CI

        lower = jnp.stack([mu - z_50 * sigma, mu - z_90 * sigma], axis=-1).squeeze(-2)
        upper = jnp.stack([mu + z_50 * sigma, mu + z_90 * sigma], axis=-1).squeeze(-2)

        output = UncertaintyOutput(
            point=mu,
            lower=lower,  # (1, 2, 2)
            upper=upper,  # (1, 2, 2)
            levels=(0.5, 0.9),
        )

        assert output.lower.shape[-1] == 2
        assert output.upper.shape[-1] == 2
        assert len(output.levels) == 2

    def test_pytree_compatibility(self):
        """UncertaintyOutput works with JAX pytree operations."""
        output = UncertaintyOutput(
            point=jnp.array([[[1.0]]]),
            lower=jnp.array([[[0.0]]]),
            upper=jnp.array([[[2.0]]]),
            levels=(0.9,),
        )

        # Should work with jax.tree.map
        doubled = jax.tree.map(lambda x: x * 2 if isinstance(x, jax.Array) else x, output)
        assert_trees_all_close(doubled.point, jnp.array([[[2.0]]]))
        assert_trees_all_close(doubled.lower, jnp.array([[[0.0]]]))
        assert_trees_all_close(doubled.upper, jnp.array([[[4.0]]]))

    def test_replace_method(self):
        """UncertaintyOutput supports flax.struct replace method."""
        output = UncertaintyOutput(
            point=jnp.array([[[1.0]]]),
            lower=jnp.array([[[0.0]]]),
            upper=jnp.array([[[2.0]]]),
            levels=(0.9,),
        )

        new_point = jnp.array([[[5.0]]])
        updated = output.replace(point=new_point)

        assert_trees_all_close(updated.point, new_point)
        # Other fields unchanged
        assert_trees_all_close(updated.lower, output.lower)
        assert_trees_all_close(updated.upper, output.upper)

    def test_integration_with_coverage_metric(self):
        """UncertaintyOutput fields work with coverage metric."""
        y_true = jnp.array([[[1.0], [2.0], [3.0]]])

        output = UncertaintyOutput(
            point=jnp.array([[[1.0], [2.0], [3.0]]]),
            lower=jnp.array([[[0.5], [1.5], [2.5]]]),
            upper=jnp.array([[[1.5], [2.5], [3.5]]]),
            levels=(0.8,),
        )

        cov = coverage(y_true, output.lower, output.upper)
        assert_trees_all_close(cov, 1.0)

    def test_numpy_conversion(self):
        """UncertaintyOutput fields can be converted to numpy."""
        output = UncertaintyOutput(
            point=jnp.array([[[1.0], [2.0]]]),
            lower=jnp.array([[[0.5], [1.5]]]),
            upper=jnp.array([[[1.5], [2.5]]]),
            levels=(0.9,),
        )

        # Direct numpy conversion works
        point_np = np.asarray(output.point)
        assert isinstance(point_np, np.ndarray)
        assert point_np.shape == (1, 2, 1)
