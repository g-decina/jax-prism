"""Tests for distribution heads."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.distributions.gaussian import GaussianHead
from jax_prism.distributions.quantile import QuantileHead


class TestGaussianHeadParamsFromRaw:
    """Tests for GaussianHead.params_from_raw."""

    def test_output_keys(self, rng_key):
        """Should return dict with 'loc' and 'scale'."""
        head = GaussianHead()
        raw = jax.random.normal(rng_key, (4, 8, 2))

        params = head.params_from_raw(raw)

        assert "loc" in params
        assert "scale" in params

    def test_output_shapes(self, rng_key):
        """Params should have shape (..., 1) from input (..., 2)."""
        head = GaussianHead()
        raw = jax.random.normal(rng_key, (4, 8, 2))

        params = head.params_from_raw(raw)

        # Shapes match TimeSeriesBatch convention: (B, T, 1)
        chex.assert_shape(params["loc"], (4, 8, 1))
        chex.assert_shape(params["scale"], (4, 8, 1))

    def test_scale_positive(self, rng_key):
        """Scale should always be positive."""
        head = GaussianHead()
        # Include very negative values to test softplus
        raw = jax.random.normal(rng_key, (100, 2)) * 10

        params = head.params_from_raw(raw)

        assert jnp.all(params["scale"] > 0)

    def test_scale_min_value(self, rng_key):
        """Scale should be at least min_scale."""
        min_scale = 1e-3
        head = GaussianHead(min_scale=min_scale)
        raw = jnp.zeros((4, 2))  # softplus(0) ≈ 0.693

        params = head.params_from_raw(raw)

        assert jnp.all(params["scale"] >= min_scale)

    def test_loc_unchanged(self, rng_key):
        """Loc should equal raw[..., 0:1] unchanged (preserving trailing dim)."""
        head = GaussianHead()
        raw = jax.random.normal(rng_key, (4, 8, 2))

        params = head.params_from_raw(raw)

        chex.assert_trees_all_close(params["loc"], raw[..., 0:1], atol=0)


class TestGaussianHeadLogProb:
    """Tests for GaussianHead.log_prob."""

    def test_output_shape(self, rng_key):
        """Log prob should match target shape."""
        head = GaussianHead()
        params = {"loc": jnp.zeros((4, 8)), "scale": jnp.ones((4, 8))}
        targets = jax.random.normal(rng_key, (4, 8))

        log_probs = head.log_prob(params, targets)

        chex.assert_shape(log_probs, (4, 8))

    def test_standard_normal_at_zero(self):
        """Log prob of 0 under N(0,1) should be -0.5*log(2π)."""
        head = GaussianHead()
        params = {"loc": jnp.array([0.0]), "scale": jnp.array([1.0])}
        targets = jnp.array([0.0])

        log_prob = head.log_prob(params, targets)

        expected = -0.5 * jnp.log(2 * jnp.pi)
        chex.assert_trees_all_close(log_prob, expected, atol=1e-5)

    def test_higher_prob_at_mean(self, rng_key):
        """Log prob should be highest at the mean."""
        head = GaussianHead()
        loc = jax.random.normal(rng_key, (10,))
        scale = jnp.ones(10)
        params = {"loc": loc, "scale": scale}

        log_prob_at_mean = head.log_prob(params, loc)
        log_prob_away = head.log_prob(params, loc + 2 * scale)

        assert jnp.all(log_prob_at_mean > log_prob_away)

    def test_smaller_scale_sharper_peak(self):
        """Smaller scale should give higher log prob at mean."""
        head = GaussianHead()
        loc = jnp.array([0.0])
        targets = jnp.array([0.0])

        params_narrow = {"loc": loc, "scale": jnp.array([0.5])}
        params_wide = {"loc": loc, "scale": jnp.array([2.0])}

        lp_narrow = head.log_prob(params_narrow, targets)
        lp_wide = head.log_prob(params_wide, targets)

        assert lp_narrow > lp_wide

    def test_no_nan(self, rng_key):
        """Should not produce NaN."""
        head = GaussianHead()
        raw = jax.random.normal(rng_key, (4, 8, 2))
        params = head.params_from_raw(raw)
        # Targets must match params shape convention (B, T, 1)
        targets = jax.random.normal(rng_key, (4, 8, 1))

        log_probs = head.log_prob(params, targets)

        assert not jnp.any(jnp.isnan(log_probs))


class TestGaussianHeadSample:
    """Tests for GaussianHead.sample."""

    def test_output_shape_no_sample_shape(self, rng_key):
        """Sample shape should match param shape when sample_shape=()."""
        head = GaussianHead()
        params = {"loc": jnp.zeros((4, 8)), "scale": jnp.ones((4, 8))}

        samples = head.sample(params, rng_key, sample_shape=())

        chex.assert_shape(samples, (4, 8))

    def test_output_shape_with_sample_shape(self, rng_key):
        """Sample shape should be (*sample_shape, *param_shape)."""
        head = GaussianHead()
        params = {"loc": jnp.zeros((4, 8)), "scale": jnp.ones((4, 8))}

        samples = head.sample(params, rng_key, sample_shape=(10,))

        chex.assert_shape(samples, (10, 4, 8))

    def test_samples_around_mean(self, rng_key):
        """Sample mean should approximate loc for many samples."""
        head = GaussianHead()
        loc = jnp.array([5.0])
        scale = jnp.array([0.1])
        params = {"loc": loc, "scale": scale}

        samples = head.sample(params, rng_key, sample_shape=(10000,))
        sample_mean = jnp.mean(samples)

        chex.assert_trees_all_close(sample_mean, loc[0], atol=0.05)

    def test_samples_std_matches_scale(self, rng_key):
        """Sample std should approximate scale for many samples."""
        head = GaussianHead()
        loc = jnp.array([0.0])
        scale = jnp.array([2.0])
        params = {"loc": loc, "scale": scale}

        samples = head.sample(params, rng_key, sample_shape=(10000,))
        sample_std = jnp.std(samples)

        chex.assert_trees_all_close(sample_std, scale[0], atol=0.1)

    def test_different_keys_different_samples(self, rng_key):
        """Different RNG keys should produce different samples."""
        head = GaussianHead()
        params = {"loc": jnp.zeros((4,)), "scale": jnp.ones((4,))}

        key1, key2 = jax.random.split(rng_key)
        samples1 = head.sample(params, key1)
        samples2 = head.sample(params, key2)

        assert not jnp.allclose(samples1, samples2)


class TestGaussianHeadMean:
    """Tests for GaussianHead.mean."""

    def test_returns_loc(self, rng_key):
        """Mean should equal loc."""
        head = GaussianHead()
        loc = jax.random.normal(rng_key, (4, 8))
        params = {"loc": loc, "scale": jnp.ones((4, 8))}

        mean = head.mean(params)

        chex.assert_trees_all_close(mean, loc, atol=0)


class TestGaussianHeadQuantile:
    """Tests for GaussianHead.quantile."""

    def test_output_shape(self, rng_key):
        """Quantile output shape should be (..., num_quantiles)."""
        head = GaussianHead()
        # Params need trailing dim for proper broadcasting with q
        params = {"loc": jnp.zeros((4, 8, 1)), "scale": jnp.ones((4, 8, 1))}
        q = jnp.array([0.1, 0.5, 0.9])

        quantiles = head.quantile(params, q)

        # (B, T, 1) * (Q,) broadcasts to (B, T, Q)
        chex.assert_shape(quantiles, (4, 8, 3))

    def test_median_equals_mean(self):
        """For Gaussian, median (q=0.5) should equal mean."""
        head = GaussianHead()
        # Add trailing dim for proper broadcasting
        loc = jnp.array([[3.0], [-2.0], [0.0]])  # (3, 1)
        params = {"loc": loc, "scale": jnp.array([[1.0], [2.0], [0.5]])}  # (3, 1)
        q = jnp.array([0.5])

        quantiles = head.quantile(params, q)

        # quantiles shape is (3, 1), loc is (3, 1)
        chex.assert_trees_all_close(quantiles, loc, atol=1e-5)

    def test_quantiles_ordered(self, rng_key):
        """Lower quantile levels should give lower values."""
        head = GaussianHead()
        # Add trailing dim for proper broadcasting with q
        params = {"loc": jnp.zeros((4, 1)), "scale": jnp.ones((4, 1))}
        q = jnp.array([0.1, 0.5, 0.9])

        quantiles = head.quantile(params, q)

        # quantiles shape is (4, 3): q=0.1 < q=0.5 < q=0.9
        assert jnp.all(quantiles[..., 0] < quantiles[..., 1])
        assert jnp.all(quantiles[..., 1] < quantiles[..., 2])

    def test_symmetric_quantiles(self):
        """For symmetric Gaussian, q and 1-q should be equidistant from mean."""
        head = GaussianHead()
        loc = jnp.array([0.0])
        scale = jnp.array([1.0])
        params = {"loc": loc, "scale": scale}
        q = jnp.array([0.1, 0.9])

        quantiles = head.quantile(params, q)

        # |q_0.1 - mean| should equal |q_0.9 - mean|
        dist_low = jnp.abs(quantiles[..., 0] - loc)
        dist_high = jnp.abs(quantiles[..., 1] - loc)

        chex.assert_trees_all_close(dist_low, dist_high, atol=1e-5)

    def test_scale_affects_spread(self):
        """Larger scale should give wider quantile spread."""
        head = GaussianHead()
        q = jnp.array([0.1, 0.9])

        params_narrow = {"loc": jnp.array([0.0]), "scale": jnp.array([1.0])}
        params_wide = {"loc": jnp.array([0.0]), "scale": jnp.array([2.0])}

        q_narrow = head.quantile(params_narrow, q)
        q_wide = head.quantile(params_wide, q)

        spread_narrow = q_narrow[..., 1] - q_narrow[..., 0]
        spread_wide = q_wide[..., 1] - q_wide[..., 0]

        assert spread_wide > spread_narrow


class TestGaussianHeadNumParams:
    """Tests for GaussianHead.num_params."""

    def test_num_params_is_two(self):
        """Gaussian has 2 parameters (loc, scale)."""
        head = GaussianHead()
        assert head.num_params == 2


class TestGaussianHeadJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """All methods should be JIT-compilable."""
        head = GaussianHead()
        # Use 3D input to match TimeSeriesBatch convention
        raw = jax.random.normal(rng_key, (4, 8, 2))

        @jax.jit
        def forward(raw):
            params = head.params_from_raw(raw)
            # Targets must have matching shape (B, T, 1)
            targets = jnp.zeros((4, 8, 1))
            lp = head.log_prob(params, targets)
            samples = head.sample(params, rng_key)
            quants = head.quantile(params, jnp.array([0.5]))
            return lp, samples, quants

        lp, samples, quants = forward(raw)

        chex.assert_shape(lp, (4, 8, 1))
        chex.assert_shape(samples, (4, 8, 1))
        # quantile broadcasts q=(1,) with loc=(4,8,1) -> (4,8,1)
        chex.assert_shape(quants, (4, 8, 1))

    def test_vmap_compatible(self, rng_key):
        """Methods should work with vmap."""
        head = GaussianHead()
        raw = jax.random.normal(rng_key, (8, 4, 2))

        # vmap over batch dimension
        vmapped_params = jax.vmap(head.params_from_raw)(raw)

        # Trailing dimension preserved: (outer_batch, inner_batch, 1)
        chex.assert_shape(vmapped_params["loc"], (8, 4, 1))
        chex.assert_shape(vmapped_params["scale"], (8, 4, 1))

    def test_gradient_through_log_prob(self, rng_key):
        """Gradients should flow through log_prob."""
        head = GaussianHead()
        raw = jax.random.normal(rng_key, (4, 2))
        targets = jax.random.normal(rng_key, (4,))

        def loss_fn(raw):
            params = head.params_from_raw(raw)
            return -jnp.mean(head.log_prob(params, targets))

        grads = jax.grad(loss_fn)(raw)

        assert not jnp.any(jnp.isnan(grads))
        assert not jnp.allclose(grads, 0)


# ============================================================================
# QuantileHead Tests
# ============================================================================


class TestQuantileHeadParamsFromRaw:
    """Tests for QuantileHead.params_from_raw."""

    def test_output_keys(self, rng_key):
        """Should return dict with 'quantile_values' and 'quantile_levels'."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        raw = jax.random.normal(rng_key, (4, 8, 3))

        params = head.params_from_raw(raw)

        assert "quantile_values" in params
        assert "quantile_levels" in params

    def test_output_shapes(self, rng_key):
        """Quantile values should match raw shape."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        raw = jax.random.normal(rng_key, (4, 8, 3))

        params = head.params_from_raw(raw)

        chex.assert_shape(params["quantile_values"], (4, 8, 3))
        chex.assert_shape(params["quantile_levels"], (3,))

    def test_values_unchanged(self, rng_key):
        """Raw values should pass through unchanged."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        raw = jax.random.normal(rng_key, (4, 8, 3))

        params = head.params_from_raw(raw)

        chex.assert_trees_all_close(params["quantile_values"], raw, atol=0)

    def test_levels_match_init(self):
        """Quantile levels should match initialization."""
        quantiles = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
        head = QuantileHead(quantiles)
        raw = jnp.zeros((4, 5))

        params = head.params_from_raw(raw)

        chex.assert_trees_all_close(params["quantile_levels"], quantiles, atol=0)


class TestQuantileHeadNumParams:
    """Tests for QuantileHead.num_params."""

    def test_num_params_matches_quantiles(self):
        """num_params should equal number of quantile levels."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        assert head.num_params == 3

    def test_num_params_single_quantile(self):
        """Single quantile should give num_params=1."""
        quantiles = jnp.array([0.5])
        head = QuantileHead(quantiles)
        assert head.num_params == 1


class TestQuantileHeadLogProb:
    """Tests for QuantileHead.log_prob."""

    def test_raises_not_implemented(self, rng_key):
        """log_prob should raise NotImplementedError."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        params = {
            "quantile_values": jax.random.normal(rng_key, (4, 3)),
            "quantile_levels": quantiles,
        }
        targets = jax.random.normal(rng_key, (4,))

        with pytest.raises(NotImplementedError):
            head.log_prob(params, targets)


class TestQuantileHeadSample:
    """Tests for QuantileHead.sample."""

    def test_output_shape_no_sample_shape(self, rng_key):
        """Sample shape should match batch dims when sample_shape=()."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        # Sorted quantile values for interpolation
        qv = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        samples = head.sample(params, rng_key, sample_shape=())

        chex.assert_shape(samples, (2,))

    def test_output_shape_with_sample_shape(self, rng_key):
        """Sample shape should be (*sample_shape, ...)."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        samples = head.sample(params, rng_key, sample_shape=(10,))

        chex.assert_shape(samples, (10, 2))

    def test_samples_within_range(self, rng_key):
        """Samples should be within (or near) quantile value range."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jnp.array([[1.0, 2.0, 3.0]])  # min=1, max=3
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        samples = head.sample(params, rng_key, sample_shape=(1000,))

        # Most samples should be in range (interpolation extends to edges)
        assert jnp.all(samples >= 0.5)  # Allow some margin
        assert jnp.all(samples <= 3.5)

    def test_different_keys_different_samples(self, rng_key):
        """Different RNG keys should produce different samples."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jnp.array([[1.0, 2.0, 3.0]])
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        key1, key2 = jax.random.split(rng_key)
        samples1 = head.sample(params, key1, sample_shape=(10,))
        samples2 = head.sample(params, key2, sample_shape=(10,))

        assert not jnp.allclose(samples1, samples2)


class TestQuantileHeadMedian:
    """Tests for QuantileHead.median."""

    def test_output_shape(self, rng_key):
        """Median should have shape (...)."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jax.random.normal(rng_key, (4, 8, 3))
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        median = head.median(params)

        chex.assert_shape(median, (4, 8, 1))

    def test_median_equals_middle_quantile(self):
        """When 0.5 is in quantiles, median should equal that value."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jnp.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]])
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        median = head.median(params)

        chex.assert_trees_all_close(median, jnp.array([[5.0], [6.0]]), atol=1e-5)

    def test_median_interpolation(self):
        """Median should interpolate when 0.5 is not in quantiles."""
        quantiles = jnp.array([0.25, 0.75])  # No 0.5
        head = QuantileHead(quantiles)
        qv = jnp.array([[2.0, 6.0]])  # Should interpolate to 4.0
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        median = head.median(params)

        chex.assert_trees_all_close(median, jnp.array([[4.0]]), atol=1e-5)


class TestQuantileHeadMean:
    """Tests for QuantileHead.mean."""

    def test_mean_equals_median(self, rng_key):
        """Mean should return median for quantile head."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jax.random.normal(rng_key, (4, 8, 3))
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        mean = head.mean(params)
        median = head.median(params)

        chex.assert_trees_all_close(mean, median, atol=0)


class TestQuantileHeadPredictionInterval:
    """Tests for QuantileHead.prediction_interval."""

    def test_output_shapes(self, rng_key):
        """Bounds should have shape (...)."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jax.random.normal(rng_key, (4, 8, 3))
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        lower, upper = head.prediction_interval(params, coverage=0.8)

        chex.assert_shape(lower, (4, 8, 1))
        chex.assert_shape(upper, (4, 8, 1))

    def test_lower_less_than_upper(self, rng_key):
        """Lower bound should be less than upper bound."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        # Ensure quantile values are sorted
        qv = jnp.sort(jax.random.normal(rng_key, (4, 8, 3)), axis=-1)
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        lower, upper = head.prediction_interval(params, coverage=0.8)

        assert jnp.all(lower < upper)

    def test_coverage_matches_quantiles(self):
        """80% coverage should use 0.1 and 0.9 quantiles."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        qv = jnp.array([[1.0, 5.0, 9.0]])
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        lower, upper = head.prediction_interval(params, coverage=0.8)

        # For 80% coverage: lower_q=0.1, upper_q=0.9
        chex.assert_trees_all_close(lower, jnp.array([[1.0]]), atol=1e-5)
        chex.assert_trees_all_close(upper, jnp.array([[9.0]]), atol=1e-5)

    def test_wider_coverage(self):
        """Higher coverage should give wider intervals."""
        quantiles = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
        head = QuantileHead(quantiles)
        qv = jnp.array([[1.0, 3.0, 5.0, 7.0, 9.0]])
        params = {"quantile_values": qv, "quantile_levels": quantiles}

        lower_50, upper_50 = head.prediction_interval(params, coverage=0.5)
        lower_90, upper_90 = head.prediction_interval(params, coverage=0.9)

        width_50 = upper_50 - lower_50
        width_90 = upper_90 - lower_90

        assert width_90 > width_50


class TestQuantileHeadJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatible(self, rng_key):
        """All methods should be JIT-compilable."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        raw = jax.random.normal(rng_key, (4, 3))

        @jax.jit
        def forward(raw, key):
            params = head.params_from_raw(raw)
            median = head.median(params)
            samples = head.sample(params, key)
            lower, upper = head.prediction_interval(params, coverage=0.8)
            return median, samples, lower, upper

        median, samples, lower, upper = forward(raw, rng_key)

        chex.assert_shape(median, (4, 1))
        chex.assert_shape(samples, (4, ))
        chex.assert_shape(lower, (4, 1))

    def test_vmap_compatible(self, rng_key):
        """Methods should work with vmap."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        raw = jax.random.normal(rng_key, (8, 4, 3))

        # vmap over batch dimension
        vmapped_params = jax.vmap(head.params_from_raw)(raw)

        chex.assert_shape(vmapped_params["quantile_values"], (8, 4, 3))

    def test_gradient_through_median(self, rng_key):
        """Gradients should flow through median."""
        quantiles = jnp.array([0.1, 0.5, 0.9])
        head = QuantileHead(quantiles)
        raw = jax.random.normal(rng_key, (4, 3))

        def loss_fn(raw):
            params = head.params_from_raw(raw)
            return jnp.mean(head.median(params))

        grads = jax.grad(loss_fn)(raw)

        assert not jnp.any(jnp.isnan(grads))