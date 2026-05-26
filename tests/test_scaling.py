"""Tests for data/scaling.py."""

import jax.numpy as jnp
import pytest
from chex import assert_shape, assert_trees_all_close

from jax_prism.data.scaling import (
    fixed_scale,
    inverse_scale,
    last_value_scale,
    median_scale,
)


class TestLastValueScale:
    """Tests for last_value_scale."""

    def test_output_shapes(self):
        """scaled_x has same shape as x, scale has shape (B, F)."""
        x = jnp.ones((4, 10, 3))  # B=4, T=10, F=3
        scaled_x, scale = last_value_scale(x)

        assert_shape(scaled_x, (4, 10, 3))
        assert_shape(scale, (4, 3))

    def test_scales_by_last_value(self):
        """Division by last time step value."""
        # Create array where last timestep has known values
        x = jnp.array([[[1.0], [2.0], [4.0]]])  # B=1, T=3, F=1, last=4
        scaled_x, scale = last_value_scale(x)

        assert_trees_all_close(scale, jnp.array([[4.0]]))
        assert_trees_all_close(scaled_x, jnp.array([[[0.25], [0.5], [1.0]]]))

    def test_uses_absolute_value(self):
        """Scale is always positive even with negative last value."""
        x = jnp.array([[[1.0], [2.0], [-4.0]]])  # last value is -4
        scaled_x, scale = last_value_scale(x)

        assert_trees_all_close(scale, jnp.array([[4.0]]))  # absolute value
        assert_trees_all_close(scaled_x, jnp.array([[[0.25], [0.5], [-1.0]]]))

    def test_zero_last_value_uses_one(self):
        """When last value is zero, use 1.0 to avoid division by zero."""
        x = jnp.array([[[5.0], [3.0], [0.0]]])  # last value is 0
        scaled_x, scale = last_value_scale(x)

        assert_trees_all_close(scale, jnp.array([[1.0]]))
        assert_trees_all_close(scaled_x, x)  # unchanged since scale=1

    def test_multivariate(self):
        """Works with multiple features."""
        x = jnp.array([[[1.0, 10.0], [2.0, 20.0], [4.0, 5.0]]])  # B=1, T=3, F=2
        scaled_x, scale = last_value_scale(x)

        assert_trees_all_close(scale, jnp.array([[4.0, 5.0]]))
        # Each feature scaled independently
        assert_trees_all_close(
            scaled_x,
            jnp.array([[[0.25, 2.0], [0.5, 4.0], [1.0, 1.0]]]),
        )


class TestMedianScale:
    """Tests for median_scale."""

    def test_output_shapes(self):
        """scaled_x has same shape as x, scale has shape (B, F)."""
        x = jnp.ones((4, 10, 3))
        scaled_x, scale = median_scale(x)

        assert_shape(scaled_x, (4, 10, 3))
        assert_shape(scale, (4, 3))

    def test_uses_median_of_all_values(self):
        """When k=None, uses median of entire series."""
        x = jnp.array([[[1.0], [3.0], [5.0], [7.0], [9.0]]])  # median=5
        scaled_x, scale = median_scale(x)

        assert_trees_all_close(scale, jnp.array([[5.0]]))

    def test_uses_last_k_values(self):
        """When k is specified, uses only last k values."""
        # Full series: [1, 3, 5, 7, 9], last 3: [5, 7, 9], median=7
        x = jnp.array([[[1.0], [3.0], [5.0], [7.0], [9.0]]])
        scaled_x, scale = median_scale(x, k=3)

        assert_trees_all_close(scale, jnp.array([[7.0]]))

    def test_uses_absolute_value(self):
        """Scale is always positive even with negative median."""
        x = jnp.array([[[-5.0], [-3.0], [-1.0]]])  # median=-3
        scaled_x, scale = median_scale(x)

        assert_trees_all_close(scale, jnp.array([[3.0]]))  # absolute value

    def test_zero_median_uses_one(self):
        """When median is zero, use 1.0 to avoid division by zero."""
        x = jnp.array([[[-1.0], [0.0], [1.0]]])  # median=0
        scaled_x, scale = median_scale(x)

        assert_trees_all_close(scale, jnp.array([[1.0]]))
        assert_trees_all_close(scaled_x, x)  # unchanged


class TestFixedScale:
    """Tests for fixed_scale."""

    def test_output_shape(self):
        """Output has same shape as input."""
        x = jnp.ones((4, 10, 3))
        scale = jnp.ones((4, 3)) * 2.0
        scaled_x = fixed_scale(x, scale)

        assert_shape(scaled_x, (4, 10, 3))

    def test_divides_by_scale(self):
        """Correctly divides by provided scale."""
        x = jnp.array([[[4.0], [8.0], [12.0]]])
        scale = jnp.array([[4.0]])
        scaled_x = fixed_scale(x, scale)

        assert_trees_all_close(scaled_x, jnp.array([[[1.0], [2.0], [3.0]]]))

    def test_per_feature_scaling(self):
        """Each feature scaled by its own scale value."""
        x = jnp.array([[[4.0, 10.0], [8.0, 20.0]]])  # B=1, T=2, F=2
        scale = jnp.array([[2.0, 5.0]])
        scaled_x = fixed_scale(x, scale)

        assert_trees_all_close(scaled_x, jnp.array([[[2.0, 2.0], [4.0, 4.0]]]))


class TestInverseScale:
    """Tests for inverse_scale."""

    def test_output_shape(self):
        """Output has same shape as input."""
        scaled_x = jnp.ones((4, 10, 3))
        scale = jnp.ones((4, 3)) * 2.0
        x = inverse_scale(scaled_x, scale)

        assert_shape(x, (4, 10, 3))

    def test_multiplies_by_scale(self):
        """Correctly multiplies by scale."""
        scaled_x = jnp.array([[[1.0], [2.0], [3.0]]])
        scale = jnp.array([[4.0]])
        x = inverse_scale(scaled_x, scale)

        assert_trees_all_close(x, jnp.array([[[4.0], [8.0], [12.0]]]))

    def test_roundtrip_last_value(self):
        """last_value_scale followed by inverse_scale recovers original."""
        x_orig = jnp.array([[[1.0, 10.0], [2.0, 20.0], [4.0, 5.0]]])
        scaled_x, scale = last_value_scale(x_orig)
        x_recovered = inverse_scale(scaled_x, scale)

        assert_trees_all_close(x_recovered, x_orig)

    def test_roundtrip_median(self):
        """median_scale followed by inverse_scale recovers original."""
        x_orig = jnp.array([[[1.0], [3.0], [5.0], [7.0], [9.0]]])
        scaled_x, scale = median_scale(x_orig)
        x_recovered = inverse_scale(scaled_x, scale)

        assert_trees_all_close(x_recovered, x_orig)

    def test_roundtrip_fixed(self):
        """fixed_scale followed by inverse_scale recovers original."""
        x_orig = jnp.array([[[4.0], [8.0], [12.0]]])
        scale = jnp.array([[4.0]])
        scaled_x = fixed_scale(x_orig, scale)
        x_recovered = inverse_scale(scaled_x, scale)

        assert_trees_all_close(x_recovered, x_orig)

    def test_2d_input_shape(self):
        """Handles 2D input (B, T) gracefully."""
        scaled_x = jnp.ones((4, 10))  # B=4, T=10, no F dimension
        scale = jnp.ones((4,)) * 2.0
        x = inverse_scale(scaled_x, scale)

        assert_shape(x, (4, 10))

    def test_2d_multiplies_correctly(self):
        """Correctly multiplies 2D input by scale."""
        scaled_x = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])  # (2, 3)
        scale = jnp.array([4.0, 2.0])  # (2,)
        x = inverse_scale(scaled_x, scale)

        expected = jnp.array([[4.0, 8.0, 12.0], [1.0, 2.0, 3.0]])
        assert_trees_all_close(x, expected)

    def test_2d_with_scale_trailing_dim(self):
        """Handles 2D input with scale shape (B, 1)."""
        scaled_x = jnp.array([[1.0, 2.0, 3.0]])  # (1, 3)
        scale = jnp.array([[4.0]])  # (1, 1) - common from 3D scaling
        x = inverse_scale(scaled_x, scale)

        expected = jnp.array([[4.0, 8.0, 12.0]])
        assert_trees_all_close(x, expected)

    def test_2d_roundtrip(self):
        """Scale 3D, squeeze to 2D, inverse_scale back."""
        x_orig = jnp.array([[[1.0], [2.0], [4.0]]])  # (1, 3, 1)
        scaled_x, scale = last_value_scale(x_orig)

        # Squeeze predictions (common pattern)
        scaled_x_2d = scaled_x.squeeze(-1)  # (1, 3)
        scale_1d = scale.squeeze(-1)  # (1,)

        x_recovered_2d = inverse_scale(scaled_x_2d, scale_1d)
        x_recovered = x_recovered_2d[..., None]  # Add back F dim

        assert_trees_all_close(x_recovered, x_orig)


class TestBatchBehavior:
    """Tests for correct batch handling."""

    def test_batch_independence_last_value(self):
        """Each batch element scaled independently."""
        # Two batches with different last values
        x = jnp.array([
            [[1.0], [2.0], [4.0]],   # batch 0, last=4
            [[10.0], [20.0], [5.0]], # batch 1, last=5
        ])
        scaled_x, scale = last_value_scale(x)

        assert_trees_all_close(scale, jnp.array([[4.0], [5.0]]))
        # Batch 0: divided by 4
        assert_trees_all_close(scaled_x[0], jnp.array([[0.25], [0.5], [1.0]]))
        # Batch 1: divided by 5
        assert_trees_all_close(scaled_x[1], jnp.array([[2.0], [4.0], [1.0]]))

    def test_batch_independence_median(self):
        """Each batch element uses its own median."""
        x = jnp.array([
            [[1.0], [3.0], [5.0]],   # batch 0, median=3
            [[10.0], [20.0], [30.0]], # batch 1, median=20
        ])
        scaled_x, scale = median_scale(x)

        assert_trees_all_close(scale, jnp.array([[3.0], [20.0]]))
