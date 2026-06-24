"""Tests for data splitting utilities."""

import jax.numpy as jnp
import pytest

from jax_prism.data.dataset import TimeSeriesDataset
from jax_prism.training.splitting import (
    expanding_window_cv,
    rolling_window_cv,
    temporal_split,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataset():
    """Create a simple TimeSeriesDataset with 1000 timesteps."""
    return TimeSeriesDataset(targets=jnp.arange(1000, dtype=jnp.float32))


@pytest.fixture
def small_dataset():
    """Create a small TimeSeriesDataset with 100 timesteps."""
    return TimeSeriesDataset(targets=jnp.arange(100, dtype=jnp.float32))


@pytest.fixture
def dataset_with_covariates():
    """Create a TimeSeriesDataset with covariates."""
    T = 500
    return TimeSeriesDataset(
        targets=jnp.arange(T, dtype=jnp.float32),
        known_covariates=jnp.ones((T, 3)),
        observed_covariates=jnp.ones((T, 2)),
    )


# =============================================================================
# temporal_split Tests
# =============================================================================


class TestTemporalSplit:
    """Tests for temporal_split function."""

    def test_basic_split(self, sample_dataset):
        """Basic split returns three datasets."""
        train, val, test = temporal_split(sample_dataset)

        assert isinstance(train, TimeSeriesDataset)
        assert isinstance(val, TimeSeriesDataset)
        assert isinstance(test, TimeSeriesDataset)

    def test_split_lengths(self, sample_dataset):
        """Split respects train_frac and val_frac."""
        train, val, test = temporal_split(
            sample_dataset, train_frac=0.7, val_frac=0.15
        )

        # With 1000 points, no gap: train=700, val=150, test=150
        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_temporal_order(self, sample_dataset):
        """Train data comes before val, val before test."""
        train, val, test = temporal_split(sample_dataset)

        # Check that max of train < min of val < max of val < min of test
        assert train.targets[-1] < val.targets[0]
        assert val.targets[-1] < test.targets[0]

    def test_with_gap(self, sample_dataset):
        """Gap creates space between splits."""
        gap = 24
        train, val, test = temporal_split(sample_dataset, gap=gap)

        # With gap, there should be 24 timesteps between train and val
        train_end_value = train.targets[-1]
        val_start_value = val.targets[0]

        # The gap means val starts gap steps after train ends
        assert val_start_value - train_end_value == gap + 1

    def test_gap_reduces_usable_length(self, sample_dataset):
        """Gap reduces the usable data length."""
        gap = 50
        train_no_gap, val_no_gap, test_no_gap = temporal_split(
            sample_dataset, gap=0
        )
        train_gap, val_gap, test_gap = temporal_split(
            sample_dataset, gap=gap
        )

        total_no_gap = len(train_no_gap) + len(val_no_gap) + len(test_no_gap)
        total_gap = len(train_gap) + len(val_gap) + len(test_gap)

        # Gap of 50 applied twice = 100 fewer usable points
        assert total_no_gap - total_gap == 2 * gap

    def test_preserves_covariates(self, dataset_with_covariates):
        """Split preserves covariate structure."""
        train, val, test = temporal_split(dataset_with_covariates)

        assert train.known_covariates is not None
        assert train.observed_covariates is not None
        assert train.known_covariates.shape[1] == 3
        assert train.observed_covariates.shape[1] == 2

    def test_invalid_fractions_negative(self, sample_dataset):
        """Raises ValueError for negative fractions."""
        with pytest.raises(ValueError):
            temporal_split(sample_dataset, train_frac=-0.1)

    def test_invalid_fractions_sum_exceeds_one(self, sample_dataset):
        """Raises ValueError when fractions exceed 1."""
        with pytest.raises(ValueError):
            temporal_split(sample_dataset, train_frac=0.7, val_frac=0.4)

    def test_dataset_too_short_for_gap(self, small_dataset):
        """Raises ValueError when dataset shorter than 2*gap."""
        with pytest.raises(ValueError):
            temporal_split(small_dataset, gap=60)  # 2*60 = 120 > 100

    def test_custom_fractions(self, sample_dataset):
        """Custom fractions work correctly."""
        train, val, test = temporal_split(
            sample_dataset, train_frac=0.5, val_frac=0.3
        )

        # 1000 points: train=500, val=300, test=200
        assert len(train) == 500
        assert len(val) == 300
        assert len(test) == 200

    def test_no_val_split(self, sample_dataset):
        """val_frac=0 creates empty validation set."""
        train, val, test = temporal_split(
            sample_dataset, train_frac=0.8, val_frac=0.0
        )

        assert len(train) == 800
        assert len(val) == 0
        assert len(test) == 200


# =============================================================================
# expanding_window_cv Tests
# =============================================================================


class TestExpandingWindowCV:
    """Tests for expanding_window_cv function."""

    def test_yields_correct_number_of_folds(self, sample_dataset):
        """Yields exactly n_folds."""
        n_folds = 5
        folds = list(expanding_window_cv(sample_dataset, n_folds=n_folds))

        assert len(folds) == n_folds

    def test_each_fold_is_tuple(self, sample_dataset):
        """Each fold is a (train, val) tuple."""
        for train, val in expanding_window_cv(sample_dataset, n_folds=3):
            assert isinstance(train, TimeSeriesDataset)
            assert isinstance(val, TimeSeriesDataset)

    def test_train_grows_each_fold(self, sample_dataset):
        """Train size increases with each fold."""
        train_sizes = []
        for train, val in expanding_window_cv(sample_dataset, n_folds=5):
            train_sizes.append(len(train))

        # Each train should be larger than the previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_temporal_order_preserved(self, sample_dataset):
        """Train comes before val in each fold."""
        for train, val in expanding_window_cv(sample_dataset, n_folds=5):
            assert train.targets[-1] < val.targets[0]

    def test_with_gap(self, sample_dataset):
        """Gap is respected between train and val."""
        gap = 24
        for train, val in expanding_window_cv(sample_dataset, n_folds=3, gap=gap):
            train_end = train.targets[-1]
            val_start = val.targets[0]
            assert val_start - train_end == gap + 1

    def test_val_size_consistent(self, sample_dataset):
        """Val size is approximately consistent across folds."""
        val_sizes = []
        for train, val in expanding_window_cv(
            sample_dataset, n_folds=5, val_frac=0.1
        ):
            val_sizes.append(len(val))

        # All val sizes should be similar (last might be truncated)
        expected_val_size = val_sizes[0]
        for size in val_sizes[:-1]:  # exclude last which might be smaller
            assert size == expected_val_size

    def test_train_starts_at_zero(self, sample_dataset):
        """Train always starts at index 0 (expanding window)."""
        for train, val in expanding_window_cv(sample_dataset, n_folds=5):
            assert train.targets[0] == 0

    def test_invalid_fractions(self, sample_dataset):
        """Raises ValueError for invalid fractions."""
        with pytest.raises(ValueError):
            list(expanding_window_cv(sample_dataset, min_train_frac=-0.1))

    def test_dataset_too_short(self, small_dataset):
        """Raises ValueError when dataset too short for gap."""
        with pytest.raises(ValueError):
            list(expanding_window_cv(small_dataset, gap=60))


# =============================================================================
# rolling_window_cv Tests
# =============================================================================


class TestRollingWindowCV:
    """Tests for rolling_window_cv function."""

    def test_yields_correct_number_of_folds(self, sample_dataset):
        """Yields exactly n_folds."""
        n_folds = 5
        folds = list(rolling_window_cv(sample_dataset, n_folds=n_folds))

        assert len(folds) == n_folds

    def test_each_fold_is_tuple(self, sample_dataset):
        """Each fold is a (train, val) tuple."""
        for train, val in rolling_window_cv(sample_dataset, n_folds=3):
            assert isinstance(train, TimeSeriesDataset)
            assert isinstance(val, TimeSeriesDataset)

    def test_train_size_constant(self, sample_dataset):
        """Train size is constant across folds (rolling window)."""
        train_sizes = []
        for train, val in rolling_window_cv(sample_dataset, n_folds=5):
            train_sizes.append(len(train))

        # All train sizes should be equal
        assert all(size == train_sizes[0] for size in train_sizes)

    def test_train_slides_forward(self, sample_dataset):
        """Train window slides forward each fold."""
        train_starts = []
        for train, val in rolling_window_cv(sample_dataset, n_folds=5):
            train_starts.append(train.targets[0])

        # Each train should start later than the previous
        for i in range(1, len(train_starts)):
            assert train_starts[i] > train_starts[i - 1]

    def test_temporal_order_preserved(self, sample_dataset):
        """Train comes before val in each fold."""
        for train, val in rolling_window_cv(sample_dataset, n_folds=5):
            assert train.targets[-1] < val.targets[0]

    def test_with_gap(self, sample_dataset):
        """Gap is respected between train and val."""
        gap = 24
        for train, val in rolling_window_cv(sample_dataset, n_folds=3, gap=gap):
            train_end = train.targets[-1]
            val_start = val.targets[0]
            assert val_start - train_end == gap + 1

    def test_val_size_consistent(self, sample_dataset):
        """Val size is approximately consistent across folds."""
        val_sizes = []
        for train, val in rolling_window_cv(
            sample_dataset, n_folds=5, val_frac=0.1
        ):
            val_sizes.append(len(val))

        # All val sizes should be similar (last might be truncated)
        expected_val_size = val_sizes[0]
        for size in val_sizes[:-1]:
            assert size == expected_val_size

    def test_different_from_expanding(self, sample_dataset):
        """Rolling differs from expanding - train size constant vs growing."""
        expanding_train_sizes = [
            len(train) for train, _ in expanding_window_cv(sample_dataset, n_folds=5)
        ]
        rolling_train_sizes = [
            len(train) for train, _ in rolling_window_cv(sample_dataset, n_folds=5)
        ]

        # Expanding: train grows
        assert expanding_train_sizes[-1] > expanding_train_sizes[0]

        # Rolling: train constant
        assert rolling_train_sizes[-1] == rolling_train_sizes[0]

    def test_invalid_fractions(self, sample_dataset):
        """Raises ValueError for invalid fractions."""
        with pytest.raises(ValueError):
            list(rolling_window_cv(sample_dataset, train_frac=-0.1))

    def test_dataset_too_short(self, small_dataset):
        """Raises ValueError when dataset too short for gap."""
        with pytest.raises(ValueError):
            list(rolling_window_cv(small_dataset, gap=60))


# =============================================================================
# Integration Tests
# =============================================================================


class TestSplittingIntegration:
    """Integration tests for splitting utilities."""

    def test_temporal_split_then_to_batches(self, sample_dataset):
        """Split datasets can be converted to batches."""
        train, val, test = temporal_split(sample_dataset)

        # This should not raise
        train_batch = train.to_batches(context_len=10, horizon_len=5)
        val_batch = val.to_batches(context_len=10, horizon_len=5)

        assert train_batch.past_targets is not None
        assert val_batch.past_targets is not None

    def test_cv_fold_to_batches(self, sample_dataset):
        """CV folds can be converted to batches."""
        for train, val in expanding_window_cv(sample_dataset, n_folds=3):
            if len(train) >= 15 and len(val) >= 15:  # need enough data
                train_batch = train.to_batches(context_len=10, horizon_len=5)
                val_batch = val.to_batches(context_len=10, horizon_len=5)

                assert train_batch.past_targets is not None
                assert val_batch.past_targets is not None