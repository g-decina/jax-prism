"""Tests for data/dataset.py and data/windowing.py."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from chex import assert_shape, assert_trees_all_close

from jax_prism.data.dataset import (
    TimeSeriesDataset,
    stack_batches,
    _detect_gaps,
)
from jax_prism.data.windowing import create_sliding_windows


class TestTimeSeriesDataset:
    """Tests for TimeSeriesDataset."""

    def test_basic_creation(self):
        """Can create with just targets."""
        targets = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ds = TimeSeriesDataset(targets=targets)

        assert len(ds) == 5
        assert ds.num_known_features == 0
        assert ds.num_observed_features == 0
        assert ds.num_static_features == 0

    def test_creation_with_covariates(self):
        """Can create with all covariate types."""
        T = 10
        targets = jnp.ones((T,))
        known = jnp.ones((T, 3))
        observed = jnp.ones((T, 2))
        static = jnp.ones((4,))

        ds = TimeSeriesDataset(
            targets=targets,
            known_covariates=known,
            observed_covariates=observed,
            static_covariates=static,
        )

        assert len(ds) == T
        assert ds.num_known_features == 3
        assert ds.num_observed_features == 2
        assert ds.num_static_features == 4

    def test_1d_covariate_counts_as_one(self):
        """1D covariates report 1 feature, not 0."""
        T = 10
        ds = TimeSeriesDataset(
            targets=jnp.ones((T,)),
            known_covariates=jnp.ones((T,)),  # 1D
            observed_covariates=jnp.ones((T,)),  # 1D
        )

        assert ds.num_known_features == 1
        assert ds.num_observed_features == 1

    def test_validation_rejects_0d_targets(self):
        """Scalar targets raise ValueError."""
        with pytest.raises(ValueError, match="at least 1D"):
            TimeSeriesDataset(targets=jnp.array(5.0))

    def test_validation_rejects_length_mismatch_known(self):
        """known_covariates with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="known_covariates length"):
            TimeSeriesDataset(
                targets=jnp.ones((10,)),
                known_covariates=jnp.ones((8, 2)),  # Wrong length
            )

    def test_validation_rejects_length_mismatch_observed(self):
        """observed_covariates with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="observed_covariates length"):
            TimeSeriesDataset(
                targets=jnp.ones((10,)),
                observed_covariates=jnp.ones((12, 2)),  # Wrong length
            )


class TestTimeSeriesDatasetSlicing:
    """Tests for __getitem__ slicing."""

    @pytest.fixture
    def dataset(self):
        """Create a dataset with all field types."""
        T = 20
        return TimeSeriesDataset(
            targets=jnp.arange(T, dtype=jnp.float32),
            known_covariates=jnp.arange(T * 2, dtype=jnp.float32).reshape(T, 2),
            observed_covariates=jnp.arange(T * 3, dtype=jnp.float32).reshape(T, 3),
            static_covariates=jnp.array([100.0, 200.0]),
            timestamps=jnp.arange(T),
        )

    def test_slice_basic(self, dataset):
        """Slicing returns new dataset with correct length."""
        sliced = dataset[5:10]

        assert len(sliced) == 5
        assert_trees_all_close(sliced.targets, jnp.array([5.0, 6.0, 7.0, 8.0, 9.0]))

    def test_slice_preserves_covariates(self, dataset):
        """Slicing preserves all covariate types."""
        sliced = dataset[0:3]

        assert sliced.known_covariates is not None
        assert sliced.observed_covariates is not None
        assert_shape(sliced.known_covariates, (3, 2))
        assert_shape(sliced.observed_covariates, (3, 3))

    def test_slice_preserves_static(self, dataset):
        """Static covariates are not sliced (time-invariant)."""
        sliced = dataset[5:10]

        assert_trees_all_close(sliced.static_covariates, dataset.static_covariates)

    def test_slice_preserves_timestamps(self, dataset):
        """Timestamps are sliced correctly."""
        sliced = dataset[5:10]

        assert_trees_all_close(sliced.timestamps, jnp.array([5, 6, 7, 8, 9]))

    def test_integer_index(self, dataset):
        """Integer index returns single-element dataset."""
        sliced = dataset[7]

        assert len(sliced) == 1
        assert_trees_all_close(sliced.targets, jnp.array([7.0]))

    def test_slice_with_none_covariates(self):
        """Slicing works when covariates are None."""
        ds = TimeSeriesDataset(targets=jnp.arange(10.0))
        sliced = ds[2:5]

        assert len(sliced) == 3
        assert sliced.known_covariates is None
        assert sliced.observed_covariates is None


class TestTimeSeriesDatasetFromDataframe:
    """Tests for from_dataframe class method."""

    def test_basic_conversion(self):
        """Converts DataFrame to dataset correctly."""
        df = pd.DataFrame({
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "known_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "known_b": [100.0, 200.0, 300.0, 400.0, 500.0],
        })

        ds = TimeSeriesDataset.from_dataframe(
            df,
            target_cols="target",
            known_cols=["known_a", "known_b"],
        )

        assert len(ds) == 5
        assert ds.num_known_features == 2
        assert_trees_all_close(ds.targets, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_static_takes_first_row(self):
        """Static covariates take first row only."""
        df = pd.DataFrame({
            "target": [1.0, 2.0, 3.0],
            "static_id": [42, 42, 42],  # Same value repeated
        })

        ds = TimeSeriesDataset.from_dataframe(
            df,
            target_cols="target",
            static_cols=["static_id"],
        )

        assert ds.num_static_features == 1
        assert_trees_all_close(ds.static_covariates, jnp.array([42.0]))

    def test_returns_jax_arrays(self):
        """Output arrays are JAX arrays, not NumPy."""
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
        ds = TimeSeriesDataset.from_dataframe(df, target_cols="target")

        # jax.Array has device attribute
        assert hasattr(ds.targets, "device") or isinstance(ds.targets, jnp.ndarray)

    def test_none_columns_stay_none(self):
        """Columns not specified remain None."""
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0]})
        ds = TimeSeriesDataset.from_dataframe(df, target_cols="target")

        assert ds.known_covariates is None
        assert ds.observed_covariates is None
        assert ds.static_covariates is None
        assert ds.timestamps is None


class TestCreateSlidingWindows:
    """Tests for create_sliding_windows."""

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset for windowing tests."""
        T = 20
        return TimeSeriesDataset(
            targets=jnp.arange(T, dtype=jnp.float32),
            known_covariates=jnp.arange(T, dtype=jnp.float32)[:, None] * 10,
        )

    def test_basic_windowing(self, dataset):
        """Creates correct number of windows."""
        windows = create_sliding_windows(
            dataset, context_len=5, horizon_len=3, stride=1
        )

        # T=20, window_size=8, stride=1 → (20-8)//1 + 1 = 13 windows
        assert len(windows) == 13

    def test_window_shapes(self, dataset):
        """Each window has correct shapes."""
        windows = create_sliding_windows(
            dataset, context_len=5, horizon_len=3, stride=1
        )

        batch = windows[0]
        assert_shape(batch.past_targets, (1, 5))
        assert_shape(batch.future_targets, (1, 3))
        assert_shape(batch.past_known_covariates, (1, 5, 1))
        assert_shape(batch.future_known_covariates, (1, 3, 1))

    def test_window_content_first(self, dataset):
        """First window contains correct data."""
        windows = create_sliding_windows(
            dataset, context_len=5, horizon_len=3, stride=1
        )

        batch = windows[0]
        # Past: [0, 1, 2, 3, 4]
        assert_trees_all_close(
            batch.past_targets[0], jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        )
        # Future: [5, 6, 7]
        assert_trees_all_close(batch.future_targets[0], jnp.array([5.0, 6.0, 7.0]))

    def test_window_content_with_stride(self, dataset):
        """Stride correctly shifts window starts."""
        windows = create_sliding_windows(
            dataset, context_len=5, horizon_len=3, stride=4
        )

        # Window 0 starts at 0
        assert_trees_all_close(
            windows[0].past_targets[0], jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        )
        # Window 1 starts at 4
        assert_trees_all_close(
            windows[1].past_targets[0], jnp.array([4.0, 5.0, 6.0, 7.0, 8.0])
        )

    def test_stride_affects_count(self, dataset):
        """Larger stride produces fewer windows."""
        windows_s1 = create_sliding_windows(
            dataset, context_len=5, horizon_len=3, stride=1
        )
        windows_s4 = create_sliding_windows(
            dataset, context_len=5, horizon_len=3, stride=4
        )

        assert len(windows_s4) < len(windows_s1)
        # (20-8)//4 + 1 = 4 windows
        assert len(windows_s4) == 4

    def test_too_short_raises(self):
        """Dataset shorter than window size raises ValueError."""
        short_ds = TimeSeriesDataset(targets=jnp.ones((5,)))

        with pytest.raises(ValueError, match="less than window size"):
            create_sliding_windows(short_ds, context_len=5, horizon_len=3)

    def test_exact_fit(self):
        """Dataset exactly matching window size produces one window."""
        ds = TimeSeriesDataset(targets=jnp.arange(8.0))
        windows = create_sliding_windows(ds, context_len=5, horizon_len=3)

        assert len(windows) == 1

    def test_none_covariates_stay_none(self):
        """Windows have None for missing covariate types."""
        ds = TimeSeriesDataset(targets=jnp.arange(20.0))
        windows = create_sliding_windows(ds, context_len=5, horizon_len=3)

        batch = windows[0]
        assert batch.past_known_covariates is None
        assert batch.future_known_covariates is None
        assert batch.past_observed_covariates is None
        assert batch.static_covariates is None

    def test_static_covariates_preserved(self):
        """Static covariates appear in every window."""
        ds = TimeSeriesDataset(
            targets=jnp.arange(20.0),
            static_covariates=jnp.array([42.0, 99.0]),
        )
        windows = create_sliding_windows(ds, context_len=5, horizon_len=3)

        for batch in windows:
            assert batch.static_covariates is not None
            assert_shape(batch.static_covariates, (1, 2))
            assert_trees_all_close(
                batch.static_covariates[0], jnp.array([42.0, 99.0])
            )


class TestToBatches:
    """Tests for TimeSeriesDataset.to_batches() method."""

    def test_basic_to_batches(self):
        """to_batches returns stacked TimeSeriesBatch."""
        ds = TimeSeriesDataset(targets=jnp.arange(20.0))
        batch = ds.to_batches(context_len=5, horizon_len=3, stride=1)

        # Should have 13 windows stacked
        assert_shape(batch.past_targets, (13, 5))
        assert_shape(batch.future_targets, (13, 3))

    def test_to_batches_with_covariates(self):
        """to_batches preserves covariates in stacked batch."""
        T = 20
        ds = TimeSeriesDataset(
            targets=jnp.arange(T, dtype=jnp.float32),
            known_covariates=jnp.ones((T, 2)),
            observed_covariates=jnp.ones((T, 3)),
        )
        batch = ds.to_batches(context_len=5, horizon_len=3)

        assert_shape(batch.past_known_covariates, (13, 5, 2))
        assert_shape(batch.future_known_covariates, (13, 3, 2))
        assert_shape(batch.past_observed_covariates, (13, 5, 3))

    def test_to_batches_with_stride(self):
        """to_batches respects stride parameter."""
        ds = TimeSeriesDataset(targets=jnp.arange(20.0))
        batch = ds.to_batches(context_len=5, horizon_len=3, stride=4)

        # (20-8)//4 + 1 = 4 windows
        assert_shape(batch.past_targets, (4, 5))

    def test_to_batches_content(self):
        """to_batches produces correct window content."""
        ds = TimeSeriesDataset(targets=jnp.arange(10.0))
        batch = ds.to_batches(context_len=3, horizon_len=2, stride=1)

        # First window: past=[0,1,2], future=[3,4]
        assert_trees_all_close(batch.past_targets[0], jnp.array([0.0, 1.0, 2.0]))
        assert_trees_all_close(batch.future_targets[0], jnp.array([3.0, 4.0]))

        # Second window: past=[1,2,3], future=[4,5]
        assert_trees_all_close(batch.past_targets[1], jnp.array([1.0, 2.0, 3.0]))
        assert_trees_all_close(batch.future_targets[1], jnp.array([4.0, 5.0]))


class TestStackBatches:
    """Tests for stack_batches utility."""

    def test_stack_batches_basic(self):
        """stack_batches combines list into single batch."""
        ds = TimeSeriesDataset(targets=jnp.arange(20.0))
        windows = create_sliding_windows(ds, context_len=5, horizon_len=3)

        stacked = stack_batches(windows)

        assert_shape(stacked.past_targets, (13, 5))
        assert_shape(stacked.future_targets, (13, 3))

    def test_stack_batches_empty_raises(self):
        """stack_batches raises on empty list."""
        with pytest.raises(ValueError, match="empty"):
            stack_batches([])

    def test_stack_batches_preserves_none(self):
        """stack_batches keeps None fields as None."""
        ds = TimeSeriesDataset(targets=jnp.arange(20.0))
        windows = create_sliding_windows(ds, context_len=5, horizon_len=3)

        stacked = stack_batches(windows)

        assert stacked.past_known_covariates is None
        assert stacked.past_observed_covariates is None
        assert stacked.static_covariates is None


class TestLazyCovariateClassification:
    """Tests for lazy covariate classification via covariate_cols."""

    def test_covariate_cols_stores_unclassified(self):
        """covariate_cols populates _unclassified_covariates."""
        df = pd.DataFrame({
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cov_a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cov_b": [100.0, 200.0, 300.0, 400.0, 500.0],
        })

        ds = TimeSeriesDataset.from_dataframe(
            df,
            target_cols="target",
            covariate_cols=["cov_a", "cov_b"],
        )

        assert ds._unclassified_covariates is not None
        assert_shape(ds._unclassified_covariates, (5, 2))
        assert ds.known_covariates is None
        assert ds.observed_covariates is None

    def test_classification_by_nan_in_future(self):
        """Covariates with NaN in last rows are classified as observed."""
        # Simulate: day_of_week is known (no NaN), temperature is observed (NaN in future)
        T = 10
        horizon = 3

        day_of_week = jnp.arange(T, dtype=jnp.float32)  # All populated
        temperature = jnp.arange(T, dtype=jnp.float32)
        # Set last horizon_len values to NaN for temperature
        temperature = temperature.at[-horizon:].set(jnp.nan)

        covariates = jnp.stack([day_of_week, temperature], axis=-1)

        ds = TimeSeriesDataset(
            targets=jnp.arange(T, dtype=jnp.float32),
            _unclassified_covariates=covariates,
        )

        known, observed = ds._classify_covariates(horizon_len=horizon)

        # day_of_week should be known (column 0)
        assert known is not None
        assert_shape(known, (T, 1))

        # temperature should be observed (column 1)
        assert observed is not None
        assert_shape(observed, (T, 1))

    def test_to_batches_classifies_covariates(self):
        """to_batches performs classification before windowing."""
        T = 15
        horizon = 3

        # Create DataFrame with known and observed pattern
        df = pd.DataFrame({
            "target": np.arange(T, dtype=np.float32),
            "day_of_week": np.arange(T, dtype=np.float32),  # Known
            "temperature": np.concatenate([
                np.arange(T - horizon, dtype=np.float32),
                np.full(horizon, np.nan),  # NaN in future
            ]),
        })

        ds = TimeSeriesDataset.from_dataframe(
            df,
            target_cols="target",
            covariate_cols=["day_of_week", "temperature"],
        )

        batch = ds.to_batches(context_len=5, horizon_len=horizon)

        # Should have future_known_covariates (day_of_week)
        assert batch.future_known_covariates is not None

        # Should have past_observed_covariates (temperature)
        assert batch.past_observed_covariates is not None

    def test_mutual_exclusivity_error(self):
        """Cannot specify both explicit and lazy modes."""
        df = pd.DataFrame({
            "target": [1.0, 2.0, 3.0],
            "cov": [10.0, 20.0, 30.0],
        })

        with pytest.raises(ValueError, match="Cannot specify both"):
            TimeSeriesDataset.from_dataframe(
                df,
                target_cols="target",
                known_cols=["cov"],
                covariate_cols=["cov"],
            )

    def test_all_known_classification(self):
        """When no NaN in future, all covariates classified as known."""
        T = 10
        covariates = jnp.ones((T, 3))  # All populated

        ds = TimeSeriesDataset(
            targets=jnp.ones(T),
            _unclassified_covariates=covariates,
        )

        known, observed = ds._classify_covariates(horizon_len=3)

        assert known is not None
        assert_shape(known, (T, 3))
        assert observed is None

    def test_all_observed_classification(self):
        """When all have NaN in future, all classified as observed."""
        T = 10
        horizon = 3
        covariates = jnp.ones((T, 2))
        covariates = covariates.at[-horizon:, :].set(jnp.nan)

        ds = TimeSeriesDataset(
            targets=jnp.ones(T),
            _unclassified_covariates=covariates,
        )

        known, observed = ds._classify_covariates(horizon_len=horizon)

        assert known is None
        assert observed is not None
        assert_shape(observed, (T, 2))


class TestGapDetection:
    """Tests for timestamp gap detection."""

    def test_no_gaps_detected(self):
        """Regular timestamps produce no gaps."""
        timestamps = jnp.arange(10)
        gaps = _detect_gaps(timestamps)

        assert gaps == []

    def test_single_gap_detected(self):
        """Single missing timestamp is detected."""
        # 0, 1, 2, 4, 5, 6 (3 is missing)
        timestamps = jnp.array([0, 1, 2, 4, 5, 6])
        gaps = _detect_gaps(timestamps)

        assert len(gaps) == 1
        assert gaps[0] == (2, 3)  # Gap between index 2 and 3

    def test_multiple_gaps_detected(self):
        """Multiple gaps are all detected."""
        # 0, 1, 3, 4, 6, 7 (2 and 5 missing)
        timestamps = jnp.array([0, 1, 3, 4, 6, 7])
        gaps = _detect_gaps(timestamps)

        assert len(gaps) == 2
        assert (1, 2) in gaps  # Gap after index 1
        assert (3, 4) in gaps  # Gap after index 3

    def test_short_sequence_no_gaps(self):
        """Sequence with < 2 elements returns empty."""
        assert _detect_gaps(jnp.array([1])) == []
        assert _detect_gaps(jnp.array([])) == []

    def test_gap_warning_emitted(self):
        """Creating dataset with gaps emits warning."""
        # Timestamps with gap
        timestamps = jnp.array([0, 1, 2, 5, 6, 7])  # 3, 4 missing

        with pytest.warns(UserWarning, match="gap"):
            TimeSeriesDataset(
                targets=jnp.ones(6),
                timestamps=timestamps,
            )
