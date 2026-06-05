"""Time series dataset container with lazy covariate classification."""

import warnings
from dataclasses import dataclass, field

import jax.numpy as jnp
import pandas as pd

from .batch import TimeSeriesBatch
from .windowing import create_sliding_windows


@dataclass
class TimeSeriesDataset:
    """Container for time series data with targets and covariates.

    Holds aligned arrays for targets and covariates, plus metadata.
    All arrays share the same time dimension (axis 0).

    Attributes:
        targets: Target values, shape (T,) or (T, num_targets).
        known_covariates: Known future covariates, shape (T, num_known) or None.
        observed_covariates: Observed-only covariates, shape (T, num_observed) or None.
        static_covariates: Static features, shape (num_static,) or None.
        timestamps: Optional array of timestamps/indices, shape (T,).
        _unclassified_covariates: Internal field for lazy classification. Set via
            from_dataframe(..., covariate_cols=...). Classified at to_batches() time.
    """

    targets: jnp.ndarray
    known_covariates: jnp.ndarray | None = None
    observed_covariates: jnp.ndarray | None = None
    static_covariates: jnp.ndarray | None = None
    timestamps: jnp.ndarray | None = None
    _unclassified_covariates: jnp.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate shape consistency across all arrays."""
        if self.targets.ndim == 0:
            raise ValueError("targets must be at least 1D.")

        T = len(self.targets)

        if self.known_covariates is not None:
            if len(self.known_covariates) != T:
                raise ValueError(
                    f"known_covariates length {len(self.known_covariates)} "
                    f"does not match targets length {T}"
                )

        if self.observed_covariates is not None:
            if len(self.observed_covariates) != T:
                raise ValueError(
                    f"observed_covariates length {len(self.observed_covariates)} "
                    f"does not match targets length {T}"
                )

        if self._unclassified_covariates is not None:
            if len(self._unclassified_covariates) != T:
                raise ValueError(
                    f"_unclassified_covariates length {len(self._unclassified_covariates)} "
                    f"does not match targets length {T}"
                )

        if self.timestamps is not None:
            if len(self.timestamps) != T:
                raise ValueError(
                    f"timestamps length {len(self.timestamps)} "
                    f"does not match targets length {T}"
                )
            # Check for gaps
            gaps = _detect_gaps(self.timestamps)
            if gaps:
                _warn_gaps(gaps, self.timestamps)

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def num_known_features(self) -> int:
        """Number of known future covariate features."""
        if self.known_covariates is None:
            return 0
        if self.known_covariates.ndim == 1:
            return 1
        return self.known_covariates.shape[-1]

    @property
    def num_observed_features(self) -> int:
        """Number of observed-only covariate features."""
        if self.observed_covariates is None:
            return 0
        if self.observed_covariates.ndim == 1:
            return 1
        return self.observed_covariates.shape[-1]

    @property
    def num_static_features(self) -> int:
        """Number of static features."""
        if self.static_covariates is None:
            return 0
        if self.static_covariates.ndim == 0:
            return 1
        return self.static_covariates.shape[-1]

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target_cols: str | list[str],
        known_cols: list[str] | None = None,
        observed_cols: list[str] | None = None,
        covariate_cols: list[str] | None = None,
        static_cols: list[str] | None = None,
        timestamp_col: str | None = None,
    ) -> "TimeSeriesDataset":
        """Create dataset from a pandas DataFrame.

        Supports two modes for covariate specification:
        1. Explicit: Provide known_cols and/or observed_cols directly.
        2. Lazy: Provide covariate_cols; classification happens at to_batches() time
           based on NaN patterns in the final horizon_len rows.

        Args:
            df: DataFrame with time series data (should be sorted by time).
            target_cols: Column name(s) for target variable(s).
            known_cols: Column names for known future covariates.
            observed_cols: Column names for observed-only covariates.
            covariate_cols: Column names for covariates to classify lazily.
                Mutually exclusive with known_cols/observed_cols.
            static_cols: Column names for static features (uses first row values).
            timestamp_col: Column name for timestamps (enables gap detection).

        Returns:
            TimeSeriesDataset instance.

        Raises:
            ValueError: If both explicit (known_cols/observed_cols) and lazy
                (covariate_cols) modes are specified.
        """
        # Validate mutually exclusive modes
        explicit_mode = known_cols is not None or observed_cols is not None
        lazy_mode = covariate_cols is not None

        if explicit_mode and lazy_mode:
            raise ValueError(
                "Cannot specify both explicit classification (known_cols/observed_cols) "
                "and lazy classification (covariate_cols). Choose one mode."
            )

        # Handle single target column
        if isinstance(target_cols, str):
            target_cols = [target_cols]

        # Extract arrays
        targets = jnp.asarray(df[target_cols].to_numpy())
        if len(target_cols) == 1:
            targets = targets.squeeze(axis=-1)

        known = None
        observed = None
        unclassified = None

        if known_cols is not None:
            known = jnp.asarray(df[known_cols].to_numpy())
        if observed_cols is not None:
            observed = jnp.asarray(df[observed_cols].to_numpy())
        if covariate_cols is not None:
            unclassified = jnp.asarray(df[covariate_cols].to_numpy())

        static = None
        if static_cols is not None:
            static = jnp.asarray(df[static_cols].iloc[0].to_numpy())

        timestamps = None
        if timestamp_col is not None:
            ts_series = df[timestamp_col]
            if pd.api.types.is_datetime64_any_dtype(ts_series):
                timestamps = jnp.asarray(ts_series.astype("int64").to_numpy())
            else:
                timestamps = jnp.asarray(ts_series.to_numpy())

        return cls(
            targets=targets,
            known_covariates=known,
            observed_covariates=observed,
            static_covariates=static,
            timestamps=timestamps,
            _unclassified_covariates=unclassified,
        )

    def __getitem__(self, idx: slice | int) -> "TimeSeriesDataset":
        """Slice the dataset along the time axis.

        Args:
            idx: Integer index or slice.

        Returns:
            New TimeSeriesDataset with sliced data.
        """
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        return TimeSeriesDataset(
            targets=self.targets[idx],
            known_covariates=(
                self.known_covariates[idx] if self.known_covariates is not None else None
            ),
            observed_covariates=(
                self.observed_covariates[idx]
                if self.observed_covariates is not None
                else None
            ),
            static_covariates=self.static_covariates,
            timestamps=self.timestamps[idx] if self.timestamps is not None else None,
            _unclassified_covariates=(
                self._unclassified_covariates[idx]
                if self._unclassified_covariates is not None
                else None
            ),
        )

    def _classify_covariates(
        self, horizon_len: int
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        """Classify unclassified covariates based on NaN in final rows.

        Checks the last horizon_len rows of _unclassified_covariates. Columns
        with any NaN are classified as observed (not available at prediction time).
        Columns fully populated are classified as known.

        Args:
            horizon_len: Number of future timesteps (forecast horizon).

        Returns:
            Tuple of (known_covariates, observed_covariates) arrays.
            Returns (self.known_covariates, self.observed_covariates) if no
            unclassified covariates exist.
        """
        if self._unclassified_covariates is None:
            return self.known_covariates, self.observed_covariates

        # Check last horizon_len rows for NaN
        future_slice = self._unclassified_covariates[-horizon_len:]
        has_nan_per_col = jnp.any(jnp.isnan(future_slice), axis=0)

        known_mask = ~has_nan_per_col
        observed_mask = has_nan_per_col

        # Extract columns
        new_known = None
        new_observed = None

        if jnp.any(known_mask):
            new_known = self._unclassified_covariates[:, known_mask]
        if jnp.any(observed_mask):
            new_observed = self._unclassified_covariates[:, observed_mask]

        # Merge with any existing classified covariates
        if self.known_covariates is not None and new_known is not None:
            new_known = jnp.concatenate([self.known_covariates, new_known], axis=-1)
        elif self.known_covariates is not None:
            new_known = self.known_covariates

        if self.observed_covariates is not None and new_observed is not None:
            new_observed = jnp.concatenate(
                [self.observed_covariates, new_observed], axis=-1
            )
        elif self.observed_covariates is not None:
            new_observed = self.observed_covariates

        return new_known, new_observed

    def to_batches(
        self,
        context_len: int,
        horizon_len: int,
        stride: int = 1,
    ) -> TimeSeriesBatch:
        """Create stacked TimeSeriesBatch from sliding windows.

        If unclassified covariates exist, they are classified first based on
        NaN patterns in the final horizon_len rows.

        Args:
            context_len: Number of historical timesteps (encoder length).
            horizon_len: Number of future timesteps (decoder/forecast length).
            stride: Step size between consecutive windows.

        Returns:
            TimeSeriesBatch with shape (num_windows, time, features).
        """
        # Classify covariates if needed
        if self._unclassified_covariates is not None:
            known, observed = self._classify_covariates(horizon_len)
            dataset = TimeSeriesDataset(
                targets=self.targets,
                known_covariates=known,
                observed_covariates=observed,
                static_covariates=self.static_covariates,
                timestamps=self.timestamps,
                _unclassified_covariates=None,
            )
        else:
            dataset = self

        windows = create_sliding_windows(dataset, context_len, horizon_len, stride)
        return stack_batches(windows)


def stack_batches(batches: list[TimeSeriesBatch]) -> TimeSeriesBatch:
    """Stack list of TimeSeriesBatch into a single batched TimeSeriesBatch.

    Each input batch is expected to have shape (1, T, ...) from create_sliding_windows.
    The output concatenates along the batch dimension to produce (N, T, ...).

    Args:
        batches: List of TimeSeriesBatch objects to stack.

    Returns:
        Single TimeSeriesBatch with batch dimension equal to len(batches).

    Raises:
        ValueError: If batches is empty.
    """
    if not batches:
        raise ValueError("Cannot stack empty list of batches.")

    def concat_field(field_name: str) -> jnp.ndarray | None:
        values = [getattr(b, field_name) for b in batches]
        if values[0] is None:
            return None
        return jnp.concatenate(values, axis=0)

    return TimeSeriesBatch(
        past_targets=concat_field("past_targets"),
        future_targets=concat_field("future_targets"),
        past_observed_covariates=concat_field("past_observed_covariates"),
        past_known_covariates=concat_field("past_known_covariates"),
        future_known_covariates=concat_field("future_known_covariates"),
        static_covariates=concat_field("static_covariates"),
        mask=concat_field("mask"),
    )


def _detect_gaps(timestamps: jnp.ndarray) -> list[tuple[int, int]]:
    """Find gaps in timestamp sequence.

    Detects where the interval between consecutive timestamps differs from
    the median interval (inferred frequency).

    Args:
        timestamps: Array of timestamp values (numeric).

    Returns:
        List of (gap_start_idx, gap_end_idx) tuples indicating where gaps occur.
        Empty list if no gaps detected.
    """
    if len(timestamps) < 2:
        return []

    diffs = jnp.diff(timestamps)
    median_diff = jnp.median(diffs)

    # Use tolerance for floating point comparison
    tolerance = jnp.abs(median_diff) * 0.01
    is_gap = jnp.abs(diffs - median_diff) > tolerance

    gaps = []
    for i, gap in enumerate(is_gap):
        if gap:
            gaps.append((int(i), int(i + 1)))

    return gaps


def _warn_gaps(gaps: list[tuple[int, int]], timestamps: jnp.ndarray) -> None:
    """Emit warning about detected gaps.

    Args:
        gaps: List of (start_idx, end_idx) tuples from _detect_gaps.
        timestamps: Original timestamp array for displaying gap boundaries.
    """
    gap_descriptions = []
    for start_idx, end_idx in gaps[:5]:
        ts_start = timestamps[start_idx]
        ts_end = timestamps[end_idx]
        gap_descriptions.append(
            f"  - Index {start_idx}-{end_idx} (timestamp {ts_start} to {ts_end})"
        )

    more = ""
    if len(gaps) > 5:
        more = f"\n  ... and {len(gaps) - 5} more gaps"

    warnings.warn(
        f"Detected {len(gaps)} gap(s) in time series:\n"
        + "\n".join(gap_descriptions)
        + more,
        UserWarning,
        stacklevel=4,
    )