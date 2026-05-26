import pandas as pd
import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass

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
    """
    targets: jnp.ndarray
    known_covariates: jnp.ndarray | None = None
    observed_covariates: jnp.ndarray | None = None
    static_covariates: jnp.ndarray | None = None
    timestamps: jnp.ndarray | None = None
    
    def __post_init__(self):
        """Validates shapes are consistent."""
        if len(self.targets.shape) == 0:
            raise ValueError("The targets must at least be 1D.")
        
        T = len(self.targets)
        
        if self.known_covariates is not None:
            if len(self.known_covariates) != T:
                raise ValueError(f"known_covariates length {len(self.known_covariates)} does not match targets length {T}")
            
        if self.observed_covariates is not None:
            if len(self.observed_covariates) != T:
                raise ValueError(f"observed_covariates length {len(self.observed_covariates)} does not match targets length {T}")

        
    def __len__(self) -> int:
        return len(self.targets)
    
    @property
    def num_known_features(self) -> int:
        if self.known_covariates is None:
            return 0
        
        if self.known_covariates.ndim == 1:
            return 1
        
        return self.known_covariates.shape[-1]
    
    @property
    def num_observed_features(self) -> int:
        if self.observed_covariates is None:
            return 0
        
        if self.observed_covariates.ndim == 1:
            return 1
        
        return self.observed_covariates.shape[-1]
    
    @property
    def num_static_features(self) -> int:
        if self.static_covariates is None:
            return 0
        
        return self.static_covariates.shape[-1]
    
    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        target_col: str,
        known_cols: list[str] | None = None,
        observed_cols: list[str] | None = None,
        static_cols: list[str] | None = None,
        timestamp_col: str | None = None,
    ) -> "TimeSeriesDataset":
        """Create dataset from a Pandas DataFrame.

        Args:
            df: DataFrame with time series data (sorted by time).
            target_col: Column name for target variable
            known_cols: Column names for known future covariates.
            observed_cols: Column names for observed-only covariates.
            static_cols: Column names for static features (takes first row).
            timestamp_col: Column name for timestamps.

        Returns:
            TimeSeriesDataset instance.
        """
        
        return TimeSeriesDataset(
            targets=jnp.asarray(df[target_col].to_numpy()),
            known_covariates=jnp.asarray(df[known_cols].to_numpy()) if known_cols is not None else None,
            observed_covariates=jnp.asarray(df[observed_cols].to_numpy()) if observed_cols is not None else None,
            static_covariates=jnp.asarray(df[static_cols].iloc[0].to_numpy()) if static_cols is not None else None,
            timestamps=jnp.asarray(df[timestamp_col].to_numpy()) if timestamp_col is not None else None,        
        )
    
    def __getitem__(self, idx: slice | int) -> "TimeSeriesDataset":
        """Slice the dataset along the time axis.

        Args:
            idx: Integer index or slice index

        Returns:
            New TimeSeriesDataset with sliced data.
        """
        
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
            
        return TimeSeriesDataset(
            targets=self.targets[idx],
            known_covariates=self.known_covariates[idx] if self.known_covariates is not None else None,
            observed_covariates=self.observed_covariates[idx] if self.observed_covariates is not None else None,
            static_covariates=self.static_covariates,
            timestamps=self.timestamps[idx] if self.timestamps is not None else None,
        )