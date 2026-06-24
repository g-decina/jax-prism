from typing import Iterator

from jax_prism.data.dataset import TimeSeriesDataset


def temporal_split(
    dataset: TimeSeriesDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    gap: int = 0
) -> tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Split a TimeSeriesDataset temporally into train/val/test.

    Maintains temporal order: train < val < test. Applies a gap between
    splits to prevent leakage from forecast horizons.

    Timeline:
        [0 ─── train_end][gap][val_start ─── val_end][gap][test_start ─── T]

    Args:
        dataset: TimeSeriesDataset to split.
        train_frac: Fraction of *usable* data for training (after gaps).
        val_frac: Fraction of *usable* data for validation.
            test_frac is implicitly 1 - train_frac - val_frac.
        gap: Timesteps to skip between splits. Set to horizon_len to
            prevent forecast horizon overlap.

    Returns:
        Tuple of (train, val, test) TimeSeriesDataset objects.

    Raises:
        ValueError: If fractions don't sum to ≤ 1, or if dataset too short.

    Example:
        >>> dataset = TimeSeriesDataset(targets=jnp.arange(1000))
        >>> train, val, test = temporal_split(dataset, gap=24)
        >>> len(train), len(val), len(test)
        (665, 142, 143)  # approximate, depends on rounding
    """
    
    # === 1. Validate inputs ===
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError(
            "train_frac and val_frac must be comprised between 0 and 1"
        )
    
    if len(dataset) < 2 * gap:
        raise ValueError(
            f"len(dataset) must be at least {2 * gap}; was {len(dataset)}"
        )
    
    # === 2. Compute effective length ===
    T = len(dataset)
    total_gap = 2 * gap
    usable_len = T - total_gap
    
    # === 3. Compute split points ===
    train_len = int(train_frac * usable_len)
    val_len = int(val_frac * usable_len)
    
    # === 4. Compute slice boundaries ===
    train_end = train_len
    val_start = train_end + gap
    val_end = val_start + val_len
    test_start = val_end + gap
    
    # === 5. Return sliced datasets ===
    return (
        dataset[:train_end],
        dataset[val_start:val_end],
        dataset[test_start:],
    )


def expanding_window_cv(
    dataset: TimeSeriesDataset,
    n_folds: int = 5,
    val_frac: float = 0.1,
    min_train_frac: float = 0.3,
    gap: int = 0,
) -> Iterator[tuple[TimeSeriesDataset, TimeSeriesDataset]]:
    """Generate expanding window cross-validation folds.

    Train window grows with each fold while validation slides forward.
    Respects temporal order: train always precedes validation.

    Fold structure:
        Fold 0: [TRAIN════]        [VAL]
        Fold 1: [TRAIN════════]         [VAL]
        Fold 2: [TRAIN════════════]          [VAL]

    Args:
        dataset: TimeSeriesDataset to split.
        n_folds: Number of CV folds.
        val_frac: Validation size as fraction of usable data.
        min_train_frac: Minimum train size (first fold) as fraction.
        gap: Timesteps between train and val to prevent leakage.
            Set to horizon_len to prevent forecast horizon overlap.

    Yields:
        (train, val) TimeSeriesDataset tuples for each fold.

    Raises:
        ValueError: If fractions invalid or dataset too short.

    Example:
        >>> for fold_idx, (train, val) in enumerate(expanding_window_cv(dataset)):
        ...     model = train_model(train)
        ...     score = evaluate(model, val)
    """
    # === 1. Validate inputs ===
    if min_train_frac < 0 or val_frac < 0 or min_train_frac + val_frac > 1:
        raise ValueError(
            "train_frac and val_frac must be comprised between 0 and 1"
        )
    
    if len(dataset) < 2 * gap:
        raise ValueError(
            f"len(dataset) must be at least {2 * gap}; was {len(dataset)}"
        )
    
    # === 2. Compute effective length ===
    T = len(dataset)
    total_gap = 2 * gap
    usable_len = T - total_gap
    
    min_train_len = int(min_train_frac * usable_len)
    increment = int((T - min_train_len - gap) / n_folds)
    val_len = int(val_frac * usable_len)
    
    for fold in range(n_folds):
        # === 3. Compute split points ===
        train_end = min_train_len + increment * fold
        
        val_start = train_end + gap
        val_end = min(val_start + val_len, T)
        
        yield dataset[:train_end], dataset[val_start:val_end]


def rolling_window_cv(
    dataset: TimeSeriesDataset,
    n_folds: int = 5,
    val_frac: float = 0.1,
    train_frac: float = 0.3,
    gap: int = 0,
) -> Iterator[tuple[TimeSeriesDataset, TimeSeriesDataset]]:
    """Generate rolling (sliding) window cross-validation folds.

    Train window has fixed size and slides forward with each fold.
    Respects temporal order: train always precedes validation.

    Fold structure:
        Fold 0: [TRAIN════]        [VAL]
        Fold 1:    [TRAIN════]          [VAL]
        Fold 2:       [TRAIN════]            [VAL]

    Args:
        dataset: TimeSeriesDataset to split.
        n_folds: Number of CV folds.
        val_frac: Validation size as fraction of usable data.
        train_frac: Fixed train size as fraction of usable data.
        gap: Timesteps between train and val to prevent leakage.
            Set to horizon_len to prevent forecast horizon overlap.

    Yields:
        (train, val) TimeSeriesDataset tuples for each fold.

    Raises:
        ValueError: If fractions invalid or dataset too short.

    Example:
        >>> for fold_idx, (train, val) in enumerate(rolling_window_cv(dataset)):
        ...     model = train_model(train)
        ...     score = evaluate(model, val)
    """
    # === 1. Validate inputs ===
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError(
            "train_frac and val_frac must be comprised between 0 and 1"
        )
    
    if len(dataset) < 2 * gap:
        raise ValueError(
            f"len(dataset) must be at least {2 * gap}; was {len(dataset)}"
        )
    
    # === 2. Compute effective length ===
    T = len(dataset)
    total_gap = 2 * gap
    usable_len = T - total_gap
    
    train_len = int(train_frac * usable_len)
    increment = int((T - train_len - gap) / n_folds)
    val_len = int(val_frac * usable_len)
    
    for fold in range(n_folds):
        # === 3. Compute split points ===
        train_start = increment * fold
        train_end = train_len + increment * fold
        
        val_start = train_end + gap
        val_end = min(val_start + val_len, T)
        
        yield dataset[train_start:train_end], dataset[val_start:val_end]