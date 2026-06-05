import jax.numpy as jnp
from jax_prism.data.batch import TimeSeriesBatch

def create_sliding_windows(
    dataset: "TimeSeriesDataset",
    context_len: int,
    horizon_len: int,
    stride: int = 1,
) -> list[TimeSeriesBatch]:
    """Generate sliding windows from a TimeSeriesDataset.
    
    Args:
        dataset: Source TimeSeriesDataset.
        context_len: Number of historical timesteps (encoder input).
        horizon_len: Number of future timesteps to predict (decoder input).
        stride: Step size between consecutive windows. Default 1.
        
    Returns:
        List of TimeSeriesBatch objects, one per window.
        
    Raises:
        ValueError: If dataset is too short for even one window.
    """
    T = len(dataset)
    window_size = context_len + horizon_len
    
    # Validate: dataset must be long enough for at least one window
    if T < window_size:
        raise ValueError(f"Dataset length {T} is less than window size {window_size}.")
    
    num_windows = (T - window_size) // stride + 1
    
    batches = []
    
    for i in range(num_windows):
        start = i * stride
        past_end = start + context_len
        future_end = past_end + horizon_len
        
        batch = TimeSeriesBatch(
            past_targets=dataset.targets[start:past_end][None, ...],
            future_targets=dataset.targets[past_end:future_end][None, ...],
            past_known_covariates=dataset.known_covariates[start:past_end][None, ...] if dataset.known_covariates is not None else None,
            future_known_covariates=dataset.known_covariates[past_end:future_end][None, ...] if dataset.known_covariates is not None else None,
            past_observed_covariates=dataset.observed_covariates[start:past_end][None, ...] if dataset.observed_covariates is not None else None,
            static_covariates=dataset.static_covariates[None, ...] if dataset.static_covariates is not None else None,
        )
        
        batches.append(batch)
    
    return batches