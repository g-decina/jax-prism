"""Output bias calibration for models with learnable output biases.

This module provides utilities for calibrating output biases to center
model predictions at target means. This is critical for models with
complex architectures (like TFT with GRN gates) where random initialization
can produce biased initial outputs.

Supports both:
- Multi-head models (param_head_configs with param_N_output_bias)
- Single-head models (use_output_bias=True with output_bias)

Example:
    >>> params = model.init(rng, sample_batch)
    >>> params = calibrate_output_bias(model, params, calibration_batch)
    >>> # Now model outputs are centered at target mean
"""

import jax.numpy as jnp
import re
from flax import core
from typing import Mapping

from jax_prism._typing import Array, Params
from jax_prism.data import TimeSeriesBatch


def calibrate_output_bias(
    model,
    params: Params,
    batch: TimeSeriesBatch,
    target_means: Mapping[int, float] | None = None,
    point_index: int = 0,
) -> Params:
    """Calibrate output biases to center predictions at target means.

    For models with output biases (either multi-head with param_head_configs
    or single-head with use_output_bias=True), this function adjusts the
    output bias so that predictions are centered at the target mean.

    Args:
        model: Flax module with output biases (e.g., TemporalFusionTransformer).
        params: Frozen params dict from model.init().
        batch: TimeSeriesBatch for calibration. Should be representative of
            training data (e.g., 100-1000 samples).
        target_means: Optional mapping from output index to target mean.
            If None, uses defaults:
            - Index at point_index: batch.future_targets.mean()
            - Other indices: no correction
        point_index: Index of the point-estimation output. Defaults to 0.
            For Gaussian: 0 (μ). For quantile with monotonicity: 0 (median).

    Returns:
        Updated params with calibrated output biases.

    Raises:
        ValueError: If model has no output biases or batch has no future_targets.

    Example (multi-head Gaussian):
        >>> params = calibrate_output_bias(model, params, calib_batch)
        >>> # Calibrates μ head (index 0) to target mean

    Example (single-head quantile):
        >>> # For quantile regression, median is at index 0 in raw outputs
        >>> params = calibrate_output_bias(model, params, calib_batch, point_index=0)
    """
    # 1. Validate inputs
    if batch.future_targets is None:
        raise ValueError("batch.future_targets must not be None.")

    # 2. Detect which bias path(s) exist
    multi_head_paths = _get_multi_head_bias_paths(params)
    single_head_path = _get_single_head_bias_path(params)

    if multi_head_paths:
        return _calibrate_multi_head(
            model, params, batch, multi_head_paths, target_means, point_index
        )
    elif single_head_path:
        return _calibrate_single_head(
            model, params, batch, single_head_path, target_means, point_index
        )
    else:
        raise ValueError(
            "No output bias found. Model must have either "
            "param_N_output_bias (multi-head) or output_bias (single-head)."
        )


def _calibrate_multi_head(
    model,
    params: Params,
    batch: TimeSeriesBatch,
    bias_paths: list[str],
    target_means: Mapping[int, float] | None,
    point_index: int,
) -> Params:
    """Calibrate multi-head models (param_head_configs path)."""
    # Forward pass to get current output means
    preds = model.apply(params, batch)
    n_heads = len(bias_paths)

    corrections = {}
    for i in range(n_heads):
        current = float(preds[..., i].mean())

        if target_means and i in target_means:
            target = target_means[i]
        elif i == point_index:
            target = float(batch.future_targets.mean())
        else:
            continue

        corrections[i] = target - current

    # Update bias parameters
    params_dict = core.unfreeze(params)

    for i, correction in corrections.items():
        raw_param = params_dict["params"][bias_paths[i]]["bias"]
        params_dict["params"][bias_paths[i]]["bias"] = raw_param + jnp.array(
            [correction]
        )

    return core.freeze(params_dict)


def _calibrate_single_head(
    model,
    params: Params,
    batch: TimeSeriesBatch,
    bias_path: str,
    target_means: Mapping[int, float] | None,
    point_index: int,
) -> Params:
    """Calibrate single-head models (use_output_bias path).

    For single-head models, the bias is a vector of shape (num_outputs,).
    We only correct the output at point_index (e.g., median for quantile).
    """
    # Forward pass to get current output means
    preds = model.apply(params, batch)
    n_outputs = preds.shape[-1]

    # Build corrections for each output index
    corrections = jnp.zeros(n_outputs)

    for i in range(n_outputs):
        current = float(preds[..., i].mean())

        if target_means and i in target_means:
            target = target_means[i]
            corrections = corrections.at[i].set(target - current)
        elif i == point_index:
            target = float(batch.future_targets.mean())
            corrections = corrections.at[i].set(target - current)

    # Update bias parameter
    params_dict = core.unfreeze(params)
    raw_param = params_dict["params"][bias_path]["bias"]
    params_dict["params"][bias_path]["bias"] = raw_param + corrections

    return core.freeze(params_dict)

def _get_multi_head_bias_paths(params: Params) -> list[str]:
    """Find all param_*_output_bias paths in params (multi-head path).

    Args:
        params: Frozen params dict.

    Returns:
        List of bias parameter paths sorted by head index,
        e.g., ['param_0_output_bias', 'param_1_output_bias'].
        Empty list if no multi-head biases found.
    """
    pattern = re.compile(r"^param_(\d+)_output_bias$")
    matches = [
        (k, int(pattern.match(k).group(1)))
        for k in params["params"].keys()
        if pattern.match(k)
    ]

    sorted_matches = sorted(matches, key=lambda x: x[1])
    return [k for k, _ in sorted_matches]


def _get_single_head_bias_path(params: Params) -> str | None:
    """Find single-head output_bias path in params.

    Args:
        params: Frozen params dict.

    Returns:
        Path string 'output_bias' if found, None otherwise.
    """
    if "output_bias" in params["params"]:
        return "output_bias"
    return None
