"""Parameter reinitialization utilities for phased training.

This module provides utilities for reinitializing specific parameter heads
between training phases. This is useful when:
- Phase 1 trains backbone + head A, causing head B's outputs to drift
- Phase 2 needs head B to start from a known initialization state
"""

import re
from flax import core
import jax

from jax_prism._typing import Params, PRNGKey


def reinitialize_param_head(
    model,
    params: Params,
    head_idx: int,
    rng: PRNGKey,
    sample_batch,
) -> Params:
    """Reinitialize a specific parameter head to fresh random weights.

    For models using param_head_configs, this reinitializes all parameters
    belonging to a specific head (gate, projection, output bias) while
    preserving all other parameters.

    This is useful in phased training where:
    - Phase 1 trains backbone + head 0, causing head 1's outputs to drift
      (even though head 1's weights are frozen, backbone changes affect its outputs)
    - Before Phase 2, reinitialize head 1 to known init state

    Args:
        model: Flax module (e.g., TemporalFusionTransformer).
        params: Current frozen params dict.
        head_idx: Index of the head to reinitialize (0-indexed).
        rng: PRNG key for random initialization.
        sample_batch: Sample input batch for model.init() shape inference.

    Returns:
        Updated params with head reinitalized to fresh random weights.

    Example:
        >>> # After Phase 1, before Phase 2
        >>> rng_reinit = jax.random.PRNGKey(9999)
        >>> params = reinitialize_param_head(
        ...     model, params, head_idx=1, rng=rng_reinit, sample_batch=calib_batch
        ... )
        >>> # Head 1 now has fresh random weights
    """
    # Get fresh initialization for the entire model
    fresh_params = model.init(rng, sample_batch)

    # Pattern to match head-specific parameter paths
    head_pattern = re.compile(rf"^param_{head_idx}_")

    # Unfreeze both param dicts
    params_dict = core.unfreeze(params)
    fresh_dict = core.unfreeze(fresh_params)

    # Copy head-specific params from fresh init
    for key in list(params_dict["params"].keys()):
        if head_pattern.match(key):
            params_dict["params"][key] = fresh_dict["params"][key]

    return core.freeze(params_dict)


def get_head_param_keys(params: Params, head_idx: int) -> list[str]:
    """Get all parameter keys belonging to a specific head.

    Args:
        params: Frozen params dict.
        head_idx: Index of the head.

    Returns:
        List of parameter keys, e.g., ['param_1_gate', 'param_1_proj', 'param_1_output_bias'].
    """
    head_pattern = re.compile(rf"^param_{head_idx}_")
    return [k for k in params["params"].keys() if head_pattern.match(k)]
