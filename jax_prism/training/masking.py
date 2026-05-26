"""Weight decay masking utilities for optax optimizers."""

from typing import Any

import jax
from flax import traverse_util

PyTree = Any


def no_weight_decay_on_bias(params: PyTree) -> PyTree:
    """Create a mask that excludes bias parameters from weight decay.

    Use with optax.adamw's `mask` argument to prevent weight decay from
    affecting bias terms. This is important when biases have been carefully
    initialized (e.g., quantile delta biases set to positive values).

    Args:
        params: Model parameters (nested dict/pytree).

    Returns:
        Boolean pytree with same structure. True = apply weight decay,
        False = no weight decay (for bias params).

    Example:
        >>> optimizer = optax.adamw(
        ...     learning_rate=1e-3,
        ...     weight_decay=1e-4,
        ...     mask=no_weight_decay_on_bias,
        ... )
        >>> opt_state = optimizer.init(params)
    """
    # Use jax.tree_util to traverse and create mask with same structure
    def _is_not_bias(path, _):
        # path is a tuple of keys, check if last key contains 'bias'
        path_str = ".".join(str(k) for k in path)
        return "bias" not in path_str

    # tree_map_with_path preserves exact structure
    return jax.tree_util.tree_map_with_path(_is_not_bias, params)


def no_weight_decay_on_pattern(pattern: str) -> callable:
    """Create a mask factory that excludes params matching a pattern.

    Args:
        pattern: String to match in parameter path (e.g., "bias", "scale").

    Returns:
        Function that takes params and returns a mask pytree.

    Example:
        >>> # Exclude both biases and layer norm scales from weight decay
        >>> optimizer = optax.adamw(
        ...     learning_rate=1e-3,
        ...     weight_decay=1e-4,
        ...     mask=no_weight_decay_on_pattern("bias"),
        ... )
    """

    def mask_fn(params: PyTree) -> PyTree:
        def _matches_pattern(path, _):
            path_str = ".".join(str(k) for k in path)
            # True = apply decay, False = skip decay
            return pattern not in path_str

        return jax.tree_util.tree_map_with_path(_matches_pattern, params)

    return mask_fn


def freeze_output_indices(
    grads: PyTree,
    frozen_indices: tuple[int, ...],
    bias_path: str = "output_bias",
) -> PyTree:
    """Zero out gradients for specific indices in output bias.

    Use this in Phase 2 training to freeze the median (index 0) while
    allowing delta biases (indices 1-4) to train.

    Args:
        grads: Gradient pytree from jax.grad.
        frozen_indices: Tuple of output indices to freeze (e.g., (0,) for median).
        bias_path: String pattern to match output bias in param path.

    Returns:
        Modified gradient pytree with frozen indices zeroed out.

    Example:
        >>> def train_step(params, batch):
        ...     loss, grads = jax.value_and_grad(loss_fn)(params)
        ...     # Freeze median (index 0), train deltas (indices 1-4)
        ...     grads = freeze_output_indices(grads, frozen_indices=(0,))
        ...     return optimizer.update(grads, opt_state, params)
    """
    import jax.numpy as jnp

    def _maybe_mask(path, grad):
        path_str = ".".join(str(k) for k in path)
        if bias_path in path_str and "bias" in path_str:
            # This is the output bias array — zero out frozen indices
            mask = jnp.ones_like(grad)
            for idx in frozen_indices:
                mask = mask.at[idx].set(0.0)
            return grad * mask
        return grad

    return jax.tree_util.tree_map_with_path(_maybe_mask, grads)
