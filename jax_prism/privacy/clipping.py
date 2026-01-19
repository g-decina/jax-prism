"""Per-sample gradient clipping for differential privacy.

Gradient clipping bounds the influence of any single training example,
which is essential for calibrating the noise in DP-SGD.

Two clipping strategies:
1. Global clipping: Clip the entire gradient pytree by its global L2 norm
2. Per-layer clipping: Clip each parameter tensor independently (not implemented in v0.1.0)

References:
    Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array, PyTree


def compute_global_norm(grads: PyTree) -> Array:
    """Compute the global L2 norm of a gradient pytree.

    The global norm is the square root of the sum of squared norms
    of all leaves in the pytree.

    Args:
        grads: Gradient pytree (e.g., from jax.grad).

    Returns:
        Scalar L2 norm across all parameters.

    Example:
        >>> grads = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> norm = compute_global_norm(grads)  # sqrt(1 + 4 + 9) = sqrt(14)
    """
    # Flatten pytree to list of arrays
    leaves = jax.tree_util.tree_leaves(grads)

    # Sum of squared norms across all leaves
    sum_sq = sum(jnp.sum(leaf ** 2) for leaf in leaves)

    return jnp.sqrt(sum_sq)


def clip_single_gradient(grads: PyTree, clip_norm: float) -> PyTree:
    """Clip a single gradient pytree to have bounded global norm.

    Applies the clipping formula:
        ĝ = g / max(1, ‖g‖₂ / C)

    This ensures ‖ĝ‖₂ ≤ C while preserving direction.

    Args:
        grads: Gradient pytree for a single example.
        clip_norm: Maximum allowed L2 norm (C).

    Returns:
        Clipped gradient pytree with ‖output‖₂ ≤ clip_norm.
    """
    # 1. Compute global L2 norm
    norm = compute_global_norm(grads)
    
    factor = clip_norm / jnp.maximum(clip_norm, norm)
    
    return jax.tree_util.tree_map(lambda leaf: leaf * factor, grads)


def clip_gradients(per_sample_grads: PyTree, clip_norm: float) -> PyTree:
    """Clip per-sample gradients and aggregate.

    Takes a pytree where each leaf has shape (batch_size, ...) representing
    per-sample gradients, clips each sample's gradient to the clip norm,
    then averages across the batch.

    This is the core operation in DP-SGD:
        1. Clip each gᵢ to have ‖gᵢ‖ ≤ C
        2. Aggregate: ḡ = (1/B) Σᵢ ĝᵢ

    Args:
        per_sample_grads: Pytree with leaves of shape (B, ...).
            Each "row" is one sample's gradient.
        clip_norm: Maximum L2 norm per sample.

    Returns:
        Aggregated clipped gradients, pytree with leaves of shape (...).
        Same structure as input but without batch dimension.

    Example:
        >>> # Per-sample gradients for batch of 4
        >>> grads = {'w': jnp.ones((4, 10)), 'b': jnp.ones((4, 5))}
        >>> clipped = clip_gradients(grads, clip_norm=1.0)
        >>> clipped['w'].shape  # (10,) — batch dimension removed
    """
    clipped = jax.vmap(
        lambda grads: clip_single_gradient(grads, clip_norm)
    )(per_sample_grads)
    
    return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), clipped)


def get_batch_size(per_sample_grads: PyTree) -> int:
    """Extract batch size from per-sample gradient pytree.

    Args:
        per_sample_grads: Pytree with leaves of shape (B, ...).

    Returns:
        Batch size B.
    """
    # Get first leaf and return its first dimension
    leaves = jax.tree_util.tree_leaves(per_sample_grads)
    if not leaves:
        raise ValueError("Empty gradient pytree")
    return leaves[0].shape[0]
