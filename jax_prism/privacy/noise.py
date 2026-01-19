"""Gaussian noise mechanism for differential privacy.

The Gaussian mechanism adds calibrated noise to query outputs to achieve
differential privacy. For DP-SGD, noise is added to the clipped gradient sum.

The noise standard deviation is: σ × C
Where:
    σ (noise_multiplier): Controls privacy-utility tradeoff
    C (clip_norm): Sensitivity bound from gradient clipping

References:
    Dwork & Roth, "The Algorithmic Foundations of Differential Privacy", 2014.
    Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array, PRNGKey, PyTree


def add_noise(
    grads: PyTree,
    noise_multiplier: float,
    clip_norm: float,
    key: PRNGKey,
) -> PyTree:
    """Add Gaussian noise to gradients for differential privacy.

    Adds independent Gaussian noise N(0, (σC)²) to each element of the
    gradient pytree, where σ is the noise_multiplier and C is the clip_norm.

    Args:
        grads: Gradient pytree (already clipped and aggregated).
        noise_multiplier: σ, controls privacy level. Higher = more privacy.
        clip_norm: C, the clipping bound used. Determines noise scale.
        key: JAX PRNG key for reproducible randomness.

    Returns:
        Noisy gradients with same structure as input.

    Example:
        >>> key = jax.random.key(0)
        >>> grads = {'w': jnp.zeros((64, 64)), 'b': jnp.zeros(64)}
        >>> noisy = add_noise(grads, noise_multiplier=1.0, clip_norm=1.0, key=key)
        >>> # noisy['w'] ~ N(0, 1) for each element
    """
    # 1. Compute noise std
    noise_std = noise_multiplier * clip_norm
    
    # 2. Generate noise for each leaf in the pytree
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    keys = jax.random.split(key, num=len(leaves))
    
    noisy_leaves = [
        leaf + noise_std * jax.random.normal(k, leaf.shape)
        for leaf, k in zip(leaves, keys)
    ]
    
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)


def generate_noise_tree(
    tree_structure: PyTree,
    noise_std: float,
    key: PRNGKey,
) -> PyTree:
    """Generate a pytree of Gaussian noise matching a template structure.

    Optional helper that separates noise generation from addition.
    Useful for debugging or to inspect the noise.

    Args:
        tree_structure: Template pytree (only structure and shapes used).
        noise_std: Standard deviation for the Gaussian noise.
        key: JAX PRNG key.

    Returns:
        Pytree of noise with same structure and shapes as template.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree_structure)
    keys = jax.random.split(key, num=len(leaves))

    noise_leaves = [
        noise_std * jax.random.normal(k, leaf.shape)
        for leaf, k in zip(leaves, keys)
    ]

    return jax.tree_util.tree_unflatten(treedef, noise_leaves)
