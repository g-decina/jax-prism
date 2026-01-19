"""Differentially private gradient computation.

This module provides the main entry point for DP-SGD gradient computation,
composing per-sample gradient clipping with Gaussian noise addition.

The DP-SGD pipeline:
    1. Compute per-sample gradients (user provides these)
    2. Clip each sample's gradient to bound sensitivity
    3. Aggregate (average) the clipped gradients
    4. Add calibrated Gaussian noise

References:
    Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

import jax
import jax.numpy as jnp

from jax_prism._typing import Array, PRNGKey, PyTree
from jax_prism.privacy.clipping import clip_gradients
from jax_prism.privacy.noise import add_noise


def dp_gradients(
    per_sample_grads: PyTree,
    clip_norm: float,
    noise_multiplier: float,
    key: PRNGKey,
) -> PyTree:
    """Compute differentially private gradients from per-sample gradients.

    This is the main DP-SGD gradient function. It takes per-sample gradients
    (where each leaf has a batch dimension), clips them, aggregates, and adds noise.

    The privacy guarantee depends on:
        - clip_norm (C): Bounds per-sample sensitivity
        - noise_multiplier (σ): Controls noise scale as σ * C
        - batch_size: Affects subsampling amplification

    Args:
        per_sample_grads: Pytree with leaves of shape (batch_size, ...).
            These are the raw per-sample gradients from your loss function.
        clip_norm: Maximum L2 norm for each sample's gradient.
        noise_multiplier: σ, the noise multiplier. Noise std = σ * C.
        key: JAX PRNG key for noise generation.

    Returns:
        Differentially private gradient pytree with batch dimension removed.
        Ready to pass to your optimizer.

    Example:
        >>> # In your training loop:
        >>> per_sample_grads = compute_per_sample_grads(params, batch)
        >>> key, subkey = jax.random.split(key)
        >>> dp_grads = dp_gradients(
        ...     per_sample_grads,
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ...     key=subkey,
        ... )
        >>> updates, opt_state = optimizer.update(dp_grads, opt_state, params)
        >>> params = optax.apply_updates(params, updates)
    """
    # Step 1-2: Clip per-sample gradients and aggregate (average)
    clipped_grads = clip_gradients(per_sample_grads, clip_norm)

    # Step 3: Add calibrated Gaussian noise
    noisy_grads = add_noise(clipped_grads, noise_multiplier, clip_norm, key)

    return noisy_grads


def compute_per_sample_gradients(
    loss_fn,
    params: PyTree,
    batch: PyTree,
) -> PyTree:
    """Compute per-sample gradients using JAX's vmap.

    This is a helper to show the standard pattern for obtaining per-sample
    gradients. You may need to adapt this to your specific loss function.

    The key insight: vmap over the batch dimension of the loss function
    gives you one gradient per sample instead of one aggregated gradient.

    Args:
        loss_fn: Function (params, single_example) -> scalar loss.
            Must take a single example, not a batch.
        params: Model parameters (pytree).
        batch: Batch of examples. Each leaf should have shape (B, ...).

    Returns:
        Per-sample gradients. Each leaf has shape (B, ...) matching
        the params structure but with a leading batch dimension.

    Example:
        >>> def single_loss(params, x, y):
        ...     pred = model.apply(params, x)
        ...     return jnp.mean((pred - y) ** 2)
        >>>
        >>> # Wrap to take (params, (x, y)) format
        >>> loss_fn = lambda p, example: single_loss(p, example[0], example[1])
        >>> per_sample_grads = compute_per_sample_gradients(loss_fn, params, (X, Y))
    """
    grad_fn = jax.grad(loss_fn)
    return jax.vmap(grad_fn, in_axes=(None, 0))(params, batch)