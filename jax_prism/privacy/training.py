"""High-level DP-SGD training utilities.

This module provides factory functions that create JIT-compiled training
step functions with built-in differential privacy.

Two levels of abstraction:
- `make_dp_train_step`: For custom training loops (Option C)
- `Trainer` class: Convenience wrapper (Option D, in nn/trainer.py)

References:
    Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from jax_prism._typing import Array, Loss, PRNGKey, PyTree, PrivacyAccountant
from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.privacy.gradients import dp_gradients


class TrainStepOutput(NamedTuple):
    """Output from a single training step.

    Used by both DP and non-DP training steps. For non-DP training,
    accountant will be None.
    """

    params: PyTree
    opt_state: PyTree
    accountant: PrivacyAccountant | None
    metrics: dict[str, Array]


def make_dp_train_step(
    model_apply: Callable[[PyTree, TimeSeriesBatch, bool, PRNGKey | None], Array],
    loss_fn: Loss,
    optimizer: optax.GradientTransformation,
    clip_norm: float,
    noise_multiplier: float,
    *,
    target_field: str = "future_targets",
    grad_transform: Callable[[PyTree], PyTree] | None = None,
) -> Callable[
    [PyTree, PyTree, PrivacyAccountant, TimeSeriesBatch, PRNGKey],
    TrainStepOutput,
]:
    """Create a JIT-compiled DP-SGD training step function.

    This factory creates a training step that:
    1. Computes per-sample gradients
    2. Clips each gradient to bound sensitivity
    3. Aggregates (averages) clipped gradients
    4. Adds calibrated Gaussian noise
    5. Optionally transforms gradients (e.g., freeze specific indices)
    6. Updates parameters with optimizer
    7. Updates privacy accountant

    Args:
        model_apply: Function (params, batch, training, rngs) -> raw_output.
            Typically model.apply for a Flax model.
        loss_fn: Loss function conforming to Loss protocol.
            Called as loss_fn(predictions, targets, mask).
        optimizer: Optax optimizer (e.g., optax.adam(1e-3)).
        clip_norm: Maximum L2 norm for per-sample gradients.
        noise_multiplier: σ, ratio of noise std to clip_norm.
        target_field: Field name in batch for targets (default: "future_targets").
        grad_transform: Optional function to transform gradients before optimizer
            update. Example: lambda g: freeze_output_indices(g, (0,))

    Returns:
        A JIT-compiled function with signature:
            (params, opt_state, accountant, batch, key) -> TrainStepOutput

    Example:
        >>> model = TFT(config)
        >>> params = model.init(key, sample_batch)
        >>> loss_fn = NLLLoss(GaussianHead())
        >>> optimizer = optax.adam(1e-3)
        >>> opt_state = optimizer.init(params)
        >>>
        >>> train_step = make_dp_train_step(
        ...     model.apply,
        ...     loss_fn,
        ...     optimizer,
        ...     clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ... )
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     key, subkey = jax.random.split(key)
        ...     output = train_step(params, opt_state, accountant, batch, subkey)
        ...     params = output.params
        ...     opt_state = output.opt_state
        ...     accountant = output.accountant
        ...     print(f"Loss: {output.metrics['loss']:.4f}")
    """

    def single_example_loss(params: PyTree, example: TimeSeriesBatch, key: PRNGKey) -> Array:
        """Compute loss for a single example."""
        # Forward pass (training=True)
        raw_output = model_apply(params, example, True, rngs={"dropout": key})

        # Get target from batch
        target = getattr(example, target_field)

        # Get mask if available
        mask = getattr(example, "mask", None)

        # Compute loss
        return loss_fn(raw_output, target, mask)

    def train_step(
        params: PyTree,
        opt_state: PyTree,
        accountant: PrivacyAccountant,
        batch: TimeSeriesBatch,
        key: PRNGKey,
    ) -> TrainStepOutput:
        """Execute one DP-SGD training step."""
        dropout_key, noise_key = jax.random.split(key)
        # Get batch size for sample rate computation
        batch_size = getattr(batch, target_field).shape[0]

        # Compute per-sample gradients using vmap
        grad_fn = jax.grad(single_example_loss)
        dropout_keys = jax.random.split(dropout_key, batch_size)
        per_sample_grads = jax.vmap(grad_fn, in_axes=(None, 0, 0))(params, batch, dropout_keys)

        # Apply DP: clip, aggregate, add noise
        dp_grads = dp_gradients(
            per_sample_grads,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            key=noise_key,
        )

        # Apply optional gradient transform
        if grad_transform is not None:
            dp_grads = grad_transform(dp_grads)

        # Compute metrics (on clipped grads before noise for interpretability)
        # We compute loss separately for metrics (no noise influence)
        raw_output = model_apply(params, batch, False, rngs={"dropout": None})
        target = getattr(batch, target_field)
        mask = getattr(batch, "mask", None)
        loss_value = loss_fn(raw_output, target, mask)

        # Update parameters
        updates, new_opt_state = optimizer.update(dp_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Update accountant
        # Note: sample_rate should be batch_size / dataset_size, but we
        # don't have dataset_size here. User should call accountant.step
        # separately if precise accounting is needed.
        # For now, we assume sample_rate=1.0 (conservative)
        new_accountant = accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=1.0,  # Conservative; override externally if needed
            num_steps=1,
        )

        metrics = {
            "loss": loss_value,
            "batch_size": jnp.array(batch_size, dtype=jnp.int32),
        }

        return TrainStepOutput(
            params=new_params,
            opt_state=new_opt_state,
            accountant=new_accountant,
            metrics=metrics,
        )

    # JIT compile the training step
    return jax.jit(train_step)
