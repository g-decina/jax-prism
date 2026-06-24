"""Training loop utilities (non-DP).

This module provides factory functions for creating JIT-compiled training steps
without differential privacy. For DP training, use privacy.training module.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from jax_prism._typing import Array, Loss, PRNGKey, PyTree
from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.privacy.training import TrainStepOutput


def make_train_step(
    model_apply: Callable[[PyTree, TimeSeriesBatch, bool, PRNGKey | None], Array],
    loss_fn: Loss,
    optimizer: optax.GradientTransformation,
    *,
    target_field: str = "future_targets",
    grad_transform: Callable[[PyTree], PyTree] | None = None,
) -> Callable[[PyTree, PyTree, TimeSeriesBatch, PRNGKey], TrainStepOutput]:
    """Create a JIT-compiled training step **WITHOUT differential privacy**.

    For DP training, use `jax_prism.privacy.training.make_dp_train_step`.

    Args:
        model_apply: Function (params, batch, training, rngs) -> raw_output.
            Typically model.apply for a Flax model.
        loss_fn: Loss function conforming to Loss protocol.
            Called as loss_fn(predictions, targets, mask).
        optimizer: Optax optimizer (e.g., optax.adam(1e-3)).
        target_field: Field name in batch for targets (default: "future_targets").
        grad_transform: Optional function to transform gradients before optimizer
            update. Example: lambda g: freeze_output_indices(g, (0,))

    Returns:
        JIT-compiled function: (params, opt_state, batch, key) -> TrainStepOutput

    Example:
        >>> model = TFT(config)
        >>> params = model.init(key, sample_batch)
        >>> loss_fn = NLLLoss(GaussianHead())
        >>> optimizer = optax.adam(1e-3)
        >>> opt_state = optimizer.init(params)
        >>>
        >>> train_step = make_train_step(
        ...     model.apply,
        ...     loss_fn,
        ...     optimizer,
        ...     target_field="future_targets",
        ... )
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     key, subkey = jax.random.split(key)
        ...     output = train_step(params, opt_state, batch, subkey)
        ...     params, opt_state = output.params, output.opt_state
        ...     print(f"Loss: {output.metrics['loss']:.4f}")
    """

    def loss_fn_wrapper(params: PyTree, batch: TimeSeriesBatch, key: PRNGKey) -> Array:
        raw_output = model_apply(params, batch, True, rngs={"dropout": key})
        target = getattr(batch, target_field)
        mask = getattr(batch, "mask", None)
        return loss_fn(raw_output, target, mask)

    @jax.jit
    def train_step(
        params: PyTree,
        opt_state: PyTree,
        batch: TimeSeriesBatch,
        key: PRNGKey
    ) -> TrainStepOutput:
        loss_value, grads = jax.value_and_grad(
            lambda p: loss_fn_wrapper(p, batch, key)
        )(params)

        # Apply optional gradient transform
        if grad_transform is not None:
            grads = grad_transform(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics = {"loss": loss_value}
        return TrainStepOutput(
            params=new_params,
            opt_state=new_opt_state,
            accountant=None,  # No accountant for non-DP
            metrics=metrics,
        )

    return train_step