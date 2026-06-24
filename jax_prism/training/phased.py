"""Phased training orchestration for multi-stage model training.

This module provides utilities for running multi-phase training, where each
phase can have different:
- Loss functions (e.g., MSE -> NLL -> NLL with fixed sigma)
- Learning rates and schedules
- Frozen parameters (backbone, specific heads, output indices)
- Calibration and reinitialization between phases

Example three-phase Gaussian training:
    Phase 1: NLL with fixed σ (train backbone + μ)
    Phase 2: MSE warmup (refine point prediction)
    Phase 3: NLL (freeze backbone + μ, train σ)

Example two-phase quantile training:
    Phase 1: Train backbone + all outputs
    Phase 2: Freeze backbone + median, train delta biases only
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence
import warnings

import jax
import jax.numpy as jnp
import optax
from flax import core

from jax_prism.losses.base import Loss
from jax_prism.data.batch import TimeSeriesBatch
from jax_prism._typing import PRNGKey, PyTree
from jax_prism.training.calibration import calibrate_output_bias
from jax_prism.training.freezing import freeze_params_by_pattern
from jax_prism.training.masking import freeze_output_indices
from jax_prism.training.reinit import reinitialize_param_head


@dataclass
class TrainingPhase:
    """Configuration for a single training phase.

    Args:
        name: Phase identifier for history tracking.
        epochs: Number of epochs to train in this phase.
        learning_rate: Either a float (constant) or a schedule function.
        loss: Loss function conforming to Loss protocol.
        frozen_patterns: Regex patterns for freeze_params_by_pattern.
            Freezes entire parameter subtrees (e.g., "backbone").
        frozen_output_indices: Indices to freeze in output bias vector.
            Uses freeze_output_indices for gradient masking.
        recalibrate_bias: Whether to run calibrate_output_bias before this phase.
        reinit_head_indices: Which heads to reinitialize before this phase.
    """
    name: str
    epochs: int
    learning_rate: float | Callable
    loss: Loss
    frozen_patterns: tuple[str, ...] = ()
    frozen_output_indices: tuple[int, ...] = ()
    recalibrate_bias: bool = False
    reinit_head_indices: tuple[int, ...] = ()


class PhasedTrainer:
    """Orchestrates multi-phase training with per-phase configuration.

    Args:
        model: Flax module (e.g., TemporalFusionTransformer).
            Needs .apply() for forward pass and .init() for reinitialization.
        phases: Sequence of TrainingPhase configs.
        base_optimizer_factory: Function (learning_rate_schedule) -> optimizer.
            Example: lambda lr: optax.adamw(lr, weight_decay=1e-4)
        target_field: Name of target field in TimeSeriesBatch (default: "future_targets").
    """

    def __init__(
        self,
        model,
        phases: Sequence[TrainingPhase],
        base_optimizer_factory: Callable[[optax.Schedule], optax.GradientTransformation],
        target_field: str = "future_targets",
    ):
        self.model = model
        self.phases = phases
        self.base_optimizer_factory = base_optimizer_factory
        self.target_field = target_field

    def fit(
        self,
        params: PyTree,
        train_data: Callable[[], Iterator[TimeSeriesBatch]],
        calibration_batch: TimeSeriesBatch,
        key: PRNGKey,
        *,
        val_data: Callable[[], Iterator[TimeSeriesBatch]] | None = None,
        val_frequency: str = "epoch",
    ) -> tuple[PyTree, dict]:
        """Run all training phases sequentially.

        Args:
            params: Initial parameters.
            train_data: Factory function returning fresh iterator over training batches.
                Called once per epoch to enable reshuffling.
            calibration_batch: Representative batch for bias calibration and head
                reinitialization.
            key: PRNG key.
            val_data: Optional factory for validation data iterator.
            val_frequency: When to run validation: "epoch", "phase", or "none".

        Returns:
            params: Final parameters after all phases.
            history: Dict mapping phase names to lists of per-epoch metrics.
                Example: {"phase1": [{"train_loss": 1.2, "val_loss": 1.3}, ...], ...}
        """
        history = {}
    
        for phase in self.phases:
            # === 1. Prepare parameters ===
            for head_idx in phase.reinit_head_indices:
                key, subkey = jax.random.split(key)
                params = reinitialize_param_head(
                    self.model, params, head_idx, subkey, calibration_batch
                )

            if phase.recalibrate_bias:
                try:
                    params = calibrate_output_bias(
                        self.model, params, calibration_batch
                    )
                except ValueError as e:
                    warnings.warn(
                        f"Phase '{phase.name}': calibrate_output_bias failed: {e}. "
                        "Skipping calibration.",
                        UserWarning
                    )

            # === 2. Build optimizer for this phase ===
            if callable(phase.learning_rate):
                lr_schedule = phase.learning_rate
            else:
                lr_schedule = optax.constant_schedule(phase.learning_rate)

            base_opt = self.base_optimizer_factory(lr_schedule)
            if phase.frozen_patterns:
                optimizer = freeze_params_by_pattern(
                    base_opt, params, list(phase.frozen_patterns)
                )
            else:
                optimizer = base_opt

            # Initialize optimizer state (resets momentum/Adam state)
            opt_state = optimizer.init(params)
                    
            # === 3. Check frozen_output_indices validity ===
            if phase.frozen_output_indices:
                if not _has_output_bias(params):
                    warnings.warn(
                        f"Phase '{phase.name}' specifies frozen_output_indices="
                        f"{phase.frozen_output_indices}, but model has no 'output_bias' "
                        "parameter. Ignoring frozen_output_indices.",
                        UserWarning
                    )

            # === 4. Define train_step for this phase ===
            @jax.jit
            def train_step(params, opt_state, batch, key):
                def loss_fn_wrapper(p):
                    raw_output = self.model.apply(p, batch, training=True, rngs={"dropout": key})
                    target = getattr(batch, self.target_field)
                    mask = getattr(batch, "mask", None)
                    return phase.loss(raw_output, target, mask)

                # Compute loss and gradients
                loss_value, grads = jax.value_and_grad(loss_fn_wrapper)(params)

                if phase.frozen_output_indices and _has_output_bias(params):
                    grads = freeze_output_indices(grads, phase.frozen_output_indices)

                # Update parameters
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)

                metrics = {"loss": loss_value}
                return new_params, new_opt_state, metrics

            # === 5. Define validation step ===
            @jax.jit
            def val_step(params, batch):
                raw_output = self.model.apply(params, batch, training=False)
                target = getattr(batch, self.target_field)
                mask = getattr(batch, "mask", None)
                loss_value = phase.loss(raw_output, target, mask)
                return {"loss": loss_value}

            # === 6. Training loop ===
            phase_history = []
            for epoch in range(phase.epochs):
                # Training
                epoch_losses = []
                for batch in train_data():
                    key, subkey = jax.random.split(key)
                    params, opt_state, metrics = train_step(params, opt_state, batch, subkey)
                    epoch_losses.append(metrics["loss"])

                # Aggregate training metrics
                epoch_metrics = {"train_loss": jnp.mean(jnp.array(epoch_losses))}

                # Validation (if requested)
                if val_data is not None and val_frequency == "epoch":
                    val_losses = []
                    for batch in val_data():
                        val_metrics = val_step(params, batch)
                        val_losses.append(val_metrics["loss"])
                    epoch_metrics["val_loss"] = jnp.mean(jnp.array(val_losses))

                phase_history.append(epoch_metrics)

            # Phase-level validation
            if val_data is not None and val_frequency == "phase":
                val_losses = []
                for batch in val_data():
                    val_metrics = val_step(params, batch)
                    val_losses.append(val_metrics["loss"])
                # Add to last epoch of phase
                if phase_history:
                    phase_history[-1]["val_loss"] = jnp.mean(jnp.array(val_losses))

            history[phase.name] = phase_history

        return params, history


def _has_output_bias(params: PyTree) -> bool:
    """Check if params contains output_bias parameter.

    Args:
        params: Frozen params dict.

    Returns:
        True if output_bias exists in params["params"], False otherwise.
    """
    params_dict = core.unfreeze(params)
    return "output_bias" in params_dict.get("params", {})