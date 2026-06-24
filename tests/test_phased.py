"""Tests for phased training orchestration."""

import jax
import jax.numpy as jnp
import optax
import pytest
from chex import assert_trees_all_close
from flax import linen as nn

from jax_prism._typing import Array
from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.distributions.gaussian import GaussianHead
from jax_prism.losses.nll import NLLLoss
from jax_prism.losses.mse import MSELoss
from jax_prism.training.phased import PhasedTrainer, TrainingPhase


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleModel(nn.Module):
    """Minimal model for testing phased training."""

    hidden_size: int = 8
    num_outputs: int = 2  # (mu, log_sigma) for Gaussian

    @nn.compact
    def __call__(self, batch: TimeSeriesBatch, training: bool = False) -> Array:
        # Simple: just process past_targets through MLP
        x = batch.past_targets  # (B, T, 1)
        x = nn.Dense(self.hidden_size, name="backbone")(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_outputs, name="head")(x)
        return x  # (B, T, num_outputs)


class ModelWithOutputBias(nn.Module):
    """Model with explicit output_bias for testing freeze_output_indices."""

    hidden_size: int = 8
    num_outputs: int = 2

    @nn.compact
    def __call__(self, batch: TimeSeriesBatch, training: bool = False) -> Array:
        x = batch.past_targets
        x = nn.Dense(self.hidden_size, name="backbone")(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_outputs, name="head")(x)
        # Add explicit output bias
        output_bias = self.param(
            "output_bias",
            nn.initializers.zeros,
            (self.num_outputs,),
        )
        return x + output_bias


@pytest.fixture
def sample_batch():
    """Create a simple TimeSeriesBatch for testing."""
    B, T = 4, 10
    return TimeSeriesBatch(
        past_targets=jnp.ones((B, T, 1)),
        future_targets=jnp.ones((B, T, 1)) * 2.0,  # Different from past for loss
    )


@pytest.fixture
def simple_model():
    """Create and initialize SimpleModel."""
    return SimpleModel()


@pytest.fixture
def model_with_bias():
    """Create and initialize ModelWithOutputBias."""
    return ModelWithOutputBias()


@pytest.fixture
def gaussian_head():
    """Create GaussianHead distribution."""
    return GaussianHead()


@pytest.fixture
def nll_loss(gaussian_head):
    """Create NLLLoss with GaussianHead."""
    return NLLLoss(distribution=gaussian_head)


@pytest.fixture
def mse_loss():
    """Create MSELoss."""
    return MSELoss()


def make_train_data(batch: TimeSeriesBatch):
    """Factory that returns iterator over single batch."""
    def iterator():
        yield batch
    return iterator


# =============================================================================
# TrainingPhase Tests
# =============================================================================


class TestTrainingPhase:
    """Tests for TrainingPhase dataclass."""

    def test_basic_construction(self, nll_loss):
        """TrainingPhase can be constructed with required fields."""
        phase = TrainingPhase(
            name="test",
            epochs=10,
            learning_rate=1e-3,
            loss=nll_loss,
        )

        assert phase.name == "test"
        assert phase.epochs == 10
        assert phase.learning_rate == 1e-3
        assert phase.loss is nll_loss

    def test_default_values(self, nll_loss):
        """TrainingPhase has correct defaults."""
        phase = TrainingPhase(
            name="test",
            epochs=10,
            learning_rate=1e-3,
            loss=nll_loss,
        )

        assert phase.frozen_patterns == ()
        assert phase.frozen_output_indices == ()
        assert phase.recalibrate_bias is False
        assert phase.reinit_head_indices == ()

    def test_with_frozen_patterns(self, nll_loss):
        """TrainingPhase accepts frozen_patterns."""
        phase = TrainingPhase(
            name="freeze_backbone",
            epochs=10,
            learning_rate=1e-3,
            loss=nll_loss,
            frozen_patterns=("backbone",),
        )

        assert phase.frozen_patterns == ("backbone",)

    def test_with_frozen_output_indices(self, nll_loss):
        """TrainingPhase accepts frozen_output_indices."""
        phase = TrainingPhase(
            name="freeze_mu",
            epochs=10,
            learning_rate=1e-3,
            loss=nll_loss,
            frozen_output_indices=(0,),  # freeze mu, train sigma
        )

        assert phase.frozen_output_indices == (0,)

    def test_with_callable_learning_rate(self, nll_loss):
        """TrainingPhase accepts callable learning rate (schedule)."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-3,
            warmup_steps=10,
            decay_steps=90,
        )

        phase = TrainingPhase(
            name="scheduled",
            epochs=10,
            learning_rate=schedule,
            loss=nll_loss,
        )

        assert callable(phase.learning_rate)

    def test_with_recalibration(self, nll_loss):
        """TrainingPhase accepts recalibrate_bias flag."""
        phase = TrainingPhase(
            name="recal",
            epochs=10,
            learning_rate=1e-3,
            loss=nll_loss,
            recalibrate_bias=True,
        )

        assert phase.recalibrate_bias is True

    def test_with_reinit_heads(self, nll_loss):
        """TrainingPhase accepts reinit_head_indices."""
        phase = TrainingPhase(
            name="reinit",
            epochs=10,
            learning_rate=1e-3,
            loss=nll_loss,
            reinit_head_indices=(1,),  # reinit sigma head
        )

        assert phase.reinit_head_indices == (1,)


# =============================================================================
# PhasedTrainer Basic Tests
# =============================================================================


class TestPhasedTrainerBasic:
    """Basic tests for PhasedTrainer."""

    def test_initialization(self, simple_model, nll_loss):
        """PhasedTrainer initializes correctly."""
        phases = [
            TrainingPhase(name="phase1", epochs=2, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        assert trainer.model is simple_model
        assert trainer.phases == phases
        assert trainer.target_field == "future_targets"

    def test_single_phase_training(self, simple_model, nll_loss, sample_batch):
        """Single phase training runs and returns params + history."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=2, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        final_params, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        # Check params returned
        assert final_params is not None
        assert "params" in final_params

        # Check history structure
        assert "train" in history
        assert len(history["train"]) == 2  # 2 epochs
        assert "train_loss" in history["train"][0]

    def test_multi_phase_training(self, simple_model, nll_loss, gaussian_head, sample_batch):
        """Multiple phases execute sequentially."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        # MSELoss needs distribution to extract mu correctly
        mse_loss = MSELoss(distribution=gaussian_head)

        phases = [
            TrainingPhase(name="phase1", epochs=2, learning_rate=1e-3, loss=nll_loss),
            TrainingPhase(name="phase2", epochs=3, learning_rate=1e-4, loss=mse_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        final_params, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        # Both phases in history
        assert "phase1" in history
        assert "phase2" in history
        assert len(history["phase1"]) == 2
        assert len(history["phase2"]) == 3

    def test_params_change_during_training(self, simple_model, nll_loss, sample_batch):
        """Parameters actually change during training."""
        key = jax.random.PRNGKey(0)
        initial_params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=5, learning_rate=1e-2, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        final_params, _ = trainer.fit(
            params=initial_params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        # At least some params should have changed
        initial_flat = jax.tree_util.tree_leaves(initial_params)
        final_flat = jax.tree_util.tree_leaves(final_params)

        some_changed = any(
            not jnp.allclose(i, f) for i, f in zip(initial_flat, final_flat)
        )
        assert some_changed, "No parameters changed during training"


# =============================================================================
# PhasedTrainer Freezing Tests
# =============================================================================


class TestPhasedTrainerFreezing:
    """Tests for parameter freezing in PhasedTrainer."""

    def test_frozen_patterns_prevent_updates(self, simple_model, nll_loss, sample_batch):
        """Frozen patterns prevent those params from updating."""
        key = jax.random.PRNGKey(0)
        initial_params = simple_model.init(key, sample_batch)

        # Freeze backbone, only head should train
        phases = [
            TrainingPhase(
                name="freeze_backbone",
                epochs=5,
                learning_rate=1e-2,
                loss=nll_loss,
                frozen_patterns=("backbone",),
            ),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        final_params, _ = trainer.fit(
            params=initial_params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        # Backbone should be unchanged
        assert_trees_all_close(
            initial_params["params"]["backbone"],
            final_params["params"]["backbone"],
        )

        # Head should have changed
        head_changed = not jnp.allclose(
            initial_params["params"]["head"]["kernel"],
            final_params["params"]["head"]["kernel"],
        )
        assert head_changed, "Head should have changed"

    def test_frozen_output_indices_with_output_bias(
        self, model_with_bias, nll_loss, sample_batch
    ):
        """Frozen output indices prevent those bias elements from updating."""
        key = jax.random.PRNGKey(0)
        initial_params = model_with_bias.init(key, sample_batch)

        # Freeze index 0 (mu), train index 1 (sigma)
        phases = [
            TrainingPhase(
                name="freeze_mu",
                epochs=10,
                learning_rate=1e-2,
                loss=nll_loss,
                frozen_output_indices=(0,),
            ),
        ]

        trainer = PhasedTrainer(
            model=model_with_bias,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        final_params, _ = trainer.fit(
            params=initial_params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        # output_bias[0] should be unchanged
        assert_trees_all_close(
            initial_params["params"]["output_bias"][0],
            final_params["params"]["output_bias"][0],
        )

        # output_bias[1] may have changed (depends on gradients)
        # Just verify no error occurred


# =============================================================================
# PhasedTrainer Validation Tests
# =============================================================================


class TestPhasedTrainerValidation:
    """Tests for validation in PhasedTrainer."""

    def test_epoch_level_validation(self, simple_model, nll_loss, sample_batch):
        """Epoch-level validation runs each epoch."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=3, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        _, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
            val_data=make_train_data(sample_batch),
            val_frequency="epoch",
        )

        # Each epoch should have val_loss
        for epoch_metrics in history["train"]:
            assert "val_loss" in epoch_metrics

    def test_phase_level_validation(self, simple_model, nll_loss, sample_batch):
        """Phase-level validation runs only at phase end."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=3, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        _, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
            val_data=make_train_data(sample_batch),
            val_frequency="phase",
        )

        # Only last epoch should have val_loss
        assert "val_loss" not in history["train"][0]
        assert "val_loss" not in history["train"][1]
        assert "val_loss" in history["train"][2]

    def test_no_validation(self, simple_model, nll_loss, sample_batch):
        """No validation when val_data is None."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=2, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        _, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
            val_data=None,
        )

        # No val_loss anywhere
        for epoch_metrics in history["train"]:
            assert "val_loss" not in epoch_metrics


# =============================================================================
# PhasedTrainer Learning Rate Tests
# =============================================================================


class TestPhasedTrainerLearningRate:
    """Tests for learning rate handling in PhasedTrainer."""

    def test_fixed_learning_rate(self, simple_model, nll_loss, sample_batch):
        """Fixed learning rate works."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=2, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        # Should not raise
        final_params, _ = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        assert final_params is not None

    def test_schedule_learning_rate(self, simple_model, nll_loss, sample_batch):
        """Callable learning rate (schedule) works."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=1e-3,
            warmup_steps=5,
            decay_steps=15,
        )

        phases = [
            TrainingPhase(name="train", epochs=2, learning_rate=schedule, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        # Should not raise
        final_params, _ = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        assert final_params is not None

    def test_different_lr_per_phase(self, simple_model, nll_loss, sample_batch):
        """Different learning rates for different phases."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="high_lr", epochs=2, learning_rate=1e-2, loss=nll_loss),
            TrainingPhase(name="low_lr", epochs=2, learning_rate=1e-5, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        final_params, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        assert "high_lr" in history
        assert "low_lr" in history


# =============================================================================
# PhasedTrainer History Tests
# =============================================================================


class TestPhasedTrainerHistory:
    """Tests for history tracking in PhasedTrainer."""

    def test_history_structure(self, simple_model, nll_loss, sample_batch):
        """History has correct structure."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="p1", epochs=2, learning_rate=1e-3, loss=nll_loss),
            TrainingPhase(name="p2", epochs=3, learning_rate=1e-3, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        _, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        # Top-level keys are phase names
        assert set(history.keys()) == {"p1", "p2"}

        # Each phase has list of epoch dicts
        assert isinstance(history["p1"], list)
        assert len(history["p1"]) == 2
        assert len(history["p2"]) == 3

        # Each epoch dict has train_loss
        for epoch_metrics in history["p1"]:
            assert isinstance(epoch_metrics, dict)
            assert "train_loss" in epoch_metrics

    def test_loss_decreases(self, simple_model, nll_loss, sample_batch):
        """Loss generally decreases during training."""
        key = jax.random.PRNGKey(0)
        params = simple_model.init(key, sample_batch)

        phases = [
            TrainingPhase(name="train", epochs=20, learning_rate=1e-2, loss=nll_loss),
        ]

        trainer = PhasedTrainer(
            model=simple_model,
            phases=phases,
            base_optimizer_factory=lambda lr: optax.adam(lr),
        )

        _, history = trainer.fit(
            params=params,
            train_data=make_train_data(sample_batch),
            calibration_batch=sample_batch,
            key=jax.random.PRNGKey(1),
        )

        losses = [e["train_loss"] for e in history["train"]]

        # Final loss should be less than initial (with some tolerance for noise)
        assert losses[-1] < losses[0], "Loss should decrease during training"
