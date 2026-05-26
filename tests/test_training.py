"""Tests for training module utilities."""

import jax
import jax.numpy as jnp
import optax
import pytest
from chex import assert_trees_all_close

from jax_prism.training.freezing import freeze_params_by_pattern
from jax_prism.training.schedules import (
    warmup_cosine_schedule,
    warmup_plateau_cosine_schedule,
)


class TestWarmupCosineSchedule:
    """Tests for warmup_cosine_schedule."""

    def test_starts_at_min_lr(self):
        """Schedule starts at min_lr."""
        schedule = warmup_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

        assert_trees_all_close(schedule(0), 1e-6, atol=1e-10)

    def test_reaches_peak_at_warmup_end(self):
        """Schedule reaches peak_lr at end of warmup."""
        schedule = warmup_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

        # At step 100 (end of warmup), should be at peak
        assert_trees_all_close(schedule(100), 1e-3, atol=1e-8)

    def test_ends_at_min_lr(self):
        """Schedule ends at min_lr after full decay."""
        schedule = warmup_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

        # At final step, should be back to min_lr
        assert_trees_all_close(schedule(1000), 1e-6, atol=1e-8)

    def test_monotonic_warmup(self):
        """LR increases monotonically during warmup."""
        schedule = warmup_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

        lrs = [schedule(i) for i in range(0, 101, 10)]
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1], f"Not increasing at step {i * 10}"

    def test_monotonic_decay(self):
        """LR decreases monotonically during decay phase."""
        schedule = warmup_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

        lrs = [schedule(i) for i in range(100, 1001, 100)]
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i + 1], f"Not decreasing at step {100 + i * 100}"


class TestWarmupPlateauCosineSchedule:
    """Tests for warmup_plateau_cosine_schedule."""

    def test_starts_at_min_lr(self):
        """Schedule starts at min_lr."""
        schedule = warmup_plateau_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            plateau_steps=200,
            min_lr=1e-6,
        )

        assert_trees_all_close(schedule(0), 1e-6, atol=1e-10)

    def test_reaches_peak_at_warmup_end(self):
        """Schedule reaches peak_lr at end of warmup."""
        schedule = warmup_plateau_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            plateau_steps=200,
            min_lr=1e-6,
        )

        assert_trees_all_close(schedule(100), 1e-3, atol=1e-8)

    def test_stays_at_peak_during_plateau(self):
        """Schedule stays at peak_lr during plateau phase."""
        schedule = warmup_plateau_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            plateau_steps=200,
            min_lr=1e-6,
        )

        # Check multiple points during plateau (steps 100-300)
        for step in [100, 150, 200, 250, 299]:
            assert_trees_all_close(schedule(step), 1e-3, atol=1e-8)

    def test_starts_decay_after_plateau(self):
        """Schedule begins decay after plateau ends."""
        schedule = warmup_plateau_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            plateau_steps=200,
            min_lr=1e-6,
        )

        # Step 300 is end of plateau, 301 should start decay
        lr_end_plateau = schedule(300)
        lr_start_decay = schedule(350)

        assert lr_start_decay < lr_end_plateau

    def test_ends_at_min_lr(self):
        """Schedule ends at min_lr."""
        schedule = warmup_plateau_cosine_schedule(
            peak_lr=1e-3,
            total_steps=1000,
            warmup_steps=100,
            plateau_steps=200,
            min_lr=1e-6,
        )

        assert_trees_all_close(schedule(1000), 1e-6, atol=1e-8)


class TestFreezeParamsByPattern:
    """Tests for freeze_params_by_pattern."""

    @pytest.fixture
    def simple_params(self):
        """Create simple nested param structure."""
        return {
            "encoder": {
                "layer1": {"kernel": jnp.ones((3, 3)), "bias": jnp.ones((3,))},
                "layer2": {"kernel": jnp.ones((3, 3)), "bias": jnp.ones((3,))},
            },
            "decoder": {
                "layer1": {"kernel": jnp.ones((3, 3)), "bias": jnp.ones((3,))},
            },
            "head": {"kernel": jnp.ones((3, 1)), "bias": jnp.ones((1,))},
        }

    def test_freezes_matching_params(self, simple_params):
        """Gradients for frozen params are zeroed."""
        base_opt = optax.sgd(learning_rate=0.1)
        frozen_opt = freeze_params_by_pattern(
            base_opt, simple_params, freeze_patterns=["encoder"]
        )

        # Create gradients (all ones)
        grads = jax.tree.map(jnp.ones_like, simple_params)

        # Initialize and apply
        opt_state = frozen_opt.init(simple_params)
        updates, _ = frozen_opt.update(grads, opt_state, simple_params)

        # Encoder updates should be zero
        assert_trees_all_close(
            updates["encoder"]["layer1"]["kernel"], jnp.zeros((3, 3))
        )
        assert_trees_all_close(
            updates["encoder"]["layer2"]["kernel"], jnp.zeros((3, 3))
        )

        # Decoder and head should have non-zero updates
        assert not jnp.allclose(updates["decoder"]["layer1"]["kernel"], 0.0)
        assert not jnp.allclose(updates["head"]["kernel"], 0.0)

    def test_multiple_patterns(self, simple_params):
        """Multiple patterns freeze their respective params."""
        base_opt = optax.sgd(learning_rate=0.1)
        frozen_opt = freeze_params_by_pattern(
            base_opt, simple_params, freeze_patterns=["encoder", "head"]
        )

        grads = jax.tree.map(jnp.ones_like, simple_params)
        opt_state = frozen_opt.init(simple_params)
        updates, _ = frozen_opt.update(grads, opt_state, simple_params)

        # Encoder and head frozen
        assert_trees_all_close(
            updates["encoder"]["layer1"]["kernel"], jnp.zeros((3, 3))
        )
        assert_trees_all_close(updates["head"]["kernel"], jnp.zeros((3, 1)))

        # Only decoder trainable
        assert not jnp.allclose(updates["decoder"]["layer1"]["kernel"], 0.0)

    def test_partial_path_match(self, simple_params):
        """Pattern matches anywhere in path."""
        base_opt = optax.sgd(learning_rate=0.1)
        # "layer1" appears in both encoder and decoder
        frozen_opt = freeze_params_by_pattern(
            base_opt, simple_params, freeze_patterns=["layer1"]
        )

        grads = jax.tree.map(jnp.ones_like, simple_params)
        opt_state = frozen_opt.init(simple_params)
        updates, _ = frozen_opt.update(grads, opt_state, simple_params)

        # layer1 in both encoder and decoder should be frozen
        assert_trees_all_close(
            updates["encoder"]["layer1"]["kernel"], jnp.zeros((3, 3))
        )
        assert_trees_all_close(
            updates["decoder"]["layer1"]["kernel"], jnp.zeros((3, 3))
        )

        # layer2 in encoder should be trainable
        assert not jnp.allclose(updates["encoder"]["layer2"]["kernel"], 0.0)

    def test_no_patterns_trains_all(self, simple_params):
        """Empty pattern list trains all params."""
        base_opt = optax.sgd(learning_rate=0.1)
        frozen_opt = freeze_params_by_pattern(
            base_opt, simple_params, freeze_patterns=[]
        )

        grads = jax.tree.map(jnp.ones_like, simple_params)
        opt_state = frozen_opt.init(simple_params)
        updates, _ = frozen_opt.update(grads, opt_state, simple_params)

        # All params should have non-zero updates
        def check_nonzero(x):
            assert not jnp.allclose(x, 0.0)

        jax.tree.map(check_nonzero, updates)

    def test_preserves_optimizer_behavior(self, simple_params):
        """Unfrozen params behave like base optimizer."""
        lr = 0.1
        base_opt = optax.sgd(learning_rate=lr)
        frozen_opt = freeze_params_by_pattern(
            base_opt, simple_params, freeze_patterns=["encoder"]
        )

        grads = jax.tree.map(jnp.ones_like, simple_params)

        # Base optimizer on decoder
        base_state = base_opt.init(simple_params["decoder"])
        base_updates, _ = base_opt.update(
            grads["decoder"], base_state, simple_params["decoder"]
        )

        # Frozen optimizer on full params
        frozen_state = frozen_opt.init(simple_params)
        frozen_updates, _ = frozen_opt.update(grads, frozen_state, simple_params)

        # Decoder updates should match
        assert_trees_all_close(frozen_updates["decoder"], base_updates)
