"""Tests for TFT model and config."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.models.base import ForecastModel
from jax_prism.models.tft.config import ParamHeadConfig, TFTConfig
from jax_prism.models.tft.model import StaticContextGenerator, TemporalFusionTransformer


class TestTFTConfig:
    """Tests for TFTConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = TFTConfig()

        assert config.hidden_size == 64
        assert config.num_heads == 4
        assert config.num_kv_heads == 4  # Defaults to num_heads (MHA)
        assert config.dropout_rate == 0.1

    def test_num_kv_heads_defaults_to_num_heads(self):
        """num_kv_heads should default to num_heads if not specified."""
        config = TFTConfig(num_heads=8)
        assert config.num_kv_heads == 8

    def test_explicit_num_kv_heads(self):
        """Can set num_kv_heads explicitly for GQA/MQA."""
        config = TFTConfig(num_heads=8, num_kv_heads=2)
        assert config.num_kv_heads == 2

    def test_mqa_config(self):
        """Can set num_kv_heads=1 for MQA."""
        config = TFTConfig(num_heads=4, num_kv_heads=1)
        assert config.num_kv_heads == 1

    def test_invalid_hidden_size(self):
        """hidden_size must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            TFTConfig(hidden_size=63, num_heads=4)

    def test_invalid_kv_heads(self):
        """num_heads must be divisible by num_kv_heads."""
        with pytest.raises(ValueError, match="divisible"):
            TFTConfig(num_heads=8, num_kv_heads=3)

    def test_head_dim(self):
        """head_dim property should compute correctly."""
        config = TFTConfig(hidden_size=128, num_heads=8)
        assert config.head_dim == 16

    def test_total_sequence_length(self):
        """total_sequence_length should sum encoder and decoder lengths."""
        config = TFTConfig(encoder_length=168, decoder_length=24)
        assert config.total_sequence_length == 192

    def test_num_encoder_features(self):
        """num_encoder_features should include target + known + observed."""
        config = TFTConfig(num_known_features=3, num_observed_features=2)
        # 1 (target) + 3 (known) + 2 (observed) = 6
        assert config.num_encoder_features == 6

    def test_num_decoder_features(self):
        """num_decoder_features should equal num_known_features."""
        config = TFTConfig(num_known_features=5)
        assert config.num_decoder_features == 5

    def test_frozen(self):
        """Config should be frozen (immutable)."""
        config = TFTConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.hidden_size = 128


class TestStaticContextGenerator:
    """Tests for StaticContextGenerator."""

    def test_output_shapes(self, rng_key):
        """Should produce 4 context vectors of correct shape."""
        config = TFTConfig(hidden_size=64, num_static_features=3)
        model = StaticContextGenerator(config)

        static_features = jax.random.normal(rng_key, (2, 3, 1))  # (B, N_static, 1)

        params = model.init(rng_key, static_features)
        c_s, c_e, c_h, c_c = model.apply(params, static_features)

        chex.assert_shape(c_s, (2, 64))
        chex.assert_shape(c_e, (2, 64))
        chex.assert_shape(c_h, (2, 64))
        chex.assert_shape(c_c, (2, 64))

    def test_different_contexts(self, rng_key):
        """4 context vectors should be different (from different GRNs)."""
        config = TFTConfig(hidden_size=64, num_static_features=3)
        model = StaticContextGenerator(config)

        static_features = jax.random.normal(rng_key, (2, 3, 1))

        params = model.init(rng_key, static_features)
        c_s, c_e, c_h, c_c = model.apply(params, static_features)

        # All should be different
        assert not jnp.allclose(c_s, c_e)
        assert not jnp.allclose(c_e, c_h)
        assert not jnp.allclose(c_h, c_c)

    def test_jit_compatible(self, rng_key):
        """Should be JIT-compilable."""
        config = TFTConfig(hidden_size=64, num_static_features=3)
        model = StaticContextGenerator(config)

        static_features = jax.random.normal(rng_key, (2, 3, 1))

        params = model.init(rng_key, static_features)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x)

        c_s, c_e, c_h, c_c = forward(params, static_features)
        chex.assert_shape(c_s, (2, 64))


class TestTFTBasic:
    """Basic tests for TemporalFusionTransformer."""

    @pytest.fixture
    def small_config(self):
        """Small config for fast tests."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            num_lstm_layers=1,
            dropout_rate=0.0,
            num_static_features=0,
            num_known_features=2,
            num_observed_features=1,
            num_output_params=2,
            encoder_length=16,
            decoder_length=4,
        )

    @pytest.fixture
    def minimal_batch(self, rng_key, small_config):
        """Minimal batch for testing."""
        batch_size = 2
        return TimeSeriesBatch(
            past_targets=jax.random.normal(rng_key, (batch_size, small_config.encoder_length, 1)),
            future_targets=jax.random.normal(rng_key, (batch_size, small_config.decoder_length, 1)),
            past_known_covariates=jax.random.normal(
                rng_key, (batch_size, small_config.encoder_length, small_config.num_known_features)
            ),
            future_known_covariates=jax.random.normal(
                rng_key, (batch_size, small_config.decoder_length, small_config.num_known_features)
            ),
            past_observed_covariates=jax.random.normal(
                rng_key, (batch_size, small_config.encoder_length, small_config.num_observed_features)
            ),
        )

    def test_output_shape(self, rng_key, small_config, minimal_batch):
        """TFT output should have correct shape."""
        model = TemporalFusionTransformer(small_config)

        params = model.init(rng_key, minimal_batch)
        output = model.apply(params, minimal_batch)

        # (B, decoder_length, num_output_params)
        chex.assert_shape(output, (2, small_config.decoder_length, small_config.num_output_params))

    def test_output_finite(self, rng_key, small_config, minimal_batch):
        """Output should not contain NaN or Inf."""
        model = TemporalFusionTransformer(small_config)

        params = model.init(rng_key, minimal_batch)
        output = model.apply(params, minimal_batch)

        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))

    def test_deterministic_inference(self, rng_key, small_config, minimal_batch):
        """Inference mode should be deterministic."""
        model = TemporalFusionTransformer(small_config)

        params = model.init(rng_key, minimal_batch)

        output1 = model.apply(params, minimal_batch, training=False)
        output2 = model.apply(params, minimal_batch, training=False)

        chex.assert_trees_all_close(output1, output2, atol=0)


class TestTFTWithStatic:
    """Tests for TFT with static features."""

    @pytest.fixture
    def config_with_static(self):
        """Config with static features."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            num_lstm_layers=1,
            dropout_rate=0.0,
            num_static_features=2,
            num_known_features=2,
            num_observed_features=1,
            num_output_params=2,
            encoder_length=16,
            decoder_length=4,
        )

    @pytest.fixture
    def batch_with_static(self, rng_key, config_with_static):
        """Batch with static features."""
        cfg = config_with_static
        batch_size = 2
        return TimeSeriesBatch(
            past_targets=jax.random.normal(rng_key, (batch_size, cfg.encoder_length, 1)),
            future_targets=jax.random.normal(rng_key, (batch_size, cfg.decoder_length, 1)),
            past_known_covariates=jax.random.normal(
                rng_key, (batch_size, cfg.encoder_length, cfg.num_known_features)
            ),
            future_known_covariates=jax.random.normal(
                rng_key, (batch_size, cfg.decoder_length, cfg.num_known_features)
            ),
            past_observed_covariates=jax.random.normal(
                rng_key, (batch_size, cfg.encoder_length, cfg.num_observed_features)
            ),
            static_covariates=jax.random.normal(rng_key, (batch_size, cfg.num_static_features)),
        )

    def test_with_static_features(self, rng_key, config_with_static, batch_with_static):
        """TFT should work with static features."""
        model = TemporalFusionTransformer(config_with_static)

        params = model.init(rng_key, batch_with_static)
        output = model.apply(params, batch_with_static)

        chex.assert_shape(
            output,
            (2, config_with_static.decoder_length, config_with_static.num_output_params),
        )

    def test_static_affects_output(self, rng_key, config_with_static, batch_with_static):
        """Different static features should produce different outputs."""
        model = TemporalFusionTransformer(config_with_static)

        params = model.init(rng_key, batch_with_static)
        output1 = model.apply(params, batch_with_static)

        # Modify static features
        batch2 = batch_with_static.replace(
            static_covariates=batch_with_static.static_covariates * 2
        )
        output2 = model.apply(params, batch2)

        assert not jnp.allclose(output1, output2)


class TestTFTMinimal:
    """Tests for TFT with minimal inputs."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal config (no covariates)."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            num_lstm_layers=1,
            dropout_rate=0.0,
            num_static_features=0,
            num_known_features=0,
            num_observed_features=0,
            num_output_params=2,
            encoder_length=16,
            decoder_length=4,
        )

    @pytest.fixture
    def target_only_batch(self, rng_key, minimal_config):
        """Batch with only targets (no covariates)."""
        batch_size = 2
        return TimeSeriesBatch(
            past_targets=jax.random.normal(rng_key, (batch_size, minimal_config.encoder_length, 1)),
            future_targets=jax.random.normal(rng_key, (batch_size, minimal_config.decoder_length, 1)),
        )

    def test_target_only(self, rng_key, minimal_config, target_only_batch):
        """TFT should work with only target values (no covariates)."""
        model = TemporalFusionTransformer(minimal_config)

        params = model.init(rng_key, target_only_batch)
        output = model.apply(params, target_only_batch)

        chex.assert_shape(
            output,
            (2, minimal_config.decoder_length, minimal_config.num_output_params),
        )


class TestTFTJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    @pytest.fixture
    def config(self):
        """Small config for JAX tests."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            num_lstm_layers=1,
            dropout_rate=0.0,
            num_static_features=0,
            num_known_features=2,
            num_observed_features=0,
            num_output_params=2,
            encoder_length=8,
            decoder_length=4,
        )

    @pytest.fixture
    def batch(self, rng_key, config):
        """Test batch."""
        batch_size = 2
        return TimeSeriesBatch(
            past_targets=jax.random.normal(rng_key, (batch_size, config.encoder_length, 1)),
            future_targets=jax.random.normal(rng_key, (batch_size, config.decoder_length, 1)),
            past_known_covariates=jax.random.normal(
                rng_key, (batch_size, config.encoder_length, config.num_known_features)
            ),
            future_known_covariates=jax.random.normal(
                rng_key, (batch_size, config.decoder_length, config.num_known_features)
            ),
        )

    def test_jit_compatible(self, rng_key, config, batch):
        """TFT should be JIT-compilable."""
        model = TemporalFusionTransformer(config)

        params = model.init(rng_key, batch)

        @jax.jit
        def forward(params, batch):
            return model.apply(params, batch, training=False)

        output = forward(params, batch)
        chex.assert_shape(output, (2, config.decoder_length, config.num_output_params))

    def test_gradient_flow(self, rng_key, config, batch):
        """Gradients should flow through TFT."""
        model = TemporalFusionTransformer(config)

        params = model.init(rng_key, batch)

        def loss_fn(params):
            output = model.apply(params, batch, training=False)
            return jnp.mean(output**2)

        grads = jax.grad(loss_fn)(params)

        # Check gradients exist and are non-zero for key layers
        assert "encoder_vsn" in grads["params"]
        assert "output_proj" in grads["params"]

        # Output projection should have non-zero gradients
        output_grads = grads["params"]["output_proj"]["kernel"]
        assert not jnp.allclose(output_grads, 0)


class TestTFTTrainingMode:
    """Tests for TFT training mode behavior."""

    @pytest.fixture
    def config_with_dropout(self):
        """Config with dropout enabled."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            num_lstm_layers=1,
            dropout_rate=0.5,  # High dropout
            attention_dropout_rate=0.5,
            num_static_features=0,
            num_known_features=2,
            num_observed_features=0,
            num_output_params=2,
            encoder_length=8,
            decoder_length=4,
        )

    @pytest.fixture
    def batch(self, rng_key, config_with_dropout):
        """Test batch."""
        batch_size = 2
        cfg = config_with_dropout
        return TimeSeriesBatch(
            past_targets=jax.random.normal(rng_key, (batch_size, cfg.encoder_length, 1)),
            future_targets=jax.random.normal(rng_key, (batch_size, cfg.decoder_length, 1)),
            past_known_covariates=jax.random.normal(
                rng_key, (batch_size, cfg.encoder_length, cfg.num_known_features)
            ),
            future_known_covariates=jax.random.normal(
                rng_key, (batch_size, cfg.decoder_length, cfg.num_known_features)
            ),
        )

    def test_training_mode_stochastic(self, rng_key, config_with_dropout, batch):
        """Training mode should produce different outputs with different RNG."""
        model = TemporalFusionTransformer(config_with_dropout)

        params = model.init(rng_key, batch)

        output1 = model.apply(
            params, batch, training=True, rngs={"dropout": jax.random.key(0)}
        )
        output2 = model.apply(
            params, batch, training=True, rngs={"dropout": jax.random.key(1)}
        )

        # Different RNG should produce different outputs with dropout
        assert not jnp.allclose(output1, output2)


class TestForecastModelProtocol:
    """Tests for ForecastModel protocol conformance."""

    def test_tft_is_forecast_model(self):
        """TFT should be a ForecastModel (runtime checkable)."""
        config = TFTConfig(hidden_size=32, num_heads=2)
        model = TemporalFusionTransformer(config)

        # Protocol is runtime_checkable, so isinstance works
        assert isinstance(model, ForecastModel)

    def test_has_num_output_params(self):
        """TFT should have num_output_params property."""
        config = TFTConfig(hidden_size=32, num_heads=2, num_output_params=3)
        model = TemporalFusionTransformer(config)

        assert model.num_output_params == 3

    def test_has_decoder_length(self):
        """TFT should have decoder_length property."""
        config = TFTConfig(hidden_size=32, num_heads=2, decoder_length=12)
        model = TemporalFusionTransformer(config)

        assert model.decoder_length == 12


class TestParamHeadConfigs:
    """Tests for per-parameter output heads."""

    @pytest.fixture
    def config_separate_heads(self):
        """Config with separate output heads for Gaussian (2 params)."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            dropout_rate=0.0,
            num_known_features=2,
            num_output_params=2,
            encoder_length=16,
            decoder_length=4,
            param_head_configs=(
                ParamHeadConfig(),  # μ: inherit defaults
                ParamHeadConfig(dropout_rate=0.3),  # σ: higher dropout
            ),
        )

    @pytest.fixture
    def config_asymmetric_heads(self):
        """Config with asymmetric head sizes."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            dropout_rate=0.1,
            num_known_features=2,
            num_output_params=2,
            encoder_length=16,
            decoder_length=4,
            param_head_configs=(
                ParamHeadConfig(),  # μ: default hidden_size=32
                ParamHeadConfig(hidden_size=16, dropout_rate=0.3),  # σ: smaller
            ),
        )

    @pytest.fixture
    def config_three_params(self):
        """Config for 3-parameter distribution (e.g., Student-t)."""
        return TFTConfig(
            hidden_size=32,
            num_heads=2,
            dropout_rate=0.0,
            num_known_features=2,
            num_output_params=3,
            encoder_length=16,
            decoder_length=4,
            param_head_configs=(
                ParamHeadConfig(),  # μ
                ParamHeadConfig(dropout_rate=0.3),  # σ
                ParamHeadConfig(hidden_size=16),  # ν
            ),
        )

    @pytest.fixture
    def minimal_batch(self, rng_key):
        """Minimal batch for testing."""
        batch_size = 2
        encoder_length = 16
        decoder_length = 4
        num_known = 2
        return TimeSeriesBatch(
            past_targets=jax.random.normal(rng_key, (batch_size, encoder_length, 1)),
            future_targets=jax.random.normal(rng_key, (batch_size, decoder_length, 1)),
            past_known_covariates=jax.random.normal(
                rng_key, (batch_size, encoder_length, num_known)
            ),
            future_known_covariates=jax.random.normal(
                rng_key, (batch_size, decoder_length, num_known)
            ),
        )

    def test_output_shape_separate_heads(self, rng_key, config_separate_heads, minimal_batch):
        """Separate heads should produce same output shape as single head."""
        model = TemporalFusionTransformer(config_separate_heads)
        params = model.init(rng_key, minimal_batch)
        output = model.apply(params, minimal_batch)

        chex.assert_shape(output, (2, config_separate_heads.decoder_length, 2))

    def test_output_shape_three_params(self, rng_key, config_three_params, minimal_batch):
        """3-parameter config should produce correct output shape."""
        model = TemporalFusionTransformer(config_three_params)
        params = model.init(rng_key, minimal_batch)
        output = model.apply(params, minimal_batch)

        chex.assert_shape(output, (2, config_three_params.decoder_length, 3))

    def test_separate_params_exist(self, rng_key, config_separate_heads, minimal_batch):
        """Separate heads should have separate parameters."""
        model = TemporalFusionTransformer(config_separate_heads)
        params = model.init(rng_key, minimal_batch)

        # Should have param_0_gate and param_1_gate
        assert "param_0_gate" in params["params"]
        assert "param_1_gate" in params["params"]
        assert "param_0_proj" in params["params"]
        assert "param_1_proj" in params["params"]

        # Should NOT have single output head
        assert "output_gate" not in params["params"]
        assert "output_proj" not in params["params"]

    def test_asymmetric_hidden_size(self, rng_key, config_asymmetric_heads, minimal_batch):
        """Asymmetric head hidden sizes should work and add input projection."""
        model = TemporalFusionTransformer(config_asymmetric_heads)
        params = model.init(rng_key, minimal_batch)

        # param_1 (σ) should have input projection due to hidden_size=16
        assert "param_1_input_proj" in params["params"]

        # param_0 (μ) should not have input projection (uses default hidden_size)
        assert "param_0_input_proj" not in params["params"]

        # σ gate should have smaller dimension
        sigma_gate_kernel = params["params"]["param_1_gate"]["fc1"]["kernel"]
        assert sigma_gate_kernel.shape[1] == 16  # hidden_size for σ

    def test_backward_compatible_default(self, rng_key, minimal_batch):
        """Default config (no param_head_configs) should use single head."""
        config = TFTConfig(
            hidden_size=32,
            num_heads=2,
            dropout_rate=0.0,
            num_known_features=2,
            num_output_params=2,
            encoder_length=16,
            decoder_length=4,
            # param_head_configs=None (default)
        )
        model = TemporalFusionTransformer(config)
        params = model.init(rng_key, minimal_batch)

        # Should have old-style single head
        assert "output_gate" in params["params"]
        assert "output_proj" in params["params"]

        # Should NOT have separate heads
        assert "param_0_gate" not in params["params"]
        assert "param_1_gate" not in params["params"]

    def test_gradient_flow_all_heads(self, rng_key, config_separate_heads, minimal_batch):
        """Gradients should flow through all parameter heads."""
        model = TemporalFusionTransformer(config_separate_heads)
        params = model.init(rng_key, minimal_batch)

        def loss_fn(params):
            output = model.apply(params, minimal_batch)
            # Loss that uses all outputs
            return jnp.mean(output**2)

        grads = jax.grad(loss_fn)(params)

        # Both heads should have non-zero gradients
        grad_0 = grads["params"]["param_0_gate"]["fc1"]["kernel"]
        grad_1 = grads["params"]["param_1_gate"]["fc1"]["kernel"]
        assert not jnp.allclose(grad_0, 0)
        assert not jnp.allclose(grad_1, 0)

    def test_jit_compatible_separate_heads(self, rng_key, config_separate_heads, minimal_batch):
        """Separate heads should be JIT-compilable."""
        model = TemporalFusionTransformer(config_separate_heads)
        params = model.init(rng_key, minimal_batch)

        @jax.jit
        def forward(params, batch):
            return model.apply(params, batch)

        output = forward(params, minimal_batch)
        chex.assert_shape(output, (2, config_separate_heads.decoder_length, 2))

    def test_invalid_param_head_configs_length(self):
        """param_head_configs total output_dims must match num_output_params."""
        with pytest.raises(ValueError, match="param_head_configs output_dims"):
            TFTConfig(
                hidden_size=32,
                num_heads=2,
                num_output_params=2,
                param_head_configs=(
                    ParamHeadConfig(),
                    ParamHeadConfig(),
                    ParamHeadConfig(),  # 3 configs but num_output_params=2
                ),
            )

    def test_output_finite_separate_heads(self, rng_key, config_separate_heads, minimal_batch):
        """Output should not contain NaN or Inf with separate heads."""
        model = TemporalFusionTransformer(config_separate_heads)
        params = model.init(rng_key, minimal_batch)
        output = model.apply(params, minimal_batch)

        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))
