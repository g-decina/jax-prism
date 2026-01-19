"""Temporal Fusion Transformer model.

PSEUDOCODE - Guillaume implements based on this specification.

References:
    Lim et al. "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting." IJF 2021.

Architecture Overview:
    1. Static Variable Selection → 4 context vectors (c_s, c_e, c_h, c_c)
    2. Encoder Variable Selection (past: target + known + observed)
    3. Decoder Variable Selection (future: known only)
    4. LSTM Encoder (initialized with c_e, c_h)
    5. LSTM Decoder (initialized with encoder final state)
    6. Static Enrichment (GRN with c_s context)
    7. Temporal Self-Attention (GQA + RoPE, causal for decoder)
    8. Position-wise Feedforward (SwiGLU)
    9. Decoder Gating (final GRN)
    10. Output Projection → distribution parameters
"""

import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_prism._typing import Array
from jax_prism.data.batch import TimeSeriesBatch
from jax_prism.models.components import (
    GroupedQueryAttention,
    RMSNorm,
    SwiGLU,
)
from jax_prism.models.tft.components import GRN, VariableSelectionNetwork
from jax_prism.models.tft.config import ParamHeadConfig, TFTConfig

class StaticContextGenerator(nn.Module):
    """Generate 4 static context vectors from static features.

    c_s: Static enrichment context
    c_e: Encoder LSTM initial hidden state
    c_h: Encoder LSTM initial cell state
    c_c: Decoder LSTM cell state context

    Attributes:
        config: TFTConfig instance.
    """

    config: TFTConfig

    @nn.compact
    def __call__(
        self,
        static_features: Array,  # (B, num_static, 1) or (B, num_static, embed_dim)
        training: bool = False,
    ) -> tuple[Array, Array, Array, Array]:
        """Generate static context vectors.

        Args:
            static_features: Static inputs, shape (B, num_static, input_dim).
            training: Whether in training mode.

        Returns:
            Tuple of (c_s, c_e, c_h, c_c), each shape (B, hidden_size).
        """
        cfg = self.config

        # Variable selection on static features
        # Need to add time dimension for VSN: (B, num_static, D) → (B, 1, num_static, D)
        static_expanded = static_features[:, jnp.newaxis, :, :]  # (B, 1, N, D)

        vsn_out = VariableSelectionNetwork(
            hidden_size=cfg.hidden_size,
            dropout_rate=cfg.dropout_rate,
            name="static_vsn",
        )(static_expanded, context=None, deterministic=not training)

        # Remove time dim: (B, 1, H) → (B, H)
        static_embed = vsn_out.selected[:, 0, :]

        # Generate 4 context vectors via GRNs
        c_s = GRN(cfg.hidden_size, cfg.dropout_rate, name="c_s_grn")(
            static_embed, deterministic=not training
        )
        c_e = GRN(cfg.hidden_size, cfg.dropout_rate, name="c_e_grn")(
            static_embed, deterministic=not training
        )
        c_h = GRN(cfg.hidden_size, cfg.dropout_rate, name="c_h_grn")(
            static_embed, deterministic=not training
        )
        c_c = GRN(cfg.hidden_size, cfg.dropout_rate, name="c_c_grn")(
            static_embed, deterministic=not training
        )

        return c_s, c_e, c_h, c_c


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for multi-horizon forecasting.

    Modernized with RoPE, RMSNorm, SwiGLU, and configurable GQA.

    Attributes:
        config: TFTConfig instance.
    """

    config: TFTConfig

    @property
    def num_output_params(self) -> int:
        """Number of output parameters per timestep."""
        return self.config.num_output_params

    @property
    def decoder_length(self) -> int:
        """Forecast horizon length."""
        return self.config.decoder_length

    def setup(self):
        """Initialize submodules."""
        cfg = self.config

        # Static context generator (only if static features exist)
        if cfg.num_static_features > 0:
            self.static_context = StaticContextGenerator(cfg, name="static_context")

        # Encoder VSN (target + known + observed)
        self.encoder_vsn = VariableSelectionNetwork(
            hidden_size=cfg.hidden_size,
            dropout_rate=cfg.dropout_rate,
            name="encoder_vsn",
        )

        # Decoder VSN (known only)
        if cfg.num_known_features > 0:
            self.decoder_vsn = VariableSelectionNetwork(
                hidden_size=cfg.hidden_size,
                dropout_rate=cfg.dropout_rate,
                name="decoder_vsn",
            )

        # LSTM layers - use nn.scan to unroll LSTMCell over time
        ScanLSTM = nn.scan(
            nn.LSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.encoder_lstm = ScanLSTM(features=cfg.hidden_size, name="encoder_lstm")
        self.decoder_lstm = ScanLSTM(features=cfg.hidden_size, name="decoder_lstm")

        # Locality enhancement (post-LSTM GRN)
        self.encoder_grn = GRN(cfg.hidden_size, cfg.dropout_rate, name="encoder_grn")
        self.decoder_grn = GRN(cfg.hidden_size, cfg.dropout_rate, name="decoder_grn")

        # Static enrichment GRN
        self.static_enrichment = GRN(
            cfg.hidden_size, cfg.dropout_rate, name="static_enrichment"
        )

        # Temporal self-attention
        self.temporal_attention = GroupedQueryAttention(
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.attention_dropout_rate,
            use_rope=True,
            name="temporal_attention",
        )
        self.attention_norm = RMSNorm(name="attention_norm")
        self.attention_gate = GRN(cfg.hidden_size, cfg.dropout_rate, name="attention_gate")

        # Position-wise feedforward (SwiGLU)
        self.feedforward = SwiGLU(
            hidden_dim=cfg.hidden_size * 4,
            use_bias=True,
            name="feedforward",
        )
        self.ff_norm = RMSNorm(name="ff_norm")
        self.ff_dropout = nn.Dropout(rate=cfg.dropout_rate)

        # Output heads
        if cfg.param_head_configs is not None:
            # Separate head per distribution parameter
            # Build lists then convert to tuples (Flax requires immutable)
            param_gates = []
            param_projs = []
            param_input_projs = []

            for i, head_cfg in enumerate(cfg.param_head_configs):
                hidden = head_cfg.hidden_size or cfg.hidden_size
                dropout = (
                    head_cfg.dropout_rate
                    if head_cfg.dropout_rate is not None
                    else cfg.dropout_rate
                )

                param_gates.append(
                    GRN(hidden, dropout, name=f"param_{i}_gate")
                )
                param_projs.append(nn.Dense(1, name=f"param_{i}_proj"))

                # Input projection if this head uses different hidden size
                if hidden != cfg.hidden_size:
                    param_input_projs.append(
                        nn.Dense(hidden, name=f"param_{i}_input_proj")
                    )
                else:
                    param_input_projs.append(None)

            self.param_gates = param_gates
            self.param_projs = param_projs
            self.param_input_projs = param_input_projs
        else:
            # Original single-head path
            self.output_gate = GRN(
                cfg.hidden_size, cfg.dropout_rate, name="output_gate"
            )
            self.output_proj = nn.Dense(cfg.num_output_params, name="output_proj")
        
        # Causal mask
        self.causal_mask = self.variable(
            "cache", "causal_mask",
            lambda: self._build_causal_mask(
                self.config.encoder_length,
                self.config.decoder_length
            )
        )

    def _prepare_encoder_inputs(
        self, batch: TimeSeriesBatch
    ) -> Array:
        """Prepare encoder inputs by stacking features.

        Args:
            batch: TimeSeriesBatch with past data.

        Returns:
            Stacked features, shape (B, T_enc, num_features, 1).
        """
        features = []

        # Target (always present): (B, T, num_targets) → (B, T, num_targets, 1)
        target = batch.past_targets[..., jnp.newaxis]
        features.append(target)

        # Known covariates: (B, T, F) → (B, T, F, 1)
        if batch.past_known_covariates is not None:
            features.append(batch.past_known_covariates[..., jnp.newaxis])

        # Observed covariates: (B, T, F) → (B, T, F, 1)
        if batch.past_observed_covariates is not None:
            features.append(batch.past_observed_covariates[..., jnp.newaxis])
        
        # Stack: (B, T, num_features, 1)
        return jnp.concatenate(features, axis=2)

    def _prepare_decoder_inputs(
        self, batch: TimeSeriesBatch
    ) -> Array | None:
        """Prepare decoder inputs (known future only).

        Args:
            batch: TimeSeriesBatch with future known covariates.

        Returns:
            Stacked features, shape (B, T_dec, num_features, 1), or None.
        """
        if batch.future_known_covariates is None:
            return None

        # (B, T, F) → (B, T, F, 1)
        return batch.future_known_covariates[..., jnp.newaxis]

    def _prepare_static_inputs(
        self, batch: TimeSeriesBatch
    ) -> Array | None:
        """Prepare static inputs.

        Args:
            batch: TimeSeriesBatch with static covariates.

        Returns:
            Static features, shape (B, num_static, 1), or None.
        """
        if batch.static_covariates is None:
            return None

        # (B, F) → (B, F, 1)
        return batch.static_covariates[..., jnp.newaxis]

    def _build_causal_mask(
        self,
        encoder_length: int,
        decoder_length: int,
    ) -> Array:
        """Build attention mask for encoder-decoder self-attention.

        Encoder positions can attend to all encoder positions.
        Decoder positions can attend to all encoder positions and
        causally to decoder positions (including self).

        Args:
            encoder_length: Number of encoder time steps.
            decoder_length: Number of decoder time steps.

        Returns:
            Mask of shape (T_total, T_total). 1 = can attend, 0 = masked.
        """
        total_length = encoder_length + decoder_length

        # Start with all-ones (encoder can attend everywhere in encoder)
        mask = jnp.ones((total_length, total_length))

        # Decoder positions: causal within decoder block
        # Position i in decoder can attend to positions <= i
        decoder_mask = jnp.tril(jnp.ones((decoder_length, decoder_length)))

        # Place decoder mask in bottom-right
        mask = mask.at[encoder_length:, encoder_length:].set(decoder_mask)

        return mask

    def __call__(
        self,
        batch: TimeSeriesBatch,
        training: bool = False,
    ) -> Array:
        """Forward pass.

        Args:
            batch: TimeSeriesBatch with all inputs.
            training: Whether in training mode (affects dropout).

        Returns:
            Output parameters, shape (B, decoder_length, num_output_params).
        """
        cfg = self.config
        batch_size = batch.batch_size

        # =========================================================
        # 1. Prepare inputs
        # =========================================================
        encoder_inputs = self._prepare_encoder_inputs(batch)  # (B, T_enc, N_enc, 1)
        decoder_inputs = self._prepare_decoder_inputs(batch)  # (B, T_dec, N_dec, 1) or None
        static_inputs = self._prepare_static_inputs(batch)    # (B, N_static, 1) or None

        # =========================================================
        # 2. Static context generation
        # =========================================================
        if cfg.num_static_features > 0 and static_inputs is not None:
            c_s, c_e, c_h, c_c = self.static_context(static_inputs, training=training)
        else:
            # No static features: use zeros
            c_s = jnp.zeros((batch_size, cfg.hidden_size))
            c_e = jnp.zeros((batch_size, cfg.hidden_size))
            c_h = jnp.zeros((batch_size, cfg.hidden_size))
            c_c = jnp.zeros((batch_size, cfg.hidden_size))

        # =========================================================
        # 3. Encoder variable selection
        # =========================================================
        context_expanded = c_s[:, jnp.newaxis, :]  # (B, H) → (B, 1, H)
        encoder_vsn_out = self.encoder_vsn(
            encoder_inputs,
            context=context_expanded,
            deterministic=not training,
        )
        encoder_selected = encoder_vsn_out.selected  # (B, T_enc, H)

        # =========================================================
        # 4. Decoder variable selection
        # =========================================================
        if cfg.num_known_features > 0 and decoder_inputs is not None:
            decoder_vsn_out = self.decoder_vsn(
                decoder_inputs,
                context=context_expanded,
                deterministic=not training,
            )
            decoder_selected = decoder_vsn_out.selected  # (B, T_dec, H)
        else:
            # No decoder inputs: use zeros or learned embedding
            decoder_selected = jnp.zeros(
                (batch_size, cfg.decoder_length, cfg.hidden_size)
            )

        # =========================================================
        # 5. LSTM Encoder
        # =========================================================
        encoder_init = (c_h, c_e)  # (cell, hidden)

        encoder_final, encoder_outputs = self.encoder_lstm(encoder_init, encoder_selected)
        # Post-LSTM gating
        encoder_outputs = self.encoder_grn(encoder_outputs, deterministic=not training)

        # =========================================================
        # 6. LSTM Decoder
        # =========================================================
        # Initialize with encoder final state, modulated by c_c
        decoder_cell = encoder_final[0] + c_c  # Add context to cell state
        decoder_init = (decoder_cell, encoder_final[1])

        _, decoder_outputs = self.decoder_lstm(decoder_init, decoder_selected)

        # Post-LSTM gating
        decoder_outputs = self.decoder_grn(decoder_outputs, deterministic=not training)

        # =========================================================
        # 7. Concatenate temporal features
        # =========================================================
        temporal = jnp.concatenate(
            [encoder_outputs, decoder_outputs], axis=1
        )  # (B, T_total, H)

        # =========================================================
        # 8. Static enrichment
        # =========================================================
        # Expand c_s for broadcasting: (B, H) → (B, 1, H) → broadcast to (B, T, H)
        enriched = self.static_enrichment(
            temporal,
            context=c_s[:, jnp.newaxis, :],  # (B, 1, H)
            deterministic=not training,
        )  # (B, T_total, H)

        # =========================================================
        # 9. Temporal self-attention
        # =========================================================
        # Build attention mask
        attn_mask = self.causal_mask.value

        # Apply attention
        attn_out = self.temporal_attention(
            enriched,
            context=None,  # Self-attention
            mask=attn_mask,
            deterministic=not training
        )  # (B, T, H)

        # Gated residual with attention output
        attn_gated = self.attention_gate(
            self.attention_norm(attn_out),
            deterministic=not training,
        )
        temporal = enriched + attn_gated  # Residual

        # =========================================================
        # 10. Position-wise feedforward
        # =========================================================
        ff_out = self.feedforward(temporal)
        ff_out = self.ff_dropout(ff_out, deterministic=not training)
        temporal = self.ff_norm(temporal + ff_out)  # Residual + norm

        # =========================================================
        # 11. Extract decoder outputs and final gating
        # =========================================================
        decoder_temporal = temporal[:, cfg.encoder_length:, :]  # (B, T_dec, H)

        # =========================================================
        # 12. Output projection
        # =========================================================
        if cfg.param_head_configs is not None:
            # Process each parameter through its own head
            param_outputs = []

            for i, (gate, proj, input_proj) in enumerate(
                zip(self.param_gates, self.param_projs, self.param_input_projs)
            ):
                # Optionally project input to head's hidden size
                head_input = decoder_temporal
                skip_input = decoder_outputs
                if input_proj is not None:
                    head_input = input_proj(head_input)
                    skip_input = input_proj(skip_input)

                # GRN + skip connection + projection
                gated = gate(head_input, deterministic=not training)
                hidden = skip_input + gated
                param_out = proj(hidden)  # (B, T_dec, 1)
                param_outputs.append(param_out)

            output = jnp.concatenate(param_outputs, axis=-1)  # (B, T_dec, num_params)
        else:
            # Original single-head path
            decoder_gated = self.output_gate(
                decoder_temporal, deterministic=not training
            )
            output_hidden = decoder_outputs + decoder_gated
            output = self.output_proj(output_hidden)  # (B, T_dec, num_params)

        return output
