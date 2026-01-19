import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import struct

from jax_prism._typing import Array
from jax_prism.models.components.normalization import RMSNorm

@struct.dataclass
class VSNOutput:
    """Output of Variable Selection Network.

    Attributes:
        selected: Weighted combination of features, shape (B, T, hidden_size).
        weights: Feature importance weights, shape (B, T, num_features).
            Sums to 1 along the feature dimension. Use for interpretability.
    """

    selected: Array
    weights: Array

# ======================================================
# GRN (Gated Residual Network)
# ======================================================

class GRN(nn.Module):
    """
    Gated Residual Network.
    The atomic unit of the TFT.
    """
    # Config: Hidden dimension size, dropout rate
    hidden_size: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None, deterministic: bool = True):
        input_dim = x.shape[-1]
        
        # Primary projection
        hidden = nn.Dense(self.hidden_size, name="fc1")(x)
        
        # Optional: Context injection
        if context is not None:
            c_proj = nn.Dense(self.hidden_size, use_bias=False, name="context_proj")(context)
            hidden = hidden + c_proj
        
        out = nn.elu(hidden)
        
        # Gated Linear Unit: split between value and gate, apply sigmoid to gate
        glu_in = nn.Dense(self.hidden_size * 2, name="fc2")(out)
        glu_in = nn.Dropout(rate = self.dropout_rate)(glu_in, deterministic=deterministic)
        value, gate = jnp.split(glu_in, 2, axis=-1)
        hidden = value * nn.sigmoid(gate)
        
        # Project if dimension mismatch
        if input_dim != self.hidden_size:
            skip = nn.Dense(self.hidden_size, use_bias=False, name="skip_proj")(x)
        else:
            skip = x
        # Residual Connection + Norm
        return RMSNorm()(skip + hidden)

# ======================================================
# VSN (Variable Selection Network)
# ======================================================

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for TFT.
    
    Computes softmax attention weights over input features
    and returns a weighted combination. Provides interpretability
    via feature weights.
    
    Attributes:
        hidden_size: hidden dimension.
        num_features: Number of input features (must be known at init).
        dropout_rate: Dropout probability.
    """
    
    hidden_size: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(
        self, 
        features: Array, # (B, T, num_features, input_dim)
        context: Array | None = None, 
        deterministic: bool = True
    ) -> VSNOutput:
        batch, time, num_features, input_dim = features.shape
        
        # 1. Project all features to hidden_size
        # Approach: (B, T, N, D) → (B * T * N, D) → Dense → (B, T, N, H)
        flat = features.reshape(-1, input_dim)
        projected = nn.Dense(self.hidden_size, name="input_proj")(flat)
        projected = projected.reshape(batch, time, num_features, self.hidden_size)
        
        # 2. Calculate Feature Weights via GRN
        flat_for_weights = projected.reshape(batch, time, -1) # (B, T, N * H)
        weight_hidden = GRN(
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            name="weight_grn"
        )(flat_for_weights, context=context, deterministic=deterministic)
        
        weights = nn.Dense(num_features, name="weight_proj")(weight_hidden)
        weights = nn.softmax(weights, axis=-1)  # (B, T, N)
        
        # 3. Process individual features through shared GRN (vmapped)
        shared_grn = GRN(
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            name="shared_feature_grn"
        ) # (B, T, H)  per slice
        
        def apply_grn(feat_slice):
            return shared_grn(feat_slice, context=context, deterministic=deterministic)
        
        processed = jax.vmap(apply_grn, in_axes=2, out_axes=2)(projected) # (B, T, N, H)
        
        # 4. Weighted Sum
        selected = jnp.einsum("btnh, btn->bth", processed, weights)
        
        return VSNOutput(selected=selected, weights=weights) # Embedding for the model, Weights for the analyst