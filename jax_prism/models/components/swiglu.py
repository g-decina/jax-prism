"""SwiGLU activation module.

PSEUDOCODE - Guillaume to implement
====================================

SwiGLU (Swish-Gated Linear Unit) from Shazeer 2020.

Math:
    SwiGLU(x) = Swish(x @ W_gate) ⊙ (x @ W_up)

    Where:
    - Swish(z) = z * sigmoid(z)  (aka SiLU)
    - ⊙ is element-wise multiplication
    - W_gate and W_up are separate projections

Architecture:
    input (d_model)
        → gate_proj (d_model → hidden_dim) → Swish
        → up_proj (d_model → hidden_dim)
        → element-wise multiply
        → down_proj (hidden_dim → d_model)
        → output (d_model)

Why three projections?
    Standard FFN: 2 projections (up, down)
    SwiGLU: 3 projections (gate, up, down)

    To keep parameter count equal to standard FFN with 4x expansion,
    use hidden_dim = (2/3) * 4 * d_model ≈ 2.67 * d_model

    In practice, round to nearest multiple of 256 for hardware efficiency.

Implementation notes:
    - Use nn.Dense with use_bias=False (standard for modern transformers)
    - jax.nn.silu IS Swish (they're identical)
    - hidden_dim is a required parameter (caller decides expansion)
"""

from flax import linen as nn

from jax_prism._typing import Array


class SwiGLU(nn.Module):
    """SwiGLU feedforward block.

    Attributes:
        hidden_dim: Intermediate dimension (after gate/up projection).
        use_bias: Whether to use bias in linear layers.
    """
    hidden_dim: int
    use_bias: bool = False
        
    @nn.compact
    def __call__(self, x: Array) -> Array:
        d_model = x.shape[-1]

        gate = nn.Dense(self.hidden_dim, use_bias=self.use_bias, name="gate_proj")(x)
        gate = nn.silu(gate)

        up = nn.Dense(self.hidden_dim, use_bias=self.use_bias, name="up_proj")(x)

        hidden = gate * up

        return nn.Dense(d_model, use_bias=self.use_bias, name="down_proj")(hidden)