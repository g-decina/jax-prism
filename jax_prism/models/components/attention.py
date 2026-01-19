from flax import linen as nn
import jax
import jax.numpy as jnp

from jax_prism._typing import Array
from jax_prism.models.components.positional import apply_rope, build_rope_frequencies

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (covers MHA, GQA, MQA).
    
    Attributes:
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads. 
                    = num_heads → MHA
                    = 1 → MQA  
                    = num_heads // k → GQA
        head_dim: Dimension per head.
        dropout_rate: Attention dropout (training only).
        use_rope: Whether to apply rotary position embeddings.
        rope_base: Base for RoPE frequency computation.
        max_seq_len: Maximum sequence length (for RoPE precomputation).
    """
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    use_rope: bool = True
    rope_base: float = 10_000.0
    max_seq_len: int = 2048
    causal: bool = False
    
    def setup(self):
        # num_heads must be divisible by num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0

        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False)
        self.k_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False)
        self.out_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False)

        if self.use_rope:
            # Precompute RoPE frequencies
            def init_rope():
                return build_rope_frequencies(self.head_dim, self.max_seq_len, self.rope_base)
            # Store as module variable (not parameter - no gradient)
            rope = self.variable("cache", "rope_freqs", init_rope)
            self.rope_cos, self.rope_sin = rope.value

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(rate=self.dropout_rate)
            
    def __call__(
        self,
        x: Array,
        context: Array | None = None,
        mask: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        """
        Args:
            x: Query input, shape (batch, seq_len, d_model)
            context: Key/Value source. If None, self-attention (K,V from x).
            mask: Additional mask, shape broadcastable to (batch, 1, q_len, kv_len).
                True = attend, False = mask out.
            deterministic: If True, disable dropout.
        
        Returns:
            Output of shape (batch, seq_len, num_heads * head_dim)
        """
        # 1. Self-attention vs cross-attention
        if context is None:
            context = x

        batch, q_len, _ = x.shape
        _, kv_len, _ = context.shape

        # 2. Project to Q, K, V
        q = self.q_proj(x)        # (batch, q_len, num_heads * head_dim)
        k = self.k_proj(context)  # (batch, kv_len, num_kv_heads * head_dim)
        v = self.v_proj(context)  # (batch, kv_len, num_kv_heads * head_dim)

        # 3. Reshape to (batch, heads, seq_len, head_dim)
        q = q.reshape(batch, q_len, self.num_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)  # (batch, num_heads, q_len, head_dim)

        k = k.reshape(batch, kv_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch, num_kv_heads, kv_len, head_dim)

        v = v.reshape(batch, kv_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch, num_kv_heads, kv_len, head_dim)

        # 4. Apply RoPE (before head expansion, on Q and K only)
        #    Slice frequencies to actual sequence length
        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:q_len], self.rope_sin[:q_len])
            k = apply_rope(k, self.rope_cos[:kv_len], self.rope_sin[:kv_len])

        # 5. Expand KV heads for GQA: (batch, num_kv_heads, ...) -> (batch, num_heads, ...)
        #    Each KV head is shared by (num_heads // num_kv_heads) query heads
        #    Use jnp.repeat, NOT jnp.tile (repeat along head axis)
        num_head_groups = self.num_heads // self.num_kv_heads
        if num_head_groups > 1:
            k = jnp.repeat(k, num_head_groups, axis=1)
            v = jnp.repeat(v, num_head_groups, axis=1)

        # 6. Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        # Shape: (batch, num_heads, q_len, kv_len)

        # 7. Apply causal mask (if enabled)
        #    Mask out positions where query attends to future keys
        if self.causal:
            # Create causal mask: True where q_pos >= k_pos
            causal_mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=bool))
            mask_value = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(causal_mask, attn_weights, mask_value)

        # 8. Apply provided mask (True = keep, False = mask)
        if mask is not None:
            mask_value = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(mask, attn_weights, mask_value)

        # 9. Softmax over key dimension
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # 10. Dropout (training only)
        if not deterministic and self.dropout_rate > 0.0:
            attn_weights = self.dropout(attn_weights, deterministic=False)

        # 11. Weighted sum of values
        out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        # Shape: (batch, num_heads, q_len, head_dim)

        # 12. Reshape back: (batch, num_heads, q_len, head_dim) -> (batch, q_len, num_heads * head_dim)
        out = out.transpose(0, 2, 1, 3)  # (batch, q_len, num_heads, head_dim)
        out = out.reshape(batch, q_len, self.num_heads * self.head_dim)

        # 13. Output projection
        return self.out_proj(out)