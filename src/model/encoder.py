"""
Unified Encoder for the Causal-JEPA World Model.

This encoder processes both text tokens and image patches in a single
transformer, producing a shared latent space where causal structure can
be learned across modalities.

Key design decisions for scalability:
- RMSNorm: ~10% faster than LayerNorm with equivalent quality at all scales
- SwiGLU FFN: consistently better perplexity/parameter across scales (Shazeer 2020)
- RoPE: extrapolates to unseen sequence lengths without retraining
- GQA: reduces KV cache memory linearly, enabling efficient inference at 1B+ params
- Flash Attention compatible: standard attention as fallback; flash_attn drops in
- Gradient checkpointing: trade compute for memory, enabling larger models on same GPU
- KV Cache: O(1) per-token cost for autoregressive generation

What makes this different from a standard transformer:
1. Modality embeddings distinguish text/image/memory tokens (3 types)
2. A gated cross-attention layer at ~60% depth injects episodic memories
3. No pretrained weights — trained from scratch to learn causal structure
4. The same encoder is used for observations, events, AND text generation
5. The KV cache supports all three modes transparently
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash attention integration — optional dependency for production scaling.
# When available, reduces attention memory from O(N²) to O(N), enabling
# much longer sequence lengths at the same VRAM budget.
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ============================================================================
#  Shared Primitives — used by encoder, causal_net, memory, and world_model
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Why RMSNorm over LayerNorm:
    - Removes the mean-centering step, which is empirically unnecessary
    - ~10% faster at training time (one less reduction operation)
    - Identical quality at all tested scales (confirmed by LLaMA, Gemma, PaLM)
    - The computational savings compound at larger scales and longer sequences

    The implementation computes in float32 for numerical stability, then casts
    back to the input dtype. This is critical for mixed-precision training where
    the input may be float16/bfloat16 but the norm needs float32 precision.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability
        input_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(input_dtype) * self.weight


def get_norm(norm_type: str, dim: int) -> nn.Module:
    """
    Factory for normalization layers — swap globally via config.
    This indirection lets us change the norm for the ENTIRE model
    by changing a single config value, which is essential for
    architecture search across scales.
    """
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# ============================================================================
#  Positional Encoding — RoPE (default, scalable) or Learned (legacy)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (Su et al., 2021).

    Why RoPE over learned absolute positions:
    - Extrapolates to sequence lengths not seen during training
    - Encodes relative position information directly in attention scores
    - No additional parameters (rotation angles are deterministic)
    - A model trained at 2K context can be extended to 8K+ with NTK-aware scaling

    The theta parameter controls the frequency base:
    - 10000.0: works well up to ~8K context (default, base/large)
    - 500000.0+: extended context for research-scale models (LLaMA 3 style)

    For mixed modalities (text + image patches), positions are sequential:
    text tokens get positions 0..S_text-1, image patches get S_text..S_total-1.
    RoPE encodes relative distances, so the model learns that adjacent patches
    are spatially close without needing explicit 2D position encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute frequency bands for dim/2 pairs.
        # Each pair of dimensions rotates at a different frequency,
        # encoding position information at multiple scales.
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for common sequence lengths
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache up to seq_len. Extends lazily for longer sequences."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (cos, sin) tensors for positions 0..seq_len-1.
        Shape: (seq_len, dim/2) each.
        Automatically extends cache if seq_len exceeds pre-computed range.
        """
        if seq_len > self.cos_cached.shape[0]:
            # Dynamically extend cache — this is key for scalability:
            # the model can handle ANY sequence length without retraining
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.

    x:   (batch, num_heads, seq_len, head_dim)
    cos: (seq_len, head_dim/2)
    sin: (seq_len, head_dim/2)

    The rotation is applied by splitting the head dimension in half:
    x = [x1 | x2], then rotating: [x1*cos - x2*sin | x2*cos + x1*sin]
    This preserves vector magnitude while encoding position in the angle.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    # Broadcast cos/sin to (1, 1, seq_len, d) for batch/head dims
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation in pairs
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)

    return rotated


# ============================================================================
#  KV Cache — for efficient autoregressive generation at any scale
# ============================================================================

class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    During generation, we only need to compute Q/K/V for the NEW token
    and reuse cached K/V from all previous positions. This reduces
    per-token cost from O(S²) to O(S) — critical for long generations.

    Scalability:
    - Memory grows linearly with sequence length (not quadratically)
    - GQA reduces cache size by gqa_groups ratio
    - Can be extended to paged KV cache for very long sequences
    """

    def __init__(self):
        self.key_cache: Optional[torch.Tensor] = None    # (B, num_kv_heads, S, D)
        self.value_cache: Optional[torch.Tensor] = None  # (B, num_kv_heads, S, D)

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return 0 if self.key_cache is None else self.key_cache.shape[2]

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V to cache and return the full cached K/V.
        key/value: (B, num_kv_heads, new_len, head_dim)
        returns: (full_key, full_value) with shape (B, num_kv_heads, total_len, head_dim)
        """
        if self.key_cache is None:
            self.key_cache = key
            self.value_cache = value
        else:
            self.key_cache = torch.cat([self.key_cache, key], dim=2)
            self.value_cache = torch.cat([self.value_cache, value], dim=2)
        return self.key_cache, self.value_cache

    def reset(self):
        """Clear the cache (start a new generation)."""
        self.key_cache = None
        self.value_cache = None


# ============================================================================
#  Attention — GQA-aware, flash-compatible, KV-cache-ready
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) support.

    GQA (Ainslie et al., 2023):
    - Standard MHA: Q, K, V all have num_heads heads
    - GQA: Q has num_heads heads, K/V have num_kv_heads heads (fewer)
    - Multiple Q heads share the same K/V head within each group
    - Reduces KV cache by num_heads/num_kv_heads ratio
    - At 1B+ params, this is the difference between fitting in VRAM or not
    - Quality is nearly identical to full MHA

    Settings:
    - num_kv_heads == num_heads: standard Multi-Head Attention
    - num_kv_heads == 1: Multi-Query Attention (MQA, most aggressive)
    - 1 < num_kv_heads < num_heads: Grouped Query Attention (balanced)

    All projection weights are bias-free (modern practice since LLaMA).
    Bias in attention projections adds parameters with negligible quality
    impact, and removing them simplifies quantization.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.gqa_groups = config.gqa_groups
        self.scale = self.head_dim ** -0.5

        # Q always has num_heads heads
        self.q_proj = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=False)
        # K/V have num_kv_heads heads (fewer for GQA)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional KV cache for generation.

        x: (B, S, D) — input hidden states
        rotary_cos/sin: (S, head_dim/2) — rotary position embeddings
        attention_mask: (B, 1, S, S) or None — additional attention mask
        use_causal_mask: whether to apply causal (autoregressive) masking
        kv_cache: optional KV cache for efficient generation
        """
        B, S, _ = x.shape

        # Project to Q/K/V
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if rotary_cos is not None:
            # For cached generation, only apply RoPE to the new positions.
            # The cache already has RoPE-rotated K, so we offset the position.
            if kv_cache is not None and kv_cache.seq_len > 0:
                offset = kv_cache.seq_len
                q = apply_rotary_pos_emb(q, rotary_cos[offset:offset + S], rotary_sin[offset:offset + S])
                k = apply_rotary_pos_emb(k, rotary_cos[offset:offset + S], rotary_sin[offset:offset + S])
            else:
                q = apply_rotary_pos_emb(q, rotary_cos[:S], rotary_sin[:S])
                k = apply_rotary_pos_emb(k, rotary_cos[:S], rotary_sin[:S])

        # Update KV cache if present
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            # After cache update, K/V have full sequence length
            kv_seq_len = k.shape[2]
        else:
            kv_seq_len = S

        # Expand KV heads for GQA: repeat each KV head for its group of Q heads
        if self.gqa_groups > 1:
            # (B, num_kv_heads, S, D) → (B, num_kv_heads, groups, S, D) → (B, num_heads, S, D)
            k = k.unsqueeze(2).expand(-1, -1, self.gqa_groups, -1, -1)
            k = k.reshape(B, self.num_heads, kv_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.gqa_groups, -1, -1)
            v = v.reshape(B, self.num_heads, kv_seq_len, self.head_dim)

        # Try flash attention for O(N) memory scaling
        use_flash = (
            FLASH_ATTN_AVAILABLE
            and x.is_cuda
            and not self.config.force_standard_attention
            and attention_mask is None  # flash_attn handles causal internally
        )

        if use_flash:
            # Flash attention expects (B, S, H, D) layout
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            out = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=use_causal_mask,
            )
            out = out.reshape(B, S, -1)
        else:
            # Standard scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if use_causal_mask:
                # Build causal mask: each position can only attend to itself and earlier
                causal_mask = torch.triu(
                    torch.full((S, kv_seq_len), float('-inf'), device=x.device, dtype=x.dtype),
                    diagonal=kv_seq_len - S + 1,
                )
                attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Softmax in float32 for numerical stability even in fp16 training
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(B, S, -1)

        return self.o_proj(out)


# ============================================================================
#  Feed-Forward Networks — SwiGLU (default) and GELU (legacy)
# ============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).

    Architecture: x → [SiLU(W_gate · x) ⊙ (W_up · x)] → W_down → out
    Where SiLU(x) = x · sigmoid(x) (the gating nonlinearity).

    Why SwiGLU over standard GELU FFN:
    - ~1-3% better perplexity at EVERY scale tested
    - The gating mechanism acts as a learned feature selector: it decides
      which dimensions to activate based on the input, providing a
      multiplicative interaction that plain GELU lacks
    - Used by virtually all modern architectures: LLaMA, Gemma, Mistral, etc.
    - For equivalent param count, SwiGLU uses 2/3 of the standard intermediate dim
      (3 matrices at 2/3 width ≈ 2 matrices at full width)

    All weights are bias-free (modern practice for cleaner quantization).
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: SiLU activation selects which features to pass through
        # Up: linear projection provides the feature values
        # Element-wise multiplication: gated feature selection
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GELUFFN(nn.Module):
    """
    Standard GELU FFN for compatibility and ablation studies.
    Used when config.ffn_type == "gelu".
    """

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


def get_ffn(config) -> nn.Module:
    """Factory for FFN layers — swap globally via config."""
    if config.ffn_type == "swiglu":
        return SwiGLUFFN(config)
    elif config.ffn_type == "gelu":
        return GELUFFN(config)
    else:
        raise ValueError(f"Unknown ffn_type: {config.ffn_type}")


# ============================================================================
#  Gated Cross-Attention — Memory injection point
# ============================================================================

class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention for episodic memory injection.

    This is where the memory system interfaces with the encoder.
    Retrieved episodic memories enter the model's residual stream
    through this layer, which is placed at ~60% depth.

    Why gated (initialized to 0.0):
    - Early in training, the episodic memory store is empty/random
    - Injecting garbage memory vectors would destabilize encoder training
    - The gate starts CLOSED (0.0) and LEARNS to open as both the encoder
      and memory store improve in quality
    - This is a critical training stability mechanism: without it, the
      memory system would hurt performance before it could help

    Why at ~60% depth:
    - Too early (layer 1-2): features are too low-level for semantic memory
    - Too late (last layer): not enough remaining layers to integrate memory
    - ~60% depth: abstract features have formed, but there's room to
      incorporate memory before producing output representations

    Architecture:
    - Query: current hidden states from the encoder
    - Key/Value: retrieved episodic memory vectors
    - Gate: learned scalar (tanh-squashed), initialized to 0.0
    - Output: residual + tanh(gate) * cross_attn(x, memory)
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        # Cross-attention projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # THE GATE: initialized to exactly 0.0
        # tanh(0.0) = 0.0, so the gate starts fully closed.
        # As training progresses and memory becomes useful, the gate opens.
        # Using tanh bounds the gate to [-1, 1], preventing unbounded scaling.
        self.gate = nn.Parameter(torch.zeros(1))

        # Pre-norm for the query before cross-attention
        self.norm = get_norm(config.norm_type, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,       # (B, S, D) — encoder hidden states
        memory: torch.Tensor,   # (B, K, D) — retrieved memory vectors (K = topk)
    ) -> torch.Tensor:
        """
        Inject episodic memories into the hidden state via gated cross-attention.
        Returns: x + tanh(gate) * cross_attn(norm(x), memory)
        """
        if memory is None or memory.shape[1] == 0:
            return x  # No memories retrieved — pass through unchanged

        B, S, D = x.shape
        K = memory.shape[1]

        residual = x
        x = self.norm(x)

        # Project query from encoder, key/value from memory
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-attention: every encoder position attends to all retrieved memories
        # No causal mask needed — memories are global context, not sequential
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, K)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        out = torch.matmul(attn, v)  # (B, H, S, D_head)
        out = out.transpose(1, 2).reshape(B, S, D)
        out = self.o_proj(out)

        # Apply the learned gate (starts closed at 0.0)
        return residual + torch.tanh(self.gate) * out


# ============================================================================
#  Transformer Block — Pre-norm with optional memory injection
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with optional memory cross-attention.

    Pre-norm (LN-before) vs Post-norm (LN-after):
    - Pre-norm: LayerNorm BEFORE attention/FFN, then add residual
    - More stable training from scratch (no warmup staircase needed)
    - Gradient flow is more uniform across depth (each layer's residual
      contribution is normalized independently)
    - Used by all modern architectures: GPT-2+, LLaMA, PaLM, etc.
    - Essential for our from-scratch training — we can't afford training instability

    Structure:
    1. Pre-norm → Self-attention → Residual add
    2. [If memory layer] Gated cross-attention with episodic memory
    3. Pre-norm → FFN → Residual add

    Each block is independently wrapped in gradient checkpointing when enabled,
    so VRAM usage is O(1) per layer instead of O(N) during backprop.
    """

    def __init__(self, config, has_memory_cross_attn: bool = False):
        super().__init__()
        # Pre-norm for self-attention
        self.attn_norm = get_norm(config.norm_type, config.hidden_dim)
        self.attn = MultiHeadAttention(config)

        # Pre-norm for FFN
        self.ffn_norm = get_norm(config.norm_type, config.hidden_dim)
        self.ffn = get_ffn(config)

        # Optional memory injection (only at config.memory_inject_layer)
        self.has_memory_cross_attn = has_memory_cross_attn
        if has_memory_cross_attn:
            self.memory_cross_attn = GatedCrossAttention(config)

        # Gradient checkpointing flag
        self._gradient_checkpointing = config.gradient_checkpointing

    def _forward(
        self,
        x: torch.Tensor,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        memory: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Core forward logic, wrapped by gradient checkpointing in forward()."""
        # 1. Pre-norm self-attention
        h = self.attn_norm(x)
        h = self.attn(
            h,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            kv_cache=kv_cache,
        )
        x = x + h

        # 2. Memory injection (only at the designated layer)
        if self.has_memory_cross_attn and memory is not None:
            x = self.memory_cross_attn(x, memory)

        # 3. Pre-norm FFN
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: Optional[torch.Tensor] = None,
        rotary_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        memory: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.

        Gradient checkpointing recomputes activations during backward instead
        of storing them, reducing memory from O(N_layers) to O(1) at the cost
        of ~30% more compute. Essential for fitting large models on limited VRAM.
        """
        if self._gradient_checkpointing and self.training:
            # Note: kv_cache is incompatible with gradient checkpointing
            # (generation doesn't need gradients anyway)
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x, rotary_cos, rotary_sin, attention_mask, use_causal_mask, memory, None,
                use_reentrant=False,
            )
        return self._forward(
            x, rotary_cos, rotary_sin, attention_mask, use_causal_mask, memory, kv_cache,
        )


# ============================================================================
#  Patch Embedding — from-scratch image understanding
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert images into sequences of patch embeddings.

    We train this from scratch instead of using a pretrained ViT. This is
    deliberate and important for the research:
    - Pretrained ViTs bake in representations optimized for ImageNet classification
    - Those representations encode "what object is this" not "what caused this"
    - Our encoder needs to learn visual features that capture causal-relevant
      information (motion, contact, state changes) that classification ViTs discard
    - Training from scratch lets the causal loss directly shape visual representations

    Architecture: unfold image into P×P patches → flatten → linear project → norm
    For standard 224×224 images with 16×16 patches: 196 patches → 196 tokens.

    The norm after projection stabilizes the patch embedding magnitudes,
    which is important when mixing with text token embeddings that have
    their own scale from the embedding table.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches

        # Linear projection from flattened patch to hidden_dim
        # Using a linear layer (not conv2d) for simplicity and clarity.
        # Conv2d with kernel=stride=patch_size is mathematically identical,
        # but the linear version makes the operation more transparent.
        self.projection = nn.Linear(config.patch_dim, config.hidden_dim, bias=False)
        self.norm = get_norm(config.norm_type, config.hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, C, H, W) — batch of images
        returns: (B, num_patches, hidden_dim) — patch embedding sequence
        """
        B, C, H, W = images.shape
        P = self.patch_size

        # Reshape into patches:
        # (B, C, H, W) → (B, C, H//P, P, W//P, P)
        #              → (B, H//P, W//P, C, P, P)    [permute]
        #              → (B, num_patches, C*P*P)       [flatten]
        x = images.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H//P, W//P, C, P, P)
        x = x.reshape(B, -1, C * P * P)                # (B, num_patches, patch_dim)

        # Project and normalize
        x = self.projection(x)
        x = self.norm(x)
        return x


# ============================================================================
#  Unified Encoder — the backbone
# ============================================================================

class UnifiedEncoder(nn.Module):
    """
    Unified encoder for the Causal-JEPA World Model.

    This is the backbone of the architecture. It processes text tokens
    and image patches in the SAME transformer, producing a shared latent
    space where causal structure can be learned across modalities.

    The encoder serves multiple roles simultaneously:
    1. Encode observations into latent vectors z (for causal prediction)
    2. Encode events into event embeddings (for transition prediction)
    3. Produce hidden states for the LM head (for text generation)
    4. Receive episodic memories via cross-attention (for infinite learning)

    This multi-role design is intentional: by sharing weights across all
    four functions, the encoder is forced to build representations that
    are simultaneously useful for understanding, prediction, generation,
    and memory retrieval. This pressure produces richer representations
    than a single-task encoder.

    Scalability:
    - Config-driven: all dimensions come from WorldModelConfig
    - GQA/Flash Attention/Gradient Checkpointing are transparent
    - KV Cache for efficient generation
    - Same code runs from 10M to 1.5B params

    Weight Initialization:
    - Embeddings: N(0, 0.02) — standard for transformers
    - Linear layers: N(0, 0.02 / sqrt(2 * num_layers)) — scaled init
    - The 1/sqrt(2N) factor ensures residual stream variance stays bounded
      as depth increases, preventing exploding activations in deep models
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- Embeddings ----
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.patch_embedding = PatchEmbedding(config)

        # Modality embeddings: distinguish text (0), image (1), memory (2)
        # Added to EVERY token before the encoder layers.
        # Without this, the model cannot distinguish why certain inputs
        # have different statistical properties from others.
        self.modality_embedding = nn.Embedding(config.num_modalities, config.hidden_dim)

        # ---- Position Encoding ----
        if config.pos_encoding == "rope":
            self.rotary = RotaryEmbedding(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta,
            )
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
            self.rotary = None

        self.embed_dropout = nn.Dropout(config.dropout)

        # ---- Transformer Layers ----
        self.layers = nn.ModuleList([
            TransformerBlock(
                config,
                has_memory_cross_attn=(i == config.memory_inject_layer),
            )
            for i in range(config.num_layers)
        ])

        # ---- Final Norm ----
        # Pre-norm architecture requires a final norm after the last layer
        # (the last layer's residual doesn't get normalized otherwise)
        self.final_norm = get_norm(config.norm_type, config.hidden_dim)

        # ---- Initialize Weights ----
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using depth-scaled initialization.

        The scaling factor 1/sqrt(2 * num_layers) for linear layers ensures
        that the variance of the residual stream stays O(1) regardless of depth.
        Without this, a 32-layer model would have ~8x the activation variance
        of a 4-layer model, causing training instability.

        This is a critical scalability feature: the same initialization
        produces stable training at ANY depth.
        """
        depth_factor = 1.0 / math.sqrt(2 * self.config.num_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * depth_factor)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,       # (B, S_text) long
        images: Optional[torch.Tensor] = None,           # (B, C, H, W) float
        modality_ids: Optional[torch.Tensor] = None,     # (B, S_total) long
        attention_mask: Optional[torch.Tensor] = None,    # (B, S_total) or (B, 1, S, S)
        memory_vectors: Optional[torch.Tensor] = None,   # (B, K, D) retrieved memories
        use_causal_mask: bool = True,
        kv_caches: Optional[List[KVCache]] = None,       # per-layer KV caches
    ) -> torch.Tensor:
        """
        Forward pass through the unified encoder.

        Handles three input configurations:
        1. Text only: input_ids provided, images=None
        2. Image only: images provided, input_ids=None
        3. Mixed: both provided, concatenated as [text_tokens | image_patches]

        Args:
            input_ids: Token IDs for text input
            images: Raw images (will be patchified internally)
            modality_ids: Per-position modality labels (0=text, 1=image, 2=memory)
                          Auto-generated if not provided.
            attention_mask: Additional attention mask (padding, etc.)
            memory_vectors: Retrieved episodic memories for cross-attention injection
            use_causal_mask: Whether to apply causal masking (True for generation)
            kv_caches: Per-layer KV caches for efficient generation

        Returns:
            (B, S_total, hidden_dim) — encoder hidden states
        """
        embeddings = []
        modality_parts = []  # for auto-generating modality IDs
        device = None

        # Text embeddings
        if input_ids is not None:
            device = input_ids.device
            text_emb = self.token_embedding(input_ids)  # (B, S_text, D)
            embeddings.append(text_emb)
            modality_parts.append(
                torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=device)
            )

        # Image patch embeddings
        patch_emb = None
        if images is not None:
            device = images.device
            patch_emb = self.patch_embedding(images)  # (B, num_patches, D)
            embeddings.append(patch_emb)
            modality_parts.append(
                torch.ones(images.shape[0], patch_emb.shape[1], dtype=torch.long, device=device)
            )

        if not embeddings:
            raise ValueError("At least one of input_ids or images must be provided")

        # Concatenate modalities into a single sequence
        x = torch.cat(embeddings, dim=1)  # (B, S_total, D)
        total_seq_len = x.shape[1]

        # Add modality embeddings
        if modality_ids is None:
            modality_ids = torch.cat(modality_parts, dim=1)
        x = x + self.modality_embedding(modality_ids)

        # Add learned positional embeddings (if not using RoPE)
        if self.pos_embedding is not None:
            positions = torch.arange(total_seq_len, device=device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

        x = self.embed_dropout(x)

        # Prepare rotary embeddings
        rotary_cos, rotary_sin = None, None
        if self.rotary is not None:
            # Request enough positions for full sequence + any cached positions
            max_pos = total_seq_len
            if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0].seq_len > 0:
                max_pos = kv_caches[0].seq_len + total_seq_len
            rotary_cos, rotary_sin = self.rotary(max_pos)

        # Prepare attention mask for padding
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert (B, S) padding mask to (B, 1, S, S) attention mask
            # padding positions (0) get -inf, valid positions (1) get 0
            extended_mask = attention_mask[:, None, None, :]  # (B, 1, 1, S)
            extended_mask = (1.0 - extended_mask.float()) * torch.finfo(x.dtype).min
            attention_mask = extended_mask

        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(
                x,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                memory=memory_vectors,
                kv_cache=layer_kv_cache,
            )

        # Final normalization
        x = self.final_norm(x)
        return x

    def create_kv_caches(self) -> List[KVCache]:
        """Create a fresh set of KV caches for generation (one per layer)."""
        return [KVCache() for _ in range(len(self.layers))]
