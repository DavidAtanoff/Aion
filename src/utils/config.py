"""
Configuration for the Causal-JEPA World Model architecture.

Scalability philosophy: Every architectural dimension is parameterized.
The same code runs a 10M-param proof-of-concept on a laptop GPU and a
1B+ param research model on a multi-GPU cluster. Scale presets provide
sensible defaults, but any parameter can be overridden independently.

The config is the SINGLE SOURCE OF TRUTH for the model architecture.
All modules read their dimensions, normalization choices, and structural
decisions from this config — no hardcoded values anywhere in the model code.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldModelConfig:
    """
    Central configuration for the Causal-JEPA World Model.

    Design rationale for defaults:
    - RMSNorm over LayerNorm: ~10% faster with no quality loss at scale (Zhang & Sennrich, 2019)
    - SwiGLU over GELU: consistently better perplexity/param ratio (Shazeer, 2020)
    - RoPE over learned positional embeddings: extrapolates to unseen sequence lengths
    - GQA-ready: num_kv_heads < num_heads enables memory-efficient attention at scale
    - Weight tying: reduces param count by sharing token embedding and LM head weights

    Usage:
        config = WorldModelConfig.base()     # ~50M params, Kaggle T4
        config = WorldModelConfig.tiny()     # ~10M params, debugging
        config = WorldModelConfig.research() # ~1.5B params, multi-GPU
    """

    # ========================
    #  Model Dimensions
    # ========================
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    # GQA: set num_kv_heads < num_heads for grouped-query attention.
    # None → defaults to num_heads (standard multi-head attention).
    # Key insight: GQA reduces KV cache memory linearly with the ratio
    # num_heads/num_kv_heads, enabling efficient inference at billion-param scales.
    num_kv_heads: Optional[int] = None
    # FFN intermediate dimension. None → auto-computed based on ffn_type:
    #   - SwiGLU: 8/3 * hidden_dim (rounded to multiple of 256 for GPU alignment)
    #   - GELU: 4 * hidden_dim (standard expansion)
    ffn_dim: Optional[int] = None
    dropout: float = 0.1

    # ========================
    #  Vocabulary & Sequence
    # ========================
    vocab_size: int = 32_000
    max_seq_len: int = 2048
    # Padding token ID — used for masking in attention and loss computation
    pad_token_id: int = 0

    # ========================
    #  Vision
    # ========================
    # We train patch embeddings from scratch (no pretrained ViT) to avoid
    # baking in representations optimized for classification rather than
    # causal structure. The model must learn its OWN visual features.
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3

    # ========================
    #  Architecture Choices
    # ========================
    # These are the scalable primitives. Each choice is motivated by
    # empirical evidence from modern LLM research showing better
    # scaling properties than their predecessors.
    norm_type: str = "rmsnorm"     # "rmsnorm" | "layernorm"
    ffn_type: str = "swiglu"       # "swiglu" | "gelu"
    pos_encoding: str = "rope"     # "rope" | "learned"
    # RoPE theta: controls frequency base for rotary embeddings.
    # 10000.0 works well up to ~8K context. For longer contexts, use 500K+
    # (as in LLaMA 3). This is the primary knob for context length scaling.
    rope_theta: float = 10000.0

    # ========================
    #  Causal Transition Network
    # ========================
    # The transition network is intentionally simpler than the encoder.
    # It operates in the compressed state space (R^D), not token space.
    # The expansion factor controls width independently of encoder width.
    transition_layers: int = 3
    transition_expansion: int = 2  # intermediate_dim = hidden_dim * expansion

    # ========================
    #  Memory System (MemOS)
    # ========================
    episodic_capacity: int = 100_000
    topk_retrieve: int = 8
    # Memory injection layer — where episodic memories enter the encoder.
    # None → auto-computed as ~60% of depth. At layer 4 in a 6-layer model,
    # abstract features have formed but haven't yet been committed to output.
    # This is empirically the sweet spot for external knowledge injection.
    memory_inject_layer: Optional[int] = None
    num_modalities: int = 3  # 0=text, 1=image_patch, 2=retrieved_memory
    # Cosine similarity threshold for concept injection from semantic memory.
    # Higher = more selective (only inject highly relevant concepts).
    concept_sim_threshold: float = 0.7
    # Number of clusters for episodic → semantic consolidation.
    # More clusters = finer-grained concepts but more storage.
    num_concept_clusters: int = 64

    # ========================
    #  Training Helpers
    # ========================
    # Gradient checkpointing: trade compute for memory. Essential for
    # fitting larger models on limited VRAM. Saves ~40% memory at ~20% speed cost.
    gradient_checkpointing: bool = False
    # Weight tying: share weights between token embedding and LM head output projection.
    # Reduces param count significantly (hidden_dim * vocab_size parameters saved).
    # Standard practice since Press & Wolf (2017), used by GPT-2, LLaMA, etc.
    tie_word_embeddings: bool = True

    # ========================
    #  EMA Target Encoder
    # ========================
    # The EMA encoder provides stable prediction targets for the causal loss.
    # Without EMA, the targets would shift every step, causing training instability.
    # Momentum annealing: start at 0.99 (more responsive early) → 0.999 (more stable late).
    ema_momentum_start: float = 0.99
    ema_momentum_end: float = 0.999
    ema_anneal_steps: int = 10_000

    # ========================
    #  Scalability Config
    # ========================
    # Multi-vector state compression: use more queries for richer state representations.
    # At base scale (512D), 1 query is sufficient. At 2048D, 4-8 queries can capture
    # more complex world state configurations.
    num_state_queries: int = 1
    # Flash attention: automatically used when available. This flag forces fallback
    # to standard attention for debugging or environments without flash_attn.
    force_standard_attention: bool = False

    def __post_init__(self):
        """Validate configuration and auto-compute derived values."""
        # Auto-compute GQA heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # Auto-compute FFN dimension
        if self.ffn_dim is None:
            if self.ffn_type == "swiglu":
                # SwiGLU has 3 weight matrices (gate, up, down) instead of 2,
                # so we use 2/3 of the standard 4x expansion to match param count.
                # Round to nearest multiple of 256 for GPU memory alignment.
                raw = int(8 / 3 * self.hidden_dim)
                self.ffn_dim = ((raw + 255) // 256) * 256
            else:
                self.ffn_dim = self.hidden_dim * 4

        # Auto-compute memory injection layer (~60% depth)
        if self.memory_inject_layer is None:
            self.memory_inject_layer = max(1, int(self.num_layers * 0.6))

        # === Validation ===
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )
        assert self.memory_inject_layer < self.num_layers, (
            f"memory_inject_layer ({self.memory_inject_layer}) must be "
            f"< num_layers ({self.num_layers})"
        )
        assert self.image_size % self.patch_size == 0, (
            f"image_size ({self.image_size}) must be divisible by "
            f"patch_size ({self.patch_size})"
        )
        assert self.norm_type in ("rmsnorm", "layernorm"), (
            f"norm_type must be 'rmsnorm' or 'layernorm', got '{self.norm_type}'"
        )
        assert self.ffn_type in ("swiglu", "gelu"), (
            f"ffn_type must be 'swiglu' or 'gelu', got '{self.ffn_type}'"
        )
        assert self.pos_encoding in ("rope", "learned"), (
            f"pos_encoding must be 'rope' or 'learned', got '{self.pos_encoding}'"
        )

    # ========================
    #  Derived Properties
    # ========================

    @property
    def head_dim(self) -> int:
        """Dimension per attention head. Must divide evenly into hidden_dim."""
        return self.hidden_dim // self.num_heads

    @property
    def num_patches(self) -> int:
        """Number of image patches for the configured image/patch sizes."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        """Flattened dimension of a single image patch."""
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def kv_dim(self) -> int:
        """Total dimension for key/value projections (GQA-aware)."""
        return self.num_kv_heads * self.head_dim

    @property
    def gqa_groups(self) -> int:
        """Number of query heads sharing each KV head."""
        return self.num_heads // self.num_kv_heads

    # ========================
    #  Scale Presets
    # ========================

    @classmethod
    def tiny(cls) -> "WorldModelConfig":
        """
        ~10M params. For debugging, unit tests, and rapid prototyping.
        Trains in minutes on a consumer GPU.
        """
        return cls(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=512,
            dropout=0.1,
            num_state_queries=1,
        )

    @classmethod
    def base(cls) -> "WorldModelConfig":
        """
        ~50M params. Kaggle T4 target (16GB VRAM).
        The primary proof-of-concept scale. Uses fp16 + gradient accumulation.
        """
        return cls()  # defaults ARE the base config

    @classmethod
    def large(cls) -> "WorldModelConfig":
        """
        ~150M params. Single A100 (40GB) or 2x T4 target.
        GQA enabled (4 KV heads < 12 query heads) for memory efficiency.
        """
        return cls(
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,  # GQA: 3 query heads per KV head
            max_seq_len=4096,
            dropout=0.1,
            num_state_queries=2,  # richer state at this scale
        )

    @classmethod
    def xl(cls) -> "WorldModelConfig":
        """
        ~350M params. Multi-GPU target (2-4 A100s).
        Aggressive GQA for inference efficiency.
        """
        return cls(
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            num_kv_heads=4,  # GQA: 4 query heads per KV head
            max_seq_len=8192,
            dropout=0.05,
            num_state_queries=4,
            gradient_checkpointing=True,  # likely needed at this scale
        )

    @classmethod
    def research(cls) -> "WorldModelConfig":
        """
        ~1.5B params. Full research scale for serious AGI experiments.
        Requires multi-GPU with FSDP or tensor parallelism.
        This is the scale where we expect the novel architectural
        properties to show the strongest signal over baselines.
        """
        return cls(
            hidden_dim=2048,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,  # GQA: 4 query heads per KV head
            max_seq_len=16384,
            dropout=0.05,
            num_state_queries=8,
            gradient_checkpointing=True,
            rope_theta=500000.0,  # extended context support
        )

    def estimate_params(self) -> dict:
        """
        Estimate parameter counts for each major component.
        Useful for verifying a config before instantiating the model.
        """
        # Token embedding
        embed_params = self.vocab_size * self.hidden_dim

        # Patch embedding
        patch_params = self.patch_dim * self.hidden_dim

        # Per-layer params
        # Self-attention: Q + KV + O projections
        attn_params = (
            self.hidden_dim * self.hidden_dim  # Q
            + self.hidden_dim * self.kv_dim  # K
            + self.hidden_dim * self.kv_dim  # V
            + self.hidden_dim * self.hidden_dim  # O
        )
        # FFN params (SwiGLU has 3 matrices, GELU has 2)
        if self.ffn_type == "swiglu":
            ffn_params = 3 * self.hidden_dim * self.ffn_dim
        else:
            ffn_params = 2 * self.hidden_dim * self.ffn_dim
        # Norm params (small, but count them)
        norm_params = 2 * self.hidden_dim  # 2 norms per layer

        layer_params = attn_params + ffn_params + norm_params
        total_layer_params = layer_params * self.num_layers

        # Memory cross-attention (only 1 layer has it)
        mem_attn_params = 4 * self.hidden_dim * self.hidden_dim + self.hidden_dim

        # Causal compressor
        compressor_params = 4 * self.hidden_dim * self.hidden_dim + self.hidden_dim * self.num_state_queries

        # Transition network
        trans_input = self.hidden_dim * 2
        trans_inter = self.hidden_dim * self.transition_expansion
        trans_params = trans_input * trans_inter + trans_inter * trans_inter * (self.transition_layers - 2) + trans_inter * self.hidden_dim

        # LM head (tied with embedding → 0 extra params if tied)
        lm_params = 0 if self.tie_word_embeddings else self.hidden_dim * self.vocab_size

        # EMA encoder is a copy but shares no gradients — same param count
        ema_params = total_layer_params + embed_params + patch_params

        return {
            "token_embedding": embed_params,
            "patch_embedding": patch_params,
            "encoder_layers": total_layer_params,
            "memory_cross_attn": mem_attn_params,
            "causal_compressor": compressor_params,
            "transition_network": trans_params,
            "lm_head": lm_params,
            "ema_encoder (frozen)": ema_params,
            "total_trainable": (
                embed_params + patch_params + total_layer_params
                + mem_attn_params + compressor_params + trans_params + lm_params
            ),
            "total_with_ema": (
                embed_params + patch_params + total_layer_params
                + mem_attn_params + compressor_params + trans_params + lm_params
                + ema_params
            ),
        }

    def __repr__(self) -> str:
        estimates = self.estimate_params()
        trainable = estimates["total_trainable"]
        scale = "B" if trainable > 1e9 else "M"
        count = trainable / 1e9 if trainable > 1e9 else trainable / 1e6
        return (
            f"WorldModelConfig("
            f"layers={self.num_layers}, "
            f"dim={self.hidden_dim}, "
            f"heads={self.num_heads}/{self.num_kv_heads}kv, "
            f"ffn={self.ffn_type}@{self.ffn_dim}, "
            f"pos={self.pos_encoding}, "
            f"norm={self.norm_type}, "
            f"~{count:.1f}{scale} trainable params"
            f")"
        )
