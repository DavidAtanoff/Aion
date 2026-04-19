"""
Language Modeling Head for the Causal-JEPA World Model.

This module provides text generation capability on top of the causal encoder.
Without it, the model would only produce latent state vectors — useful for
research but not for conversation or interaction.

WHY WE NEED AN LM HEAD:
=======================
The causal world model learns to predict state transitions in latent space.
But humans communicate in language. The LM head bridges this gap:
- Takes encoder hidden states (rich with causal understanding)
- Projects them to vocabulary logits
- Enables autoregressive text generation

The hypothesis is that a model whose encoder is shaped by causal prediction
(not just next-token prediction) will produce text that reflects deeper
understanding — better causal reasoning, more accurate counterfactual
statements, and more coherent explanations of cause-and-effect.

WEIGHT TYING:
=============
By default, the LM head shares weights with the token embedding layer.
This is standard practice (Press & Wolf 2017) and:
- Saves hidden_dim × vocab_size parameters (~16M for 512 × 32K)
- Forces the embedding space to be useful for both input AND output
- Makes the model more parameter-efficient at all scales

GENERATION:
===========
The generate() method supports:
- Top-p (nucleus) sampling: sample from the smallest set of tokens whose
  cumulative probability exceeds p. This produces diverse but coherent text.
- Temperature scaling: lower = more deterministic, higher = more creative
- KV cache integration: O(1) per-token cost after initial prompt processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LanguageModelingHead(nn.Module):
    """
    Project encoder hidden states to vocabulary logits for text generation.

    Architecture:
    - Normalization (RMSNorm/LayerNorm) for stable output distribution
    - Linear projection: hidden_dim → vocab_size
    - Optional weight tying with the token embedding layer

    The norm before projection is important: without it, the raw encoder
    output at different layers/positions can have very different magnitudes,
    leading to unstable logit distributions. The norm ensures consistent
    input scale to the projection layer.
    """

    def __init__(self, config, token_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        from .encoder import get_norm
        self.config = config

        # Normalization before projection
        self.norm = get_norm(config.norm_type, config.hidden_dim)

        # LM projection
        if config.tie_word_embeddings and token_embedding is not None:
            # Weight tying: share weights with the token embedding layer.
            # The projection weight IS the embedding weight, transposed.
            # This means updating embeddings also updates the output projection
            # and vice versa — they co-evolve to maintain a consistent space.
            self.proj = None
            self.tied_embedding = token_embedding
        else:
            self.proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
            self.tied_embedding = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden_states: (B, S, hidden_dim) — encoder output

        Returns:
            (B, S, vocab_size) — logits over vocabulary
        """
        x = self.norm(hidden_states)

        if self.tied_embedding is not None:
            # Weight-tied projection: multiply by embedding weight transposed
            logits = F.linear(x, self.tied_embedding.weight)
        else:
            logits = self.proj(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        Sample next token from the model's output distribution.

        Uses nucleus (top-p) sampling by default:
        1. Compute logits from hidden states
        2. Apply temperature scaling
        3. Sort logits and compute cumulative probabilities
        4. Zero out tokens outside the top-p nucleus
        5. Sample from the renormalized distribution

        Why top-p over greedy:
        - Greedy decoding produces repetitive, boring text
        - Pure random sampling can produce incoherent text
        - Top-p balances quality and diversity: it adapts the sampling set
          based on how confident the model is. When the model is confident,
          the nucleus is small (few tokens). When uncertain, it's larger.

        Args:
            hidden_states: (B, 1, D) — hidden state at the current position
            temperature: <1.0 = more deterministic, >1.0 = more creative
            top_p: cumulative probability threshold for nucleus sampling
            top_k: if > 0, only consider top-k tokens (0 = disabled)

        Returns:
            (B, 1) — sampled token IDs
        """
        # Get logits for the LAST position only
        logits = self.forward(hidden_states[:, -1:, :])  # (B, 1, V)
        logits = logits[:, 0, :]  # (B, V) — last position

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering (optional)
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_top_k, float('-inf'))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            # Shift right so that the first token above threshold is kept
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')

            # Scatter back to original ordering
            logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)

        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        return next_token
