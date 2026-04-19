"""
Causal State Compressor and Transition Network.

This module contains the core novel components of the Causal-JEPA architecture:
the state compressor and the transition predictor.

PHILOSOPHICAL MOTIVATION:
========================
Standard LLMs learn P(next_token | context). This is powerful but shallow:
the model never needs to build an explicit model of the world — it only needs
statistical correlations between token sequences.

Our architecture separates two fundamentally different tasks:
1. OBSERVATION → STATE: "What does the world look like?" (CausalStateCompressor)
2. STATE × EVENT → NEXT STATE: "How does the world change?" (CausalTransitionNetwork)

By learning an explicit transition function f(s_t, e_t) → s_{t+1}, the model
is forced to build representations where:
- States are stable across modalities (text description ≈ image observation)
- Events are parameterized as causes, not just token sequences
- Counterfactual reasoning is free: just pass a different event e' and compare

This is what separates a world model from a language model.

SCALABILITY:
============
- CausalStateCompressor uses multi-query attention pooling (can increase
  num_state_queries at larger scales for richer state representations)
- CausalTransitionNetwork is a simple MLP but with residual connections
  that enable stable training at any depth
- Both components operate in the compressed state space (R^D), not the
  full token sequence space, making them computationally cheap relative
  to the encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import get_norm


class CausalStateCompressor(nn.Module):
    """
    Compress encoder output sequence into a single causal state vector.

    WHY ATTENTION POOLING (not mean pooling):
    =========================================
    Mean pooling treats every token/patch equally. But causal states are sparse:
    - In "a ball hits a window": the ball's velocity and the window's position
      are causally relevant; the color of the ceiling is not
    - In a physics simulation frame: only the objects near collision points
      matter for the next-state prediction; the background is irrelevant

    Attention pooling uses a learned query vector that attends over the encoder
    output, learning WHICH parts of the observation carry causal information.
    This is a critical inductive bias: causation is selective, not diffuse.

    MULTI-QUERY SCALING:
    ====================
    At larger scales (1B+ params), a single D-dimensional vector may not
    capture complex world states with many interacting objects. The
    num_state_queries parameter allows multiple learned queries, each
    attending to different aspects of the state:
    - Query 1 might attend to object positions
    - Query 2 might attend to object velocities
    - Query 3 might attend to environmental context

    The multiple query outputs are then merged via a learned projection.
    This provides a natural scaling axis for state representation capacity
    that's independent of the encoder width.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_queries = config.num_state_queries
        self.scale = self.head_dim ** -0.5

        # Learned query vector(s) for attention pooling.
        # Initialize with small random values — the model will learn what to
        # attend to based on the causal prediction loss signal.
        self.state_queries = nn.Parameter(
            torch.randn(1, self.num_queries, config.hidden_dim) * 0.02
        )

        # Cross-attention projections: queries attend to encoder sequence
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # If using multiple queries, merge them into a single state vector.
        # This is the "state bottleneck": all world state information must
        # pass through a fixed-dimensional vector, forcing abstraction.
        if self.num_queries > 1:
            self.merge_proj = nn.Linear(
                self.num_queries * config.hidden_dim, config.hidden_dim, bias=False
            )
        else:
            self.merge_proj = None

        self.norm = get_norm(config.norm_type, config.hidden_dim)

    def forward(
        self,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compress encoder sequence into a single causal state vector.

        Args:
            encoder_output: (B, S, D) — output from UnifiedEncoder
            mask: (B, S) — 1 for valid positions, 0 for padding

        Returns:
            (B, D) — compressed causal state vector
        """
        B, S, D = encoder_output.shape

        # Expand learned queries for the batch
        queries = self.state_queries.expand(B, -1, -1)  # (B, Q, D)

        # Project Q/K/V
        q = self.q_proj(queries).view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_output).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_output).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: learned queries attend to encoder output
        # Shape: (B, num_heads, num_queries, S)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Mask out padding positions (if any)
        if mask is not None:
            # Cast to bool — attention masks may arrive as int64 from DataLoaders
            bool_mask = mask.bool() if mask.dtype != torch.bool else mask
            attn_weights = attn_weights.masked_fill(
                ~bool_mask[:, None, None, :],  # (B, 1, 1, S)
                float('-inf'),
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Weighted sum of encoder values
        out = torch.matmul(attn_weights, v)  # (B, num_heads, num_queries, head_dim)
        out = out.transpose(1, 2).reshape(B, self.num_queries, D)
        out = self.o_proj(out)
        out = self.norm(out)

        # Merge multiple queries into a single state vector
        if self.merge_proj is not None:
            out = out.reshape(B, -1)           # (B, Q * D)
            out = self.merge_proj(out)          # (B, D)
        else:
            out = out.squeeze(1)               # (B, D)

        return out


class CausalTransitionNetwork(nn.Module):
    """
    Predicts the next world state given current state and an event.

    THIS IS THE CORE NOVEL COMPONENT.

    It implements the causal transition function:
        s_{t+1} = f(s_t, e_t)

    Where:
    - s_t ∈ R^D is the current world state (from CausalStateCompressor)
    - e_t ∈ R^D is the event embedding (same encoder + compressor applied to
      a description of the event/action that occurs)
    - s_{t+1} ∈ R^D is the predicted next state

    WHY THIS MATTERS FOR AGI:
    =========================
    A standard autoregressive LM predicts: P(token_{n+1} | token_1..n)
    Our transition network predicts: s_{t+1} = f(s_t, e_t)

    The difference is fundamental:
    1. LMs predict in TOKEN space → can memorize surface patterns
       Transition networks predict in STATE space → must understand abstractions
    2. LMs entangle observation with prediction → no explicit world model
       Transition networks separate them → explicit, manipulable world model
    3. Counterfactual reasoning in an LM requires: "imagine a different prefix"
       In our model: just call f(s_t, e_different) → direct comparison

    ARCHITECTURE:
    =============
    - Input: concat(s_t, e_t) ∈ R^(2D)
    - MLP with configurable depth and width
    - Residual connection from input state (most transitions are small changes)
    - Output normalized (cosine similarity loss works on unit hypersphere)

    The residual connection encodes the INDUCTIVE BIAS that most world state
    transitions are incremental — the world doesn't completely change at each
    timestep. This is true in physics (continuity), conversation (coherence),
    and virtually all causal systems.

    SCALABILITY:
    ============
    - transition_layers controls depth (more layers = more expressive transitions)
    - transition_expansion controls width (wider = more capacity per layer)
    - Residual connections ensure gradient flow at any depth
    - Layer normalization prevents activation explosion at scale
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        input_dim = config.hidden_dim * 2  # concat(state, event)
        intermediate_dim = config.hidden_dim * config.transition_expansion

        # Build the transition MLP dynamically based on config.
        # Each layer: Linear → Norm → GELU → Dropout
        # The GELU activation is specifically chosen for the transition network
        # (not SwiGLU) because the transition function should be a smooth
        # approximation to the true state dynamics, and GELU provides
        # smoother gradients than ReLU/SiLU for regression-like tasks.
        layers = []
        current_dim = input_dim

        for i in range(config.transition_layers - 1):
            layers.append(nn.Linear(current_dim, intermediate_dim, bias=False))
            layers.append(get_norm(config.norm_type, intermediate_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            current_dim = intermediate_dim

        # Final projection back to state space
        layers.append(nn.Linear(current_dim, config.hidden_dim, bias=False))

        self.mlp = nn.Sequential(*layers)

        # Residual projection: allows the state to "pass through" if the event
        # causes minimal change. Without this, the network must learn the identity
        # mapping for "nothing happened" events, which wastes capacity.
        # For shallow networks (≤2 layers), use identity to save parameters.
        if config.transition_layers > 2:
            self.residual_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

        self.output_norm = get_norm(config.norm_type, config.hidden_dim)

        # Learnable residual scaling — initialized to 0.5 so that the network's
        # prediction and the residual contribute equally at initialization.
        # This balances "stay similar to current state" vs "predict something new".
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        state: torch.Tensor,   # (B, D) — current world state
        event: torch.Tensor,   # (B, D) — event embedding
    ) -> torch.Tensor:
        """
        Predict the next world state.

        The prediction is: s_{t+1} = norm(mlp([s_t; e_t]) + α * proj(s_t))
        where α is a learned residual scaling factor.

        Args:
            state: Current causal state s_t (from CausalStateCompressor)
            event: Event embedding e_t (from UnifiedEncoder + CausalStateCompressor)

        Returns:
            Predicted next state s_{t+1} ∈ R^D (normalized for cosine similarity loss)
        """
        # Concatenate state and event to form the transition input
        x = torch.cat([state, event], dim=-1)  # (B, 2D)

        # Forward through transition MLP
        predicted = self.mlp(x)  # (B, D)

        # Add residual from current state (most transitions are incremental)
        residual = self.residual_proj(state)  # (B, D)
        predicted = predicted + torch.sigmoid(self.residual_scale) * residual

        # Normalize output — the causal prediction loss uses cosine similarity,
        # so predictions should live on the unit hypersphere. Normalization after
        # the residual prevents the residual from dominating the direction.
        predicted = self.output_norm(predicted)

        return predicted

    def predict_counterfactual(
        self,
        state: torch.Tensor,           # (B, D) — current world state
        real_event: torch.Tensor,       # (B, D) — actual event
        counterfactual_event: torch.Tensor,  # (B, D) — hypothetical event
    ) -> dict:
        """
        Predict both real and counterfactual next states.

        This is the key operation for counterfactual reasoning:
        "What would have happened if event E' occurred instead of E?"

        Returns a dict with:
        - real_next_state: f(s_t, e_real)
        - cf_next_state: f(s_t, e_counterfactual)
        - causal_distance: cosine distance between the two predictions

        A well-trained model should produce causal_distance that correlates
        with the true causal distance between the events' effects.
        """
        real_next = self.forward(state, real_event)
        cf_next = self.forward(state, counterfactual_event)

        # Causal distance: how different are the outcomes?
        # 0.0 = identical outcomes (event doesn't matter)
        # 1.0 = maximally different outcomes (event is critical)
        cosine_sim = F.cosine_similarity(real_next, cf_next, dim=-1)
        causal_distance = 1.0 - cosine_sim

        return {
            "real_next_state": real_next,
            "cf_next_state": cf_next,
            "causal_distance": causal_distance,
        }
