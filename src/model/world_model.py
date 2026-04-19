"""
CausalWorldModel — Full Model Orchestrator.

This is the top-level module that composes all components into a complete
Causal-JEPA World Model. It coordinates:
1. The unified encoder (text + image → latent vectors)
2. The causal state compressor (sequence → single state vector)
3. The causal transition network (state + event → next state)
4. The EMA target encoder (stable prediction targets)
5. The memory system (episodic + semantic memory)
6. The language modeling head (latent → text generation)

ARCHITECTURE OVERVIEW:
======================
                    ┌─────────────────┐
        text/image →│  Unified Encoder │→ hidden states → LM Head → tokens
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ State Compressor │→ state vector s_t
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
        event e_t →│ Transition Net   │→ predicted s_{t+1}
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   EMA Encoder   │→ target s_{t+1} (detached)
                    └─────────────────┘

        ┌─────────────────────────────────┐
        │  Memory System (CPU)            │
        │  ┌───────────┐ ┌─────────────┐  │
        │  │ Episodic   │ │  Semantic   │  │
        │  │ (FAISS)    │→│  (Concepts) │  │
        │  └───────────┘ └─────────────┘  │
        └─────────────────────────────────┘

The EMA encoder is a separate copy of the encoder + compressor that is
updated via exponential moving average (not gradient descent). It provides
stable prediction targets for the causal loss, preventing the pathological
case where the model learns to predict its own shifting representations
(representational collapse). This technique comes from BYOL/DINO/I-JEPA.

FORWARD MODES:
==============
The model supports multiple forward modes depending on the task:
1. FULL TRAINING: encode(obs_t) → compress → transition(state, event) → losses
2. ENCODE ONLY: input → latent vectors (for embedding extraction)
3. GENERATE: autoregressive text generation with memory retrieval
4. PREDICT: state transition prediction (for causal reasoning)
5. EMA UPDATE: update EMA encoder weights (after optimizer step)

SCALABILITY:
============
- All components read dimensions from the same WorldModelConfig
- The same CausalWorldModel class runs from 10M to 1.5B params
- EMA encoder doubles memory but NOT VRAM training cost (no gradients)
- Memory system runs on CPU — no VRAM impact regardless of memory size
- KV cache for O(1) per-token generation cost
- Checkpoint save/load preserves entire model state including memory
"""

import os
import copy
import logging
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import WorldModelConfig
from .encoder import UnifiedEncoder, KVCache
from .causal_net import CausalStateCompressor, CausalTransitionNetwork
from .memory import MemoryManager
from .lm_head import LanguageModelingHead

logger = logging.getLogger(__name__)


class CausalWorldModel(nn.Module):
    """
    Complete Causal-JEPA World Model.

    This is the main model class that researchers interact with. It combines
    all architectural components and provides clean interfaces for training,
    inference, and generation.

    Usage:
        config = WorldModelConfig.base()
        model = CausalWorldModel(config)

        # Training forward pass
        outputs = model(
            input_ids=tokens,
            images=images,
            event_ids=event_tokens,
            target_ids=next_observation_tokens,
        )
        loss = outputs["total_loss"]
        loss.backward()
        optimizer.step()
        model.update_ema(step)

        # Generation
        response_ids = model.generate(prompt_ids, max_length=100)

        # Causal reasoning
        state = model.encode_state(observation_tokens)
        next_state = model.predict_next_state(state, event_embedding)
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config

        # ---- Core Components ----
        # The encoder is the backbone — processes all inputs into latent space
        self.encoder = UnifiedEncoder(config)

        # The compressor reduces encoder sequence to a single state vector
        self.compressor = CausalStateCompressor(config)

        # The transition network predicts state changes from events
        self.transition_net = CausalTransitionNetwork(config)

        # The LM head enables text generation (weight-tied with embeddings)
        self.lm_head = LanguageModelingHead(config, self.encoder.token_embedding)

        # ---- Memory System (CPU-side stores + GPU-side projections) ----
        self.memory = MemoryManager(config)

        # ---- EMA Target Encoder ----
        # CRITICAL: The EMA encoder is a SEPARATE COPY that receives no gradients.
        # It is updated via exponential moving average AFTER the optimizer step.
        # Purpose: provide stable prediction targets for the causal loss.
        # Without EMA, the targets shift every step → training chases a moving target.
        self.ema_encoder = UnifiedEncoder(config)
        self.ema_compressor = CausalStateCompressor(config)
        self._init_ema()

        # Log model statistics
        logger.info(f"Initialized CausalWorldModel: {config}")
        logger.info(f"Trainable params: {self.count_parameters():,}")
        logger.info(f"Total params (incl. EMA): {self.count_parameters(include_ema=True):,}")

    def _init_ema(self):
        """
        Initialize EMA encoder as an exact copy of the main encoder,
        then disable all gradients.

        This is called once at model creation. After this, the EMA encoder
        is ONLY updated via update_ema(), never by gradient descent.
        """
        # Copy weights from main encoder to EMA encoder
        self.ema_encoder.load_state_dict(self.encoder.state_dict())
        self.ema_compressor.load_state_dict(self.compressor.state_dict())

        # Disable gradients — EMA encoder is never trained directly
        for param in self.ema_encoder.parameters():
            param.requires_grad = False
        for param in self.ema_compressor.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_ema(self, step: int) -> float:
        """
        Update EMA encoder weights AFTER the optimizer step.

        CRITICAL: This must be called AFTER optimizer.step(), not before.
        The EMA encoder tracks the main encoder with a momentum parameter:
            ema_param = momentum * ema_param + (1 - momentum) * main_param

        The momentum is annealed from ema_momentum_start to ema_momentum_end
        over ema_anneal_steps training steps:
        - Early training (high momentum): EMA responds slowly, providing
          stable targets while the main encoder is still noisy
        - Late training (low momentum): EMA tracks the main encoder more
          closely, providing fresher targets
        - Momentum schedule: cosine annealing from start to end

        Args:
            step: current training step (for momentum scheduling)

        Returns:
            current momentum value (for logging)
        """
        # Compute annealed momentum (cosine schedule)
        if step >= self.config.ema_anneal_steps:
            momentum = self.config.ema_momentum_end
        else:
            # Cosine annealing: smooth transition from start to end
            progress = step / self.config.ema_anneal_steps
            cos_factor = (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()) / 2.0
            momentum = (
                self.config.ema_momentum_end
                + (self.config.ema_momentum_start - self.config.ema_momentum_end) * cos_factor
            )

        # Update EMA encoder parameters
        for ema_param, main_param in zip(
            self.ema_encoder.parameters(), self.encoder.parameters()
        ):
            ema_param.data.mul_(momentum).add_(main_param.data, alpha=1.0 - momentum)

        # Update EMA compressor parameters
        for ema_param, main_param in zip(
            self.ema_compressor.parameters(), self.compressor.parameters()
        ):
            ema_param.data.mul_(momentum).add_(main_param.data, alpha=1.0 - momentum)

        return momentum

    # =========================================================================
    #  Core Forward Methods
    # =========================================================================

    def encode(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        modality_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_memory: bool = False,
        use_causal_mask: bool = True,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> torch.Tensor:
        """
        Encode inputs into latent vectors.

        This is the primary encoding method. It processes text, images, or both
        through the unified encoder, optionally retrieving and injecting
        episodic memories.

        Args:
            input_ids: (B, S) text token IDs
            images: (B, C, H, W) raw images
            modality_ids: (B, S) modality labels per position
            attention_mask: (B, S) padding mask
            use_memory: whether to retrieve and inject episodic memories
            use_causal_mask: whether to use causal attention masking
            kv_caches: per-layer KV caches for generation

        Returns:
            (B, S, hidden_dim) — encoder hidden states
        """
        # Retrieve memories if requested and available
        memory_vectors = None
        if use_memory and self.memory.episodic_store.size > 0:
            # We need the current state to query memory, but we haven't encoded yet.
            # Solution: do a lightweight "pre-encode" WITHOUT memory, extract state,
            # query memory, then do the FULL encode WITH memory.
            # This adds one encoder pass but is necessary for memory retrieval.
            # TODO(research): Investigate caching the pre-encode state to avoid
            # double encoding. At inference time this is fine; at training time
            # we typically don't use memory retrieval (it's added to the training
            # batch externally).
            with torch.no_grad():
                pre_hidden = self.encoder(
                    input_ids=input_ids,
                    images=images,
                    modality_ids=modality_ids,
                    attention_mask=attention_mask,
                    use_causal_mask=use_causal_mask,
                )
                pre_state = self.compressor(pre_hidden)
                memory_vectors = self.memory.retrieve_memories(
                    pre_state, device=pre_hidden.device,
                )

        # Full encode with memory injection
        hidden_states = self.encoder(
            input_ids=input_ids,
            images=images,
            modality_ids=modality_ids,
            attention_mask=attention_mask,
            memory_vectors=memory_vectors,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
        )

        return hidden_states

    def encode_state(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode inputs and compress to a single state vector.

        Convenience method that chains encode() → compress().

        Returns:
            (B, hidden_dim) — compressed causal state vector
        """
        hidden = self.encode(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            use_causal_mask=False,  # bidirectional for state extraction
        )

        # Create padding mask from attention_mask
        mask = attention_mask if attention_mask is not None else None
        state = self.compressor(hidden, mask=mask)

        return state

    @torch.no_grad()
    def ema_encode_state(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode using the EMA target encoder (no gradients).

        This produces the TARGETS for the causal prediction loss.
        The .detach() is implicit because the EMA encoder has requires_grad=False.

        Returns:
            (B, hidden_dim) — EMA-encoded state vector (detached)
        """
        hidden = self.ema_encoder(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            use_causal_mask=False,
        )
        mask = attention_mask if attention_mask is not None else None
        state = self.ema_compressor(hidden, mask=mask)
        return state

    def predict_next_state(
        self,
        state: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the next world state given current state and event.

        Args:
            state: (B, D) — current world state
            event: (B, D) — event/action embedding

        Returns:
            (B, D) — predicted next state
        """
        return self.transition_net(state, event)

    def predict_counterfactual(
        self,
        state: torch.Tensor,
        real_event: torch.Tensor,
        counterfactual_event: torch.Tensor,
    ) -> dict:
        """
        Predict both real and counterfactual outcomes.

        This is the key operation for testing Claim 4 (Counterfactual Calibration).

        Returns dict with real_next_state, cf_next_state, and causal_distance.
        """
        return self.transition_net.predict_counterfactual(
            state, real_event, counterfactual_event,
        )

    # =========================================================================
    #  Training Forward Pass
    # =========================================================================

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        event_ids: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        counterfactual_event_ids: Optional[torch.Tensor] = None,
        causal_distance_labels: Optional[torch.Tensor] = None,
        delta_text_ids: Optional[torch.Tensor] = None,
        delta_image: Optional[torch.Tensor] = None,
        delta_text_target_ids: Optional[torch.Tensor] = None,
        delta_image_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full training forward pass.

        This method computes all the representations needed for the four
        training losses. The actual loss computation happens in losses.py.

        This design separates representation computation from loss computation,
        which is cleaner and more modular — loss functions can be changed
        without modifying the forward pass.

        Args:
            input_ids: (B, S) observation text tokens
            images: (B, C, H, W) observation images
            attention_mask: (B, S) padding mask
            labels: (B, S) next-token labels for LM loss
            event_ids: (B, S_event) event description tokens
            target_ids: (B, S_target) next observation text tokens
            target_images: (B, C, H, W) next observation images
            counterfactual_event_ids: (B, S_cf) counterfactual event tokens
            causal_distance_labels: (B,) causal distance annotations [0, 1]
            delta_text_ids: (B, S) text observation at time t (for delta align)
            delta_image: (B, C, H, W) image at time t (for delta align)
            delta_text_target_ids: (B, S) text at t+1 (for delta align)
            delta_image_target: (B, C, H, W) image at t+1 (for delta align)

        Returns:
            Dictionary of tensors needed for loss computation:
            - encoder_output: (B, S, D) — hidden states from encoder
            - logits: (B, S, V) — LM logits
            - state_t: (B, D) — current state (from main encoder)
            - state_t1_pred: (B, D) — predicted next state (from transition net)
            - state_t1_target: (B, D) — actual next state (from EMA encoder)
            - state_cf: (B, D) — counterfactual next state
            - delta_z_text: (B, D) — text state transition delta
            - delta_z_image: (B, D) — image state transition delta
        """
        outputs = {}

        # ---- 1. Encode current observation ----
        encoder_output = self.encode(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            use_causal_mask=True,
        )
        outputs["encoder_output"] = encoder_output

        # ---- 2. LM logits (for L_lm) ----
        logits = self.lm_head(encoder_output)
        outputs["logits"] = logits

        # ---- 3. Compress to state vector ----
        state_t = self.compressor(encoder_output, mask=attention_mask)
        outputs["state_t"] = state_t

        # ---- 4. Causal prediction: state + event → predicted next state (for L_causal) ----
        if event_ids is not None:
            # Encode the event using the same encoder
            event_hidden = self.encoder(
                input_ids=event_ids,
                use_causal_mask=False,  # events are encoded bidirectionally
            )
            event_emb = self.compressor(event_hidden)
            outputs["event_embedding"] = event_emb

            # Predict next state
            state_t1_pred = self.transition_net(state_t, event_emb)
            outputs["state_t1_pred"] = state_t1_pred

            # Get EMA target for next observation
            if target_ids is not None or target_images is not None:
                state_t1_target = self.ema_encode_state(
                    input_ids=target_ids,
                    images=target_images,
                )
                outputs["state_t1_target"] = state_t1_target

        # ---- 5. Counterfactual prediction (for L_cf) ----
        if counterfactual_event_ids is not None and event_ids is not None:
            cf_hidden = self.encoder(
                input_ids=counterfactual_event_ids,
                use_causal_mask=False,
            )
            cf_event_emb = self.compressor(cf_hidden)
            outputs["cf_event_embedding"] = cf_event_emb

            state_cf = self.transition_net(state_t, cf_event_emb)
            outputs["state_cf"] = state_cf

            if causal_distance_labels is not None:
                outputs["causal_distance_labels"] = causal_distance_labels

        # ---- 6. Cross-modal delta alignment (for L_align) ----
        # CRITICAL: We compute the DELTA vectors (state transitions),
        # not the states themselves. We align how the world CHANGES,
        # not what the world LOOKS LIKE. This is fundamentally different
        # from CLIP, which aligns static representations.
        if delta_text_ids is not None and delta_image is not None:
            # Text state at time t and t+1
            text_state_t = self.encode_state(input_ids=delta_text_ids)
            text_state_t1 = self.encode_state(input_ids=delta_text_target_ids)
            # Image state at time t and t+1
            image_state_t = self.encode_state(images=delta_image)
            image_state_t1 = self.encode_state(images=delta_image_target)

            # Compute transition deltas (NOT absolute states)
            delta_z_text = text_state_t1 - text_state_t
            delta_z_image = image_state_t1 - image_state_t

            outputs["delta_z_text"] = delta_z_text
            outputs["delta_z_image"] = delta_z_image

        return outputs

    # =========================================================================
    #  Generation
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        use_memory: bool = True,
        store_as_episode: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with episodic memory.

        The generation loop:
        1. Process prompt through encoder (with optional memory retrieval)
        2. For each new token:
           a. Get logits from LM head
           b. Sample next token (top-p/temperature)
           c. Append to sequence
           d. Update KV cache
        3. Optionally store the full conversation as an episode

        Memory integration:
        - Before generation, retrieve relevant episodic memories
        - These are injected via cross-attention at the memory layer
        - After generation, store the conversation as a new episode
        - This means the model LEARNS from every conversation

        Args:
            input_ids: (B, S) — prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k filtering (0 to disable)
            eos_token_id: stop generation at this token
            use_memory: whether to use episodic memory
            store_as_episode: whether to store this interaction in memory

        Returns:
            (B, S + new_tokens) — full sequence including prompt and generated tokens
        """
        B = input_ids.shape[0]
        device = input_ids.device
        generated = input_ids.clone()

        # Retrieve memories once for the entire generation
        memory_vectors = None
        if use_memory and self.memory.episodic_store.size > 0:
            pre_hidden = self.encoder(input_ids=input_ids, use_causal_mask=True)
            pre_state = self.compressor(pre_hidden)
            memory_vectors = self.memory.retrieve_memories(pre_state, device=device)

        # Create KV caches for efficient generation
        kv_caches = self.encoder.create_kv_caches()

        # Process the prompt (prefill phase)
        hidden = self.encoder(
            input_ids=input_ids,
            memory_vectors=memory_vectors,
            use_causal_mask=True,
            kv_caches=kv_caches,
        )

        # Generate tokens one at a time (decode phase)
        for _ in range(max_new_tokens):
            # Sample next token from the last position's hidden state
            next_token = self.lm_head.generate(
                hidden,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )  # (B, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Encode the new token with KV cache (only process 1 token)
            hidden = self.encoder(
                input_ids=next_token,
                use_causal_mask=True,
                kv_caches=kv_caches,
            )

        # Optionally store this conversation as a new episode
        if store_as_episode and B == 1:
            # Encode the full conversation for memory storage
            full_hidden = self.encoder(input_ids=generated, use_causal_mask=False)
            full_state = self.compressor(full_hidden)

            # Use the prompt state as "before" and full state as "after"
            prompt_hidden = self.encoder(input_ids=input_ids, use_causal_mask=False)
            prompt_state = self.compressor(prompt_hidden)

            self.memory.store_episode(
                state=prompt_state[0],
                event=prompt_state[0],  # the prompt IS the event
                outcome=full_state[0],
                modality_tag=0,  # text
                metadata={"type": "conversation", "length": generated.shape[1]},
            )

        return generated

    # =========================================================================
    #  Utility Methods
    # =========================================================================

    def count_parameters(self, include_ema: bool = False) -> int:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if include_ema:
            total += sum(p.numel() for p in self.ema_encoder.parameters())
            total += sum(p.numel() for p in self.ema_compressor.parameters())
        return total

    def get_optimizer_groups(self, weight_decay: float = 0.1) -> list:
        """
        Separate parameters into groups for optimizer.

        Standard practice: apply weight decay to all 2D+ parameters (linear weights),
        but NOT to 1D parameters (norms, biases, embeddings).

        This is important because:
        - Weight decay on norms destabilizes training (norms have scale meaning)
        - Weight decay on embeddings can hurt rare token representations
        - Weight decay on linear weights acts as L2 regularization, preventing overfitting
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # No decay for 1D params: norms, biases, embeddings, gates
            if param.dim() <= 1 or "embedding" in name or "gate" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def save_checkpoint(self, path: str, step: int = 0, optimizer_state: Optional[dict] = None) -> None:
        """
        Save full model checkpoint including memory state.

        Saves:
        - Model weights (main encoder + transition net + LM head + compressor)
        - EMA encoder weights
        - Memory projections
        - Optimizer state (optional)
        - Config
        - Training step
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "step": step,
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, path)

        # Save memory separately (it's pickle-based, not torch-based)
        memory_dir = os.path.join(os.path.dirname(path) or ".", "memory")
        self.memory.save(memory_dir)

        logger.info(f"Saved checkpoint to {path} (step {step})")

    def load_checkpoint(self, path: str) -> dict:
        """
        Load model checkpoint and memory state.

        Returns the checkpoint dict (contains optimizer state, step, etc.)
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])

        # Load memory
        memory_dir = os.path.join(os.path.dirname(path) or ".", "memory")
        if os.path.exists(memory_dir):
            self.memory.load(memory_dir)

        logger.info(
            f"Loaded checkpoint from {path} (step {checkpoint.get('step', 'unknown')})"
        )
        return checkpoint

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for all encoder layers."""
        for layer in self.encoder.layers:
            layer._gradient_checkpointing = enable
        for layer in self.ema_encoder.layers:
            layer._gradient_checkpointing = False  # EMA never needs gradients

    def summary(self) -> str:
        """Print a human-readable model summary."""
        lines = [
            "=" * 60,
            "Causal-JEPA World Model Summary",
            "=" * 60,
            f"Config: {self.config}",
            f"",
            f"Components:",
            f"  Encoder:        {sum(p.numel() for p in self.encoder.parameters()):>12,} params",
            f"  Compressor:     {sum(p.numel() for p in self.compressor.parameters()):>12,} params",
            f"  Transition Net: {sum(p.numel() for p in self.transition_net.parameters()):>12,} params",
            f"  LM Head:        {sum(p.numel() for p in self.lm_head.parameters()):>12,} params",
            f"  Memory Proj:    {sum(p.numel() for p in self.memory.parameters()):>12,} params",
            f"  ───────────────────────────────────",
            f"  Trainable:      {self.count_parameters():>12,} params",
            f"  EMA (frozen):   {sum(p.numel() for p in self.ema_encoder.parameters()) + sum(p.numel() for p in self.ema_compressor.parameters()):>12,} params",
            f"",
            f"Architecture:",
            f"  Layers:         {self.config.num_layers}",
            f"  Hidden dim:     {self.config.hidden_dim}",
            f"  Attention:      {self.config.num_heads} heads ({self.config.num_kv_heads} KV heads, GQA×{self.config.gqa_groups})",
            f"  FFN:            {self.config.ffn_type} (dim={self.config.ffn_dim})",
            f"  Norm:           {self.config.norm_type}",
            f"  Position:       {self.config.pos_encoding}",
            f"  Memory inject:  layer {self.config.memory_inject_layer}/{self.config.num_layers}",
            f"  State queries:  {self.config.num_state_queries}",
            f"",
            f"Memory System:",
            f"  Episodic:       {self.memory.episodic_store.size}/{self.config.episodic_capacity} episodes",
            f"  Semantic:       {self.memory.semantic_memory.num_concepts} concepts",
            f"  Retrieval:      top-{self.config.topk_retrieve}",
            "=" * 60,
        ]
        return "\n".join(lines)
