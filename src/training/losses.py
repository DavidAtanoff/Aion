"""
Training Loss Functions for the Causal-JEPA World Model.

This module implements the four simultaneous training objectives that
differentiate our architecture from standard language models. Each loss
encodes a specific inductive bias about how intelligence should work.

LOSS SUMMARY:
=============
  L_total = 1.0 * L_causal + 0.3 * L_cf + 0.2 * L_align + 0.5 * L_lm

  L_causal:  "Predict the next world state from current state + event"
  L_cf:      "Counterfactual events should produce proportionally different states"
  L_align:   "The same causal transition expressed in text or images should match"
  L_lm:      "Generate coherent text (so we can talk to the model)"

WHY FOUR LOSSES SIMULTANEOUSLY:
================================
Each loss shapes a different aspect of the learned representations:

1. L_causal forces the model to build an explicit state transition function.
   Without it, the model has no incentive to separate "what the world is"
   from "how the world changes."

2. L_cf forces the representational space to have METRIC STRUCTURE aligned
   with causal distance. Without it, two events that produce similar outcomes
   might be far apart in embedding space (or vice versa).

3. L_align forces cross-modal causal consistency. Without it, the text
   encoder and image encoder could learn completely different transition
   structures — the model would "understand" causality differently in
   each modality.

4. L_lm ensures the model can communicate. Without it, the model builds
   useful internal representations but has no way to express them.

The weights (1.0, 0.3, 0.2, 0.5) are chosen so that:
- L_causal dominates: this is our novel contribution
- L_lm is secondary but significant: the model must be conversational
- L_cf and L_align are lower weight: they provide structural regularization
  but shouldn't overwhelm the primary objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CausalLatentPredictionLoss(nn.Module):
    """
    Loss 1: Causal Latent Prediction (primary, weight=1.0)

    Formula:
        z_pred = transition_net(state_t, event_t)
        z_true = ema_encoder(observation_{t+1}).detach()
        L_causal = 1 - cosine_similarity(z_pred, z_true).mean()

    THEORETICAL JUSTIFICATION:
    ==========================
    This is THE core training signal. It forces the model to learn:
      "Given the world in state S and event E, predict the resulting state."

    Key design choices:

    1. COSINE SIMILARITY (not MSE):
       - MSE in high-dimensional spaces is dominated by magnitude differences
       - Cosine similarity only compares DIRECTION, which is what matters
         for semantic representation
       - This prevents the model from collapsing to zero-magnitude vectors
         (which would minimize MSE trivially)

    2. EMA TARGETS (not online targets):
       - If we used the main encoder to produce targets, the model would be
         predicting its own shifting representations → representational collapse
       - The EMA encoder provides STABLE targets that change slowly
       - This is the same insight from BYOL, DINO, and I-JEPA

    3. NO RECONSTRUCTION LOSS:
       - We do NOT ask the model to reconstruct the raw observation
       - Reconstruction losses (like in VAEs) incentivize memorizing surface
         patterns (pixel values, exact word sequences)
       - By predicting in LATENT space, the model is forced to capture
         abstract, causal structure — not surface statistics
       - This is the fundamental difference from autoregressive LMs

    Range: [0.0, 2.0] — 0.0 = perfect prediction, 2.0 = opposite direction
    Typical good values: 0.1 - 0.4 after training
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_state: torch.Tensor,   # (B, D) — from transition network
        target_state: torch.Tensor,       # (B, D) — from EMA encoder (detached)
    ) -> torch.Tensor:
        """
        Compute causal latent prediction loss.

        Both inputs should be normalized (our transition_net and compressor
        both apply normalization). If not, we normalize here for safety.
        """
        # Normalize for cosine similarity (defensive — should already be normalized)
        pred_norm = F.normalize(predicted_state, dim=-1)
        target_norm = F.normalize(target_state, dim=-1)

        # Cosine similarity: 1.0 = identical, -1.0 = opposite
        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=-1)

        # Loss: 1 - cos_sim → 0.0 when perfectly aligned
        loss = (1.0 - cos_sim).mean()

        return loss


class CounterfactualConsistencyLoss(nn.Module):
    """
    Loss 2: Counterfactual Consistency (novel, weight=0.3)

    Formula:
        z_cf = transition_net(state_t, counterfactual_event)
        z_real = transition_net(state_t, real_event)
        causal_distance = label from dataset (float 0.0 - 1.0)
        L_cf = MSE(cosine_distance(z_cf, z_real), causal_distance)

    THEORETICAL JUSTIFICATION:
    ==========================
    This loss forces the model to build a METRIC SPACE where representational
    distance corresponds to causal distance. No existing training objective
    does this.

    Why it matters:
    - A standard model might place "ball hits window hard" and "ball taps
      window gently" far apart in embedding space (different words!)
    - But causally, they're similar events with measurably different magnitudes
    - This loss forces: distance(embed("hard hit"), embed("gentle tap")) ≈ 0.3
      because the causal distance is 0.3 (same mechanism, different magnitude)

    Causal distance labels:
    - 0.0 = identical event (same cause, same effect)
    - 0.5 = same category, different magnitude
    - 1.0 = completely different causal mechanism

    A model that learns this correctly can reason:
    "How different would the outcome be if the cause were slightly different?"
    This is the foundation of calibrated counterfactual reasoning.

    Range: [0.0, 1.0] — MSE between predicted and labeled causal distances
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        real_next_state: torch.Tensor,   # (B, D) — transition_net(state, real_event)
        cf_next_state: torch.Tensor,     # (B, D) — transition_net(state, cf_event)
        causal_distance_labels: torch.Tensor,  # (B,) — ground truth in [0, 1]
    ) -> torch.Tensor:
        """
        Compute counterfactual consistency loss.

        Measures whether the model's representational distance between
        real and counterfactual outcomes matches the labeled causal distance.
        """
        # Normalize for cosine computation
        real_norm = F.normalize(real_next_state, dim=-1)
        cf_norm = F.normalize(cf_next_state, dim=-1)

        # Compute cosine distance: 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite
        cos_sim = F.cosine_similarity(real_norm, cf_norm, dim=-1)
        predicted_distance = 1.0 - cos_sim  # Map to [0, 2], typically [0, 1]

        # Clamp to [0, 1] to match label range
        predicted_distance = predicted_distance.clamp(0.0, 1.0)

        # Ensure labels are in [0, 1]
        labels_clamped = causal_distance_labels.clamp(0.0, 1.0)

        # MSE between predicted causal distance and labeled causal distance
        loss = F.mse_loss(predicted_distance, labels_clamped)

        return loss


class CrossModalDeltaAlignmentLoss(nn.Module):
    """
    Loss 3: Cross-Modal Causal Alignment (novel, weight=0.2)

    Formula:
        delta_z_image = state_from_image_{t+1} - state_from_image_t
        delta_z_text  = state_from_text_{t+1}  - state_from_text_t
        L_align = 1 - cosine_similarity(delta_z_image, delta_z_text).mean()

    THEORETICAL JUSTIFICATION:
    ==========================
    CRITICAL DISTINCTION FROM CLIP:
    - CLIP aligns STATIC representations: embed("a cat") ≈ embed(photo_of_cat)
    - We align TRANSITION VECTORS: delta("ball flying") ≈ delta(video_of_ball_flying)
    - This is a fundamentally different claim about what should be shared across modalities

    Why delta alignment (not static alignment):
    - A text description of "a window" and a photo of "a window" should NOT
      necessarily have the same embedding — they contain different information
      (the text is abstract, the image has specific details like color, size)
    - BUT: the STATE CHANGE caused by "a ball hitting a window" should be the
      same regardless of whether you observe it as text or image
    - This forces the model to learn that causation is modality-invariant:
      the same cause produces the same transition in state space, whether
      you observe the cause through language or vision

    IMPLEMENTATION NOTE:
    The spec says "compute delta BEFORE pooling, not after." But our
    architecture pools first (CausalStateCompressor produces a single vector),
    so we compute delta on the pooled state vectors. This is a pragmatic
    simplification. The key insight — aligning transitions not states — is
    preserved.

    TODO(research): Investigate computing delta on the full encoder sequence
    (before pooling) by aligning per-position transition vectors. This would
    preserve more fine-grained causal information but is computationally
    more expensive and requires careful alignment of text/image positions.

    Range: [0.0, 2.0] — 0.0 = perfectly aligned transitions
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        delta_z_text: torch.Tensor,    # (B, D) — text state transition
        delta_z_image: torch.Tensor,   # (B, D) — image state transition
    ) -> torch.Tensor:
        """
        Align causal transition vectors across modalities.

        Both delta vectors represent the same causal event observed through
        different modalities. The loss pushes them to point the same direction.
        """
        # Normalize deltas — we care about the DIRECTION of change, not magnitude
        delta_text_norm = F.normalize(delta_z_text, dim=-1)
        delta_image_norm = F.normalize(delta_z_image, dim=-1)

        # Cosine similarity of transition vectors
        cos_sim = F.cosine_similarity(delta_text_norm, delta_image_norm, dim=-1)

        # Loss: 1 - cos_sim → 0.0 when transitions are aligned
        loss = (1.0 - cos_sim).mean()

        return loss


class LanguageModelingLoss(nn.Module):
    """
    Loss 4: Language Modeling (weight=0.5)

    Formula:
        L_lm = cross_entropy(lm_head(encoder_output), next_token_ids)

    THEORETICAL JUSTIFICATION:
    ==========================
    This is a standard causal language modeling loss, identical to GPT-2.
    We include it for two reasons:

    1. COMMUNICATION: Without an LM loss, the model learns rich internal
       representations but has no way to EXPRESS them in language. The LM
       head is the model's "mouth" — it converts latent understanding into
       words.

    2. COMPLEMENTARY SIGNAL: The LM loss provides dense, per-token gradients
       that complement the sparse, per-sequence gradients from L_causal.
       This helps the encoder learn fine-grained language understanding
       alongside coarse causal structure.

    Why weight=0.5 (not 1.0):
    - We don't want the LM objective to DOMINATE training — that would
      turn this into just another language model
    - At 0.5, the LM loss contributes significant gradient signal for
      language fluency, but the causal loss (weight=1.0) remains primary
    - The model should be "causal reasoner first, language model second"

    Label smoothing (default 0.1):
    - Prevents the model from becoming overconfident in next-token predictions
    - Distributes some probability mass to non-target tokens
    - Improves calibration, which is important for our counterfactual claims

    Range: depends on vocab size (log(vocab_size) ≈ 10.4 for 32K vocab at random)
    Good values: 2.0 - 4.0 after training on Alpaca-scale data
    """

    def __init__(self, vocab_size: int, pad_token_id: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,      # (B, S, vocab_size)
        labels: torch.Tensor,      # (B, S) — next-token IDs
        attention_mask: Optional[torch.Tensor] = None,  # (B, S) — 1=valid, 0=pad
    ) -> torch.Tensor:
        """
        Compute language modeling loss (next-token prediction).

        The logits are shifted internally: we predict token[i+1] from
        the hidden state at position[i]. This is standard causal LM practice.
        """
        # Shift: predict position i+1 from position i
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Reshape for cross-entropy: (B*(S-1), V) and (B*(S-1),)
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        loss = self.loss_fn(shift_logits, shift_labels)

        return loss


class CausalWorldModelLoss(nn.Module):
    """
    Combined loss for the Causal-JEPA World Model.

    Computes all four losses and combines them with configured weights.
    Only computes losses for which the required data is present in the
    model outputs — this allows training on partial data (e.g., text-only
    batches that don't have image or counterfactual data).

    Total loss: L = w1*L_causal + w2*L_cf + w3*L_align + w4*L_lm

    Default weights: 1.0, 0.3, 0.2, 0.5
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
        weight_causal: float = 1.0,
        weight_cf: float = 0.3,
        weight_align: float = 0.2,
        weight_lm: float = 0.5,
    ):
        super().__init__()

        self.weight_causal = weight_causal
        self.weight_cf = weight_cf
        self.weight_align = weight_align
        self.weight_lm = weight_lm

        # Individual loss modules
        self.causal_loss = CausalLatentPredictionLoss()
        self.cf_loss = CounterfactualConsistencyLoss()
        self.align_loss = CrossModalDeltaAlignmentLoss()
        self.lm_loss = LanguageModelingLoss(vocab_size, pad_token_id, label_smoothing)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all applicable losses from model outputs.

        Args:
            model_outputs: Dictionary from CausalWorldModel.forward() containing:
                - logits: (B, S, V) — for L_lm
                - state_t1_pred: (B, D) — for L_causal
                - state_t1_target: (B, D) — for L_causal
                - state_t1_pred: (B, D) — for L_cf (reused)
                - state_cf: (B, D) — for L_cf
                - causal_distance_labels: (B,) — for L_cf
                - delta_z_text: (B, D) — for L_align
                - delta_z_image: (B, D) — for L_align
            labels: (B, S) — next-token labels for LM loss
            attention_mask: (B, S) — padding mask

        Returns:
            Dictionary with individual losses and total:
                - loss: total combined loss (for backward())
                - loss_causal: L_causal value
                - loss_cf: L_cf value
                - loss_align: L_align value
                - loss_lm: L_lm value
                - losses_active: number of active losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self._get_device(model_outputs))
        active_count = 0

        # ---- Loss 1: Causal Latent Prediction ----
        if "state_t1_pred" in model_outputs and "state_t1_target" in model_outputs:
            l_causal = self.causal_loss(
                model_outputs["state_t1_pred"],
                model_outputs["state_t1_target"],
            )
            losses["loss_causal"] = l_causal
            total_loss = total_loss + self.weight_causal * l_causal
            active_count += 1

        # ---- Loss 2: Counterfactual Consistency ----
        if (
            "state_t1_pred" in model_outputs
            and "state_cf" in model_outputs
            and "causal_distance_labels" in model_outputs
        ):
            l_cf = self.cf_loss(
                model_outputs["state_t1_pred"],
                model_outputs["state_cf"],
                model_outputs["causal_distance_labels"],
            )
            losses["loss_cf"] = l_cf
            total_loss = total_loss + self.weight_cf * l_cf
            active_count += 1

        # ---- Loss 3: Cross-Modal Delta Alignment ----
        if "delta_z_text" in model_outputs and "delta_z_image" in model_outputs:
            l_align = self.align_loss(
                model_outputs["delta_z_text"],
                model_outputs["delta_z_image"],
            )
            losses["loss_align"] = l_align
            total_loss = total_loss + self.weight_align * l_align
            active_count += 1

        # ---- Loss 4: Language Modeling ----
        if "logits" in model_outputs and labels is not None:
            l_lm = self.lm_loss(
                model_outputs["logits"],
                labels,
                attention_mask=attention_mask,
            )
            losses["loss_lm"] = l_lm
            total_loss = total_loss + self.weight_lm * l_lm
            active_count += 1

        losses["loss"] = total_loss
        losses["losses_active"] = torch.tensor(float(active_count))

        return losses

    @staticmethod
    def _get_device(outputs: dict) -> torch.device:
        """Get device from any tensor in the outputs dict."""
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")
