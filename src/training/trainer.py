"""
Training Loop for the Causal-JEPA World Model.

This module implements the complete training pipeline with:
- Mixed precision (fp16 via torch.cuda.amp)
- Gradient accumulation (effective batch size > physical batch size)
- Cosine learning rate schedule with linear warmup
- EMA target encoder updates (after optimizer step)
- Early stopping on validation loss
- Periodic memory consolidation
- Checkpoint management
- Comprehensive logging

KAGGLE COMPATIBILITY:
====================
This trainer is designed to run on Kaggle's free GPU tier:
- 16GB VRAM (T4) — uses fp16 + gradient accumulation
- 30h/week GPU quota — checkpoints every epoch for resume
- Single GPU — no distributed training needed
- CPU-side memory — FAISS runs on CPU, no VRAM impact

The TrainingConfig dataclass contains all training hyperparameters,
separate from the model's WorldModelConfig. This separation is
intentional: the same model architecture can be trained with
different training regimes.
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from ..utils.config import WorldModelConfig
from ..model.world_model import CausalWorldModel
from .losses import CausalWorldModelLoss
from .ema import EMAScheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training hyperparameters — separate from model architecture config.

    This separation matters because the same model can be trained with
    different learning rates, batch sizes, etc. across different hardware.
    """

    # ========================
    #  Optimization
    # ========================
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5   # floor for cosine decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95                # lower than default 0.999 for more responsive adaptation
    max_grad_norm: float = 1.0         # gradient clipping threshold
    eps: float = 1e-8

    # ========================
    #  Batch & Accumulation
    # ========================
    # Physical batch size per GPU (limited by VRAM)
    batch_size: int = 8
    # Number of accumulation steps → effective batch = batch_size * accumulation_steps
    gradient_accumulation_steps: int = 4
    # Effective batch size = 8 * 4 = 32

    # ========================
    #  Schedule
    # ========================
    num_epochs: int = 10
    max_steps: Optional[int] = None     # if set, overrides num_epochs
    warmup_steps: int = 500             # linear warmup before cosine decay
    warmup_ratio: float = 0.0          # alternative: warmup as fraction of total steps

    # ========================
    #  Mixed Precision
    # ========================
    use_fp16: bool = True               # use fp16 mixed precision (for T4/V100)
    use_bf16: bool = False              # use bf16 (for A100+, more stable than fp16)

    # ========================
    #  Loss Weights
    # ========================
    weight_causal: float = 1.0
    weight_cf: float = 0.3
    weight_align: float = 0.2
    weight_lm: float = 0.5
    label_smoothing: float = 0.1

    # ========================
    #  Early Stopping
    # ========================
    early_stopping_patience: int = 3    # epochs without improvement
    early_stopping_min_delta: float = 0.001  # minimum improvement to count

    # ========================
    #  Checkpointing
    # ========================
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000      # save checkpoint every N steps
    save_every_n_epochs: int = 1        # save checkpoint every N epochs
    keep_n_checkpoints: int = 3         # keep only the N most recent

    # ========================
    #  Logging
    # ========================
    log_every_n_steps: int = 10         # log metrics every N steps
    eval_every_n_steps: int = 500       # run validation every N steps

    # ========================
    #  Memory
    # ========================
    store_episodes_during_training: bool = True
    episode_store_interval: int = 100    # store episode every N steps
    consolidation_interval: int = 2000   # consolidate memory every N steps

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class MetricsTracker:
    """
    Simple metrics tracker for training loss curves.
    Tracks running averages and supports logging to console.
    """

    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.step_metrics: Dict[str, float] = {}

    def update(self, metrics: Dict[str, float], step: int):
        """Record metrics for a step."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((step, value))
            self.step_metrics[key] = value

    def get_average(self, key: str, last_n: int = 100) -> float:
        """Get average of last N values for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        values = [v for _, v in self.metrics[key][-last_n:]]
        return sum(values) / len(values)

    def format_metrics(self, step: int, epoch: int, lr: float, momentum: float) -> str:
        """Format current metrics as a log string."""
        parts = [f"step={step}", f"epoch={epoch}", f"lr={lr:.2e}", f"ema_m={momentum:.4f}"]
        for key in ["loss", "loss_causal", "loss_cf", "loss_align", "loss_lm"]:
            if key in self.step_metrics:
                parts.append(f"{key}={self.step_metrics[key]:.4f}")
        return " | ".join(parts)


class CausalWorldModelTrainer:
    """
    Complete training loop for the Causal-JEPA World Model.

    This trainer handles the full training pipeline:
    1. Forward pass through CausalWorldModel
    2. Loss computation via CausalWorldModelLoss
    3. Backward pass with mixed precision and gradient accumulation
    4. Optimizer step with gradient clipping
    5. EMA update (AFTER optimizer step — critical ordering)
    6. Learning rate scheduling
    7. Periodic evaluation and early stopping
    8. Checkpoint management
    9. Episodic memory storage during training

    Usage:
        model = CausalWorldModel(model_config)
        trainer = CausalWorldModelTrainer(
            model=model,
            model_config=model_config,
            training_config=training_config,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
        )
        trainer.train()

    The trainer accepts standard PyTorch DataLoaders. Each batch should be
    a dictionary with keys matching CausalWorldModel.forward() args:
        {
            "input_ids": (B, S),
            "attention_mask": (B, S),
            "labels": (B, S),           # next-token labels for L_lm
            "event_ids": (B, S_e),      # optional: for L_causal
            "target_ids": (B, S_t),     # optional: for L_causal
            ...
        }
    """

    def __init__(
        self,
        model: CausalWorldModel,
        model_config: WorldModelConfig,
        training_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        # Loss function
        self.criterion = CausalWorldModelLoss(
            vocab_size=model_config.vocab_size,
            pad_token_id=model_config.pad_token_id,
            label_smoothing=training_config.label_smoothing,
            weight_causal=training_config.weight_causal,
            weight_cf=training_config.weight_cf,
            weight_align=training_config.weight_align,
            weight_lm=training_config.weight_lm,
        )

        # Optimizer — AdamW with separate weight decay groups
        self.optimizer = torch.optim.AdamW(
            model.get_optimizer_groups(training_config.weight_decay),
            lr=training_config.learning_rate,
            betas=(training_config.beta1, training_config.beta2),
            eps=training_config.eps,
        )

        # Mixed precision — determine device type for autocast context
        self.use_amp = training_config.use_fp16 or training_config.use_bf16
        self.amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
        if training_config.use_bf16:
            self.amp_dtype = torch.bfloat16
        elif training_config.use_fp16:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # GradScaler only works with CUDA fp16 — disable on CPU and bf16
        scaler_enabled = training_config.use_fp16 and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=scaler_enabled)

        # EMA scheduler
        self.ema_scheduler = EMAScheduler(
            momentum_start=model_config.ema_momentum_start,
            momentum_end=model_config.ema_momentum_end,
            anneal_steps=model_config.ema_anneal_steps,
        )

        # Metrics
        self.metrics = MetricsTracker()

        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.current_momentum = model_config.ema_momentum_start

        # Compute total training steps
        self.steps_per_epoch = len(train_loader) // training_config.gradient_accumulation_steps
        if training_config.max_steps is not None:
            self.total_steps = training_config.max_steps
        else:
            self.total_steps = self.steps_per_epoch * training_config.num_epochs

        # Compute warmup steps
        if training_config.warmup_ratio > 0:
            self.warmup_steps = int(self.total_steps * training_config.warmup_ratio)
        else:
            self.warmup_steps = training_config.warmup_steps

        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {'fp16' if training_config.use_fp16 else 'bf16' if training_config.use_bf16 else 'fp32'}")
        logger.info(f"  Effective batch size: {training_config.effective_batch_size}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Steps per epoch: {self.steps_per_epoch}")

    def get_lr(self, step: int) -> float:
        """
        Cosine learning rate schedule with linear warmup.

        Phase 1 (step < warmup_steps): linear increase from 0 to max_lr
        Phase 2 (step >= warmup_steps): cosine decay from max_lr to min_lr

        This schedule is standard for transformer training:
        - Warmup prevents early instability (large gradients on random weights)
        - Cosine decay provides a smooth reduction without sudden drops
        """
        max_lr = self.config.learning_rate
        min_lr = self.config.min_learning_rate

        if step < self.warmup_steps:
            # Linear warmup
            return max_lr * (step + 1) / (self.warmup_steps + 1)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in a batch dict to the training device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        return device_batch

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute one training step (one gradient accumulation cycle).

        This is where everything comes together:
        1. Forward pass through the model (with AMP)
        2. Compute all applicable losses
        3. Scale loss for gradient accumulation
        4. Backward pass through scaler
        5. (After accumulation_steps) Clip gradients, optimizer step, EMA update

        Returns dict of loss values for logging.
        """
        self.model.train()
        batch = self.move_batch_to_device(batch)

        # Extract labels before forward pass (some args go to model, labels go to loss)
        labels = batch.pop("labels", None)
        attention_mask = batch.get("attention_mask", None)

        # Forward pass with automatic mixed precision
        with autocast(device_type=self.amp_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
            model_outputs = self.model(**batch)
            loss_outputs = self.criterion(model_outputs, labels=labels, attention_mask=attention_mask)
            loss = loss_outputs["loss"]

            # Scale loss for gradient accumulation
            # If accumulating over N steps, each step contributes 1/N of the gradient
            scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass (through scaler for fp16 stability)
        self.scaler.scale(scaled_loss).backward()

        # Return loss values (unscaled) for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_outputs.items()}

    def optimizer_step(self):
        """
        Execute optimizer step with gradient clipping.
        Called after gradient_accumulation_steps backward passes.

        ORDERING IS CRITICAL:
        1. Unscale gradients (for correct clipping magnitude)
        2. Clip gradients (prevent exploding gradients)
        3. Optimizer step (update weights)
        4. Scaler update (adjust fp16 loss scale)
        5. Zero gradients (prepare for next accumulation cycle)
        6. EMA update (AFTER optimizer step — the EMA must track the UPDATED weights)
        7. Learning rate update
        """
        # Unscale before clipping so clip threshold is in true gradient scale
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping — prevents training instability from outlier batches
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Zero gradients for next accumulation cycle
        self.optimizer.zero_grad(set_to_none=True)

        # EMA update — MUST be after optimizer.step()
        self.current_momentum = self.ema_scheduler.update_multiple(
            ema_models=[self.model.ema_encoder, self.model.ema_compressor],
            main_models=[self.model.encoder, self.model.compressor],
            step=self.global_step,
        )

        # Update learning rate
        lr = self.get_lr(self.global_step)
        self.set_lr(lr)

        self.global_step += 1

        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation and return average loss.

        Uses the same loss computation as training but without
        gradient computation or accumulation.
        """
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = self.move_batch_to_device(batch)
            labels = batch.pop("labels", None)
            attention_mask = batch.get("attention_mask", None)

            with autocast(device_type=self.amp_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                model_outputs = self.model(**batch)
                loss_outputs = self.criterion(model_outputs, labels=labels, attention_mask=attention_mask)

            total_loss += loss_outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop based on validation loss.

        Returns True if training should continue, False if it should stop.
        """
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            # Improvement found
            self.best_val_loss = val_loss
            self.patience_counter = 0
            logger.info(f"New best validation loss: {val_loss:.4f}")
            return True
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement for {self.patience_counter}/{self.config.early_stopping_patience} epochs "
                f"(best: {self.best_val_loss:.4f}, current: {val_loss:.4f})"
            )
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered.")
                return False
            return True

    def save_checkpoint(self, tag: str = ""):
        """Save a training checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        filename = f"checkpoint_step{self.global_step}"
        if tag:
            filename += f"_{tag}"
        filepath = os.path.join(self.config.checkpoint_dir, f"{filename}.pt")

        self.model.save_checkpoint(
            path=filepath,
            step=self.global_step,
            optimizer_state=self.optimizer.state_dict(),
        )

        # Save training state separately
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(
            training_state,
            os.path.join(self.config.checkpoint_dir, f"{filename}_training_state.pt"),
        )

        logger.info(f"Saved checkpoint: {filepath}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint."""
        checkpoint = self.model.load_checkpoint(path)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training state
        state_path = path.replace(".pt", "_training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location="cpu")
            self.global_step = training_state["global_step"]
            self.current_epoch = training_state["current_epoch"]
            self.best_val_loss = training_state["best_val_loss"]
            self.patience_counter = training_state["patience_counter"]
            if "scaler_state" in training_state:
                self.scaler.load_state_dict(training_state["scaler_state"])

        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")

    def _cleanup_checkpoints(self):
        """Keep only the N most recent checkpoints."""
        if not os.path.exists(self.config.checkpoint_dir):
            return

        checkpoints = sorted([
            f for f in os.listdir(self.config.checkpoint_dir)
            if f.startswith("checkpoint_step") and f.endswith(".pt")
            and "_training_state" not in f
        ])

        while len(checkpoints) > self.config.keep_n_checkpoints:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.config.checkpoint_dir, oldest))
            # Also remove corresponding training state
            state_file = oldest.replace(".pt", "_training_state.pt")
            state_path = os.path.join(self.config.checkpoint_dir, state_file)
            if os.path.exists(state_path):
                os.remove(state_path)

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns the average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        accumulated_losses = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Training step (forward + backward, no optimizer step yet)
            step_losses = self.train_step(batch)

            # Accumulate metrics
            for k, v in step_losses.items():
                if isinstance(v, (int, float)):
                    accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v

            # Optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                grad_norm = self.optimizer_step()

                # Average accumulated losses
                avg_losses = {
                    k: v / self.config.gradient_accumulation_steps
                    for k, v in accumulated_losses.items()
                }
                avg_losses["grad_norm"] = grad_norm

                # Track metrics
                self.metrics.update(avg_losses, self.global_step)

                epoch_loss += avg_losses.get("loss", 0.0)
                epoch_steps += 1

                # Log
                if self.global_step % self.config.log_every_n_steps == 0:
                    lr = self.get_lr(self.global_step)
                    log_str = self.metrics.format_metrics(
                        self.global_step, self.current_epoch, lr, self.current_momentum,
                    )
                    logger.info(log_str)

                # Periodic validation
                if (
                    self.val_loader is not None
                    and self.global_step % self.config.eval_every_n_steps == 0
                    and self.global_step > 0
                ):
                    val_loss = self.validate()
                    self.metrics.update({"val_loss": val_loss}, self.global_step)
                    self.model.train()

                # Periodic checkpoint
                if (
                    self.global_step % self.config.save_every_n_steps == 0
                    and self.global_step > 0
                ):
                    self.save_checkpoint()

                # Periodic memory consolidation
                if (
                    self.config.store_episodes_during_training
                    and self.global_step % self.config.consolidation_interval == 0
                    and self.global_step > 0
                ):
                    self.model.memory.consolidate()

                # Reset accumulator
                accumulated_losses = {}

                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

        return epoch_loss / max(epoch_steps, 1)

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Runs for num_epochs (or max_steps), with validation and
        early stopping. Returns a summary dict of the training run.

        This is the main entry point for training:
            trainer.train()
        """
        logger.info("=" * 60)
        logger.info("Starting Causal-JEPA World Model Training")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_config}")
        logger.info(f"Training config: {self.config}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info("")

        start_time = time.time()
        training_complete = False

        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                logger.info(f"--- Epoch {epoch + 1}/{self.config.num_epochs} ---")

                # Train one epoch
                avg_train_loss = self.train_epoch()
                epoch_time = time.time() - epoch_start

                logger.info(
                    f"Epoch {epoch + 1} complete: "
                    f"avg_loss={avg_train_loss:.4f}, "
                    f"time={epoch_time:.1f}s, "
                    f"memory_episodes={self.model.memory.episodic_store.size}"
                )

                # Validation
                if self.val_loader is not None and (epoch + 1) % self.config.save_every_n_epochs == 0:
                    val_loss = self.validate()
                    self.metrics.update({"val_loss": val_loss}, self.global_step)

                    # Early stopping check
                    if not self.check_early_stopping(val_loss):
                        logger.info("Early stopping — restoring best checkpoint.")
                        training_complete = True
                        break

                # Save epoch checkpoint
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self.save_checkpoint(tag=f"epoch{epoch + 1}")

                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max_steps ({self.config.max_steps}). Stopping.")
                    training_complete = True
                    break

            if not training_complete:
                training_complete = True

        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving checkpoint...")
            self.save_checkpoint(tag="interrupted")

        # Save final checkpoint
        self.save_checkpoint(tag="final")

        total_time = time.time() - start_time
        summary = {
            "total_steps": self.global_step,
            "total_epochs": self.current_epoch + 1,
            "total_time_seconds": total_time,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.metrics.get_average("loss", last_n=100),
            "memory_episodes": self.model.memory.episodic_store.size,
            "memory_concepts": self.model.memory.semantic_memory.num_concepts,
        }

        logger.info("=" * 60)
        logger.info("Training Complete")
        logger.info(f"  Steps: {summary['total_steps']}")
        logger.info(f"  Time: {summary['total_time_seconds']:.1f}s")
        logger.info(f"  Best val loss: {summary['best_val_loss']:.4f}")
        logger.info(f"  Final train loss: {summary['final_train_loss']:.4f}")
        logger.info(f"  Episodic memories: {summary['memory_episodes']}")
        logger.info(f"  Semantic concepts: {summary['memory_concepts']}")
        logger.info("=" * 60)

        return summary
