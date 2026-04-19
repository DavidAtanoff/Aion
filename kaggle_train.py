#!/usr/bin/env python3
"""
Kaggle Training Script for the Causal-JEPA World Model.
=======================================================

This script handles EVERYTHING needed to train on Kaggle:
  - Auto-downloads dataset from HuggingFace (or loads from Kaggle input)
  - Tokenizes with GPT-2 tokenizer (auto-downloaded)
  - Formats data as causal triples (observation, event, outcome)
  - Supports 2x T4 GPUs via DataParallel or DDP
  - Mixed precision fp16 training
  - Checkpoint save/resume for Kaggle's time limits

DATASET:
========
Primary:   yahma/alpaca-cleaned (52K instruction pairs, ~13M tokens)
           - Clean, deduplicated version of Stanford Alpaca
           - Clear causal structure: instruction (cause) -> response (effect)
           - Trains in ~1-2 hours on 2x T4

The data is formatted as causal triples:
  - Observation (input_ids):  "### Instruction: {instruction}\n### Input: {input}\n### Response:"
  - Event (event_ids):        The instruction itself (the causal trigger)
  - Outcome (target_ids):     The response (the resulting state after the event)
  - Labels:                   Next-token prediction targets (= full text, shifted)

This framing teaches the model:
  "Given a world state described by the instruction context,
   and an event described by the instruction itself,
   the world transitions to a state described by the response."

MULTI-GPU:
==========
  Single GPU:     python kaggle_train.py
  DataParallel:   python kaggle_train.py                    (auto-detects 2 GPUs)
  DDP (fastest):  torchrun --nproc_per_node=2 kaggle_train.py --ddp

USAGE ON KAGGLE:
================
  1. Upload causal_jepa.py and kaggle_train.py to your notebook
  2. In a notebook cell, run:  !python kaggle_train.py
  3. Checkpoints are saved to /kaggle/working/checkpoints/
  4. Monitor training via the notebook output
"""

import os
import sys

# ===========================================================================
#  Authenication setup
# ===========================================================================
# Injected HF Token to prevent API throttling on Kaggle
# os.environ["HF_TOKEN"] = "your_token_here"  # Set your HF token for private datasets

import math
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

# Multi-GPU
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# HuggingFace
from datasets import load_dataset
from transformers import AutoTokenizer

# Our compiled model
from causal_jepa import (
    WorldModelConfig,
    CausalWorldModel,
    CausalWorldModelLoss,
    EMAScheduler,
)

# ===========================================================================
#  Configuration
# ===========================================================================

IS_KAGGLE = os.path.exists("/kaggle")

@dataclass
class KaggleTrainingConfig:
    """All training hyperparameters in one place."""

    # ---- Model ----
    model_scale: str = "base"       # tiny=13M, base=42M, large=111M, xl=323M

    # ---- Dataset ----
    dataset_name: str = "teknium/OpenHermes-2.5"
    tokenizer_name: str = "gpt2"
    max_seq_length: int = 256       # max tokens for observation
    max_event_length: int = 64      # max tokens for event (instruction)

    # ---- Training ----
    batch_size: int = 16            # per GPU
    gradient_accumulation_steps: int = 2  # effective batch = 16 * 2 * 2 GPUs = 64
    num_epochs: int = 2
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_ratio: float = 0.05     # 5% of total steps
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # ---- Mixed Precision ----
    use_fp16: bool = True

    # ---- Loss Weights ----
    weight_causal: float = 1.0      # L_causal: state transition prediction
    weight_lm: float = 0.5          # L_lm: next-token prediction
    weight_cf: float = 0.3          # L_cf: counterfactual (if data available)
    weight_align: float = 0.2       # L_align: cross-modal (if data available)
    label_smoothing: float = 0.1

    # ---- Checkpointing ----
    save_dir: str = "/kaggle/working/checkpoints" if IS_KAGGLE else "./checkpoints"
    log_every: int = 25             # log every N optimizer steps
    save_every_epoch: bool = True
    resume_from: Optional[str] = None

    @property
    def effective_batch_size(self):
        ngpu = max(torch.cuda.device_count(), 1)
        return self.batch_size * self.gradient_accumulation_steps * ngpu


# ===========================================================================
#  Dataset: Alpaca -> Causal Triples
# ===========================================================================

class AlpacaCausalDataset(Dataset):
    """
    Wraps the Alpaca dataset as causal training triples.

    Each example produces:
      input_ids:       Tokenized full text (instruction + input + response)
      attention_mask:  1 for real tokens, 0 for padding
      labels:          Same as input_ids (shifted internally by the LM loss)
      event_ids:       Tokenized instruction only (the causal event)
      target_ids:      Tokenized response only (the causal outcome)

    The causal framing:
      - The INSTRUCTION is the EVENT that acts on the world
      - The INPUT (if any) is additional CONTEXT about the world state
      - The RESPONSE is the OUTCOME of applying the event to the state
    """

    def __init__(self, dataset, tokenizer, max_seq_length=256, max_event_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_event_length = max_event_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Support for ShareGPT format (OpenHermes-2.5)
        if "conversations" in example:
            convs = example["conversations"]
            instruction = next((c["value"] for c in convs if c["from"] in ("human", "user")), "")
            output_text = next((c["value"] for c in convs if c["from"] in ("gpt", "assistant", "model")), "")
            input_text = ""
        # Support for Alpaca format
        else:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "") or ""
            output_text = example.get("output", "")

        # Build the full causal prompt
        if input_text.strip():
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        full_text = prompt + output_text

        # Tokenize the full observation (for LM loss)
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize the instruction as the causal event
        event_enc = self.tokenizer(
            instruction,
            truncation=True,
            max_length=self.max_event_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize the response as the causal outcome (target state)
        target_enc = self.tokenizer(
            output_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": full_enc["input_ids"].squeeze(0),
            "attention_mask": full_enc["attention_mask"].squeeze(0),
            "labels": full_enc["input_ids"].squeeze(0),           # next-token targets
            "event_ids": event_enc["input_ids"].squeeze(0),
            "target_ids": target_enc["input_ids"].squeeze(0),
        }


def load_data(config: KaggleTrainingConfig, tokenizer):
    """
    Load and prepare the training dataset.

    Tries:
      1. Local Kaggle input directory (if dataset was attached)
      2. HuggingFace download (auto-cached)
    """
    logger = logging.getLogger(__name__)

    # Try loading from Kaggle input
    dataset_short = config.dataset_name.split("/")[-1]
    kaggle_path = f"/kaggle/input/{dataset_short}"

    if IS_KAGGLE and os.path.exists(kaggle_path):
        logger.info(f"Loading dataset from Kaggle input: {kaggle_path}")
        json_files = [
            os.path.join(kaggle_path, f)
            for f in os.listdir(kaggle_path)
            if f.endswith(".json")
        ]
        if json_files:
            raw_dataset = load_dataset("json", data_files=json_files, split="train")
        else:
            # Try as a HuggingFace dataset directory
            raw_dataset = load_dataset(kaggle_path, split="train")
    else:
        logger.info(f"Downloading dataset from HuggingFace: {config.dataset_name}")
        raw_dataset = load_dataset(config.dataset_name, split="train")

    logger.info(f"Dataset loaded: {len(raw_dataset)} examples")

    # Split into train/val (95/5)
    split = raw_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split["train"]
    val_data = split["test"]

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = AlpacaCausalDataset(
        train_data, tokenizer,
        max_seq_length=config.max_seq_length,
        max_event_length=config.max_event_length,
    )
    val_dataset = AlpacaCausalDataset(
        val_data, tokenizer,
        max_seq_length=config.max_seq_length,
        max_event_length=config.max_event_length,
    )

    return train_dataset, val_dataset


# ===========================================================================
#  Multi-GPU Setup
# ===========================================================================

def setup_distributed(use_ddp: bool):
    """
    Setup multi-GPU training.

    Returns: (rank, world_size, is_ddp)

    Modes:
      - DDP (torchrun):  Best performance, requires torchrun launcher
      - DataParallel:    Simpler, works in notebooks, slightly less efficient
      - Single GPU:      Fallback
    """
    if use_ddp and "RANK" in os.environ:
        # Launched with torchrun -> full DDP
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world_size, True

    # Notebook / single-process mode
    world_size = torch.cuda.device_count()
    return 0, world_size, False


def cleanup_distributed(is_ddp: bool):
    """Cleanup DDP process group."""
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


# ===========================================================================
#  Training Loop
# ===========================================================================

def get_lr(step: int, warmup_steps: int, total_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(config: KaggleTrainingConfig, use_ddp: bool = False):
    """
    Complete training pipeline for Kaggle.

    This function handles:
      1. Tokenizer + dataset loading
      2. Model creation + multi-GPU wrapping
      3. Training loop with AMP, gradient accumulation, EMA
      4. Validation + checkpointing
    """
    # ---- Setup ----
    rank, world_size, is_ddp = setup_distributed(use_ddp)
    is_main = (rank == 0)  # only main process logs and saves

    # Logging (only on main process)
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # CRITICAL: Overrides Kaggle/Jupyter's default silent logger
    )
    logger = logging.getLogger(__name__)

    if is_main:
        logger.info("=" * 60)
        logger.info("  Causal-JEPA World Model — Kaggle Training")
        logger.info("=" * 60)
        logger.info(f"  GPUs: {world_size}x {'T4' if IS_KAGGLE else 'GPU'}")
        logger.info(f"  Mode: {'DDP' if is_ddp else 'DataParallel' if world_size > 1 else 'Single GPU'}")
        logger.info(f"  Model scale: {config.model_scale}")
        logger.info(f"  Dataset: {config.dataset_name}")
        logger.info(f"  Effective batch size: {config.effective_batch_size}")

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ---- Tokenizer ----
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    train_dataset, val_dataset = load_data(config, tokenizer)

    # DataLoaders with optional distributed sampling
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    # ---- Model ----
    model_config = getattr(WorldModelConfig, config.model_scale)()
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_token_id = tokenizer.pad_token_id

    model = CausalWorldModel(model_config)
    raw_model = model  # keep reference for EMA and checkpoint access

    if is_main:
        logger.info(f"\n{raw_model.summary()}\n")

    model = model.to(device)

    # Multi-GPU wrapping
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        logger.info(f"[Rank {rank}] DDP initialized")
    elif world_size > 1:
        model = DataParallel(model)
        logger.info(f"DataParallel across {world_size} GPUs")

    # ---- Loss ----
    criterion = CausalWorldModelLoss(
        vocab_size=model_config.vocab_size,
        pad_token_id=model_config.pad_token_id,
        label_smoothing=config.label_smoothing,
        weight_causal=config.weight_causal,
        weight_cf=config.weight_cf,
        weight_align=config.weight_align,
        weight_lm=config.weight_lm,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        raw_model.get_optimizer_groups(config.weight_decay),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )

    # ---- AMP ----
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler_enabled = config.use_fp16 and device.type == "cuda"
    scaler = GradScaler(enabled=scaler_enabled)

    # ---- EMA ----
    ema_scheduler = EMAScheduler(
        momentum_start=model_config.ema_momentum_start,
        momentum_end=model_config.ema_momentum_end,
        anneal_steps=model_config.ema_anneal_steps,
    )

    # ---- Schedule ----
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio) if config.warmup_ratio > 0 else 200

    if is_main:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    # ---- Resume ----
    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from: {config.resume_from}")
        ckpt = raw_model.load_checkpoint(config.resume_from)
        global_step = ckpt.get("step", 0)
        start_epoch = global_step // steps_per_epoch
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ================================================================
    #  TRAINING LOOP
    # ================================================================
    if is_main:
        logger.info("\n" + "=" * 60)
        logger.info("  Training started")
        logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        accum_losses = {}

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            attention_mask = batch.get("attention_mask")

            # Forward + loss (with AMP)
            with autocast(device_type=amp_device_type, enabled=config.use_fp16):
                outputs = model(**batch)

                # Handle DataParallel output (already reduced)
                loss_dict = criterion(outputs, labels=labels, attention_mask=attention_mask)
                loss = loss_dict["loss"]
                scaled_loss = loss / config.gradient_accumulation_steps

            # Backward
            scaler.scale(scaled_loss).backward()

            # Track losses
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                accum_losses[k] = accum_losses.get(k, 0.0) + val

            # Optimizer step every N accumulation steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Unscale + clip
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), config.max_grad_norm,
                )

                # Step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA update (AFTER optimizer step — critical)
                momentum = ema_scheduler.update_multiple(
                    ema_models=[raw_model.ema_encoder, raw_model.ema_compressor],
                    main_models=[raw_model.encoder, raw_model.compressor],
                    step=global_step,
                )

                # LR schedule
                lr = get_lr(global_step, warmup_steps, total_steps,
                            config.learning_rate, config.min_learning_rate)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                global_step += 1

                # Average accumulated losses
                avg_losses = {k: v / config.gradient_accumulation_steps
                              for k, v in accum_losses.items()}
                epoch_loss += avg_losses.get("loss", 0.0)
                epoch_steps += 1

                # Log
                if is_main and global_step % config.log_every == 0:
                    elapsed = time.time() - start_time
                    parts = [
                        f"step={global_step}/{total_steps}",
                        f"epoch={epoch+1}",
                        f"lr={lr:.2e}",
                        f"ema={momentum:.4f}",
                    ]
                    for k in ["loss", "loss_causal", "loss_lm"]:
                        if k in avg_losses:
                            parts.append(f"{k}={avg_losses[k]:.4f}")
                    if isinstance(grad_norm, torch.Tensor):
                        parts.append(f"gnorm={grad_norm.item():.2f}")
                    parts.append(f"t={elapsed:.0f}s")
                    logger.info(" | ".join(parts))

                accum_losses = {}

        # ---- End of epoch ----
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

        # Validation
        if val_loader is not None:
            val_loss = validate(model, criterion, val_loader, device,
                                config.use_fp16, amp_device_type)
            if is_main:
                logger.info(
                    f"Epoch {epoch+1}/{config.num_epochs} complete | "
                    f"train_loss={avg_epoch_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"memory={raw_model.memory.episodic_store.size} episodes"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"  New best val loss: {best_val_loss:.4f}")
        else:
            if is_main:
                logger.info(
                    f"Epoch {epoch+1}/{config.num_epochs} complete | "
                    f"train_loss={avg_epoch_loss:.4f}"
                )

        # Save checkpoint
        if is_main and config.save_every_epoch:
            os.makedirs(config.save_dir, exist_ok=True)
            ckpt_path = os.path.join(config.save_dir, f"epoch_{epoch+1}.pt")
            raw_model.save_checkpoint(
                ckpt_path, step=global_step,
                optimizer_state=optimizer.state_dict(),
            )
            logger.info(f"  Saved checkpoint: {ckpt_path}")

    # ---- Training complete ----
    total_time = time.time() - start_time
    if is_main:
        logger.info("\n" + "=" * 60)
        logger.info("  Training Complete!")
        logger.info("=" * 60)
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Total steps: {global_step}")
        logger.info(f"  Best val loss: {best_val_loss:.4f}")
        logger.info(f"  Episodic memories: {raw_model.memory.episodic_store.size}")
        logger.info(f"  Checkpoints: {config.save_dir}")

        # Save final
        final_path = os.path.join(config.save_dir, "final.pt")
        raw_model.save_checkpoint(final_path, step=global_step)
        logger.info(f"  Final model: {final_path}")

    cleanup_distributed(is_ddp)


@torch.no_grad()
def validate(model, criterion, val_loader, device, use_fp16, amp_device_type):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        attention_mask = batch.get("attention_mask")

        with autocast(device_type=amp_device_type, enabled=use_fp16):
            outputs = model(**batch)
            loss_dict = criterion(outputs, labels=labels, attention_mask=attention_mask)

        total_loss += loss_dict["loss"].item()
        num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


# ===========================================================================
#  Entry Point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-JEPA Kaggle Training")

    parser.add_argument("--scale", type=str, default="base",
                        choices=["tiny", "base", "large", "xl"],
                        help="Model scale preset (default: base = 42M params)")
    parser.add_argument("--dataset", type=str, default="teknium/OpenHermes-2.5",
                        help="HuggingFace dataset name or Kaggle input path")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--max-seq-length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--ddp", action="store_true",
                        help="Use DDP instead of DataParallel (requires torchrun)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-dir", type=str,
                        default="/kaggle/working/checkpoints" if IS_KAGGLE else "./checkpoints",
                        help="Directory for checkpoints")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = KaggleTrainingConfig(
        model_scale=args.scale,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        save_dir=args.save_dir,
        resume_from=args.resume,
    )

    train(config, use_ddp=args.ddp)
