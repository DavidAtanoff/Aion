"""
EMA (Exponential Moving Average) Scheduler for the target encoder.

The EMA encoder provides stable prediction targets for the causal loss.
This module handles the momentum scheduling and parameter update logic.

WHY EMA:
========
Without EMA, the causal loss would be:
    L = distance(transition_net(state), encoder(next_obs))

But the encoder is being trained! So the target shifts every step.
This creates a "chasing your own tail" dynamic that leads to
representational collapse (everything maps to the same vector).

EMA solves this by maintaining a slowly-moving copy of the encoder:
    ema_param = momentum * ema_param + (1 - momentum) * main_param

The EMA encoder changes slowly enough to provide stable targets,
but fast enough to track the main encoder's improving representations.

MOMENTUM SCHEDULE:
==================
- Start at 0.99: responsive early, when the encoder is changing rapidly
- Anneal to 0.999: stable late, when representations are converging
- Cosine schedule: smooth transition (no sudden jumps)

CRITICAL IMPLEMENTATION DETAIL:
The EMA update MUST happen AFTER optimizer.step(), not before.
If you update EMA before the optimizer step, the EMA encoder would
be one step AHEAD of the main encoder, creating a subtle but harmful
mismatch between the prediction source and target.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class EMAScheduler:
    """
    Manages EMA updates for the target encoder with momentum annealing.

    Usage:
        ema_scheduler = EMAScheduler(
            momentum_start=0.99,
            momentum_end=0.999,
            anneal_steps=10000,
        )

        # In training loop, AFTER optimizer.step():
        momentum = ema_scheduler.update(
            ema_model=model.ema_encoder,
            main_model=model.encoder,
            step=current_step,
        )
    """

    def __init__(
        self,
        momentum_start: float = 0.99,
        momentum_end: float = 0.999,
        anneal_steps: int = 10_000,
        schedule: str = "cosine",
    ):
        """
        Args:
            momentum_start: initial momentum (lower = more responsive)
            momentum_end: final momentum (higher = more stable)
            anneal_steps: steps over which to anneal momentum
            schedule: "cosine" or "linear" annealing
        """
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.anneal_steps = anneal_steps
        self.schedule = schedule

    def get_momentum(self, step: int) -> float:
        """
        Compute the current momentum value based on training step.

        Cosine schedule:
            momentum = end + (start - end) * (1 + cos(pi * step / total)) / 2

        This provides smooth annealing that spends more steps near
        the endpoints (slow at start and end, faster in the middle).
        """
        if step >= self.anneal_steps:
            return self.momentum_end

        if self.schedule == "cosine":
            progress = step / self.anneal_steps
            cos_factor = (1.0 + math.cos(math.pi * progress)) / 2.0
            return self.momentum_end + (self.momentum_start - self.momentum_end) * cos_factor
        elif self.schedule == "linear":
            progress = step / self.anneal_steps
            return self.momentum_start + (self.momentum_end - self.momentum_start) * progress
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    @torch.no_grad()
    def update(
        self,
        ema_model: nn.Module,
        main_model: nn.Module,
        step: int,
    ) -> float:
        """
        Update EMA model parameters from main model.

        MUST be called AFTER optimizer.step() in the training loop.

        The update rule for each parameter:
            ema_param = momentum * ema_param + (1 - momentum) * main_param

        When momentum is high (0.999), the EMA changes very slowly.
        When momentum is low (0.99), the EMA tracks the main model more closely.

        Args:
            ema_model: the EMA (target) model to update
            main_model: the main (online) model providing new parameters
            step: current training step (for momentum scheduling)

        Returns:
            The momentum value used for this update (for logging)
        """
        momentum = self.get_momentum(step)

        for ema_param, main_param in zip(ema_model.parameters(), main_model.parameters()):
            # In-place update: ema = momentum * ema + (1 - momentum) * main
            ema_param.data.mul_(momentum).add_(main_param.data, alpha=1.0 - momentum)

        return momentum

    @torch.no_grad()
    def update_multiple(
        self,
        ema_models: list,
        main_models: list,
        step: int,
    ) -> float:
        """
        Update multiple EMA model pairs at once (e.g., encoder + compressor).

        Args:
            ema_models: list of EMA modules
            main_models: list of corresponding main modules
            step: current training step

        Returns:
            momentum value used
        """
        momentum = self.get_momentum(step)

        for ema_model, main_model in zip(ema_models, main_models):
            for ema_param, main_param in zip(ema_model.parameters(), main_model.parameters()):
                ema_param.data.mul_(momentum).add_(main_param.data, alpha=1.0 - momentum)

        return momentum

    @staticmethod
    def initialize_ema(ema_model: nn.Module, main_model: nn.Module) -> None:
        """
        Initialize EMA model as an exact copy with gradients disabled.
        Call this once at model creation time.
        """
        ema_model.load_state_dict(main_model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
