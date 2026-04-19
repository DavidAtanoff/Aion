"""Verification for training losses and trainer."""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("  Training Module Verification")
print("=" * 60)

# Test 1: Individual losses
print("\n[1/4] Testing individual losses...")
from src.training.losses import (
    CausalLatentPredictionLoss,
    CounterfactualConsistencyLoss,
    CrossModalDeltaAlignmentLoss,
    LanguageModelingLoss,
)

B, D, S, V = 4, 256, 32, 32000

# L_causal
pred = torch.randn(B, D)
target = torch.randn(B, D)
loss_causal = CausalLatentPredictionLoss()
l1 = loss_causal(pred, target)
print(f"  L_causal: {l1.item():.4f} (range [0, 2])")
assert 0.0 <= l1.item() <= 2.0

# Perfect prediction should give ~0.0
perfect_l = loss_causal(pred, pred)
print(f"  L_causal (perfect): {perfect_l.item():.6f} (should be ~0.0)")
assert perfect_l.item() < 0.01

# L_cf
real_next = torch.randn(B, D)
cf_next = torch.randn(B, D)
distance_labels = torch.rand(B)
loss_cf = CounterfactualConsistencyLoss()
l2 = loss_cf(real_next, cf_next, distance_labels)
print(f"  L_cf: {l2.item():.4f} (MSE)")

# L_align
delta_text = torch.randn(B, D)
delta_image = torch.randn(B, D)
loss_align = CrossModalDeltaAlignmentLoss()
l3 = loss_align(delta_text, delta_image)
print(f"  L_align: {l3.item():.4f} (range [0, 2])")

# Same deltas should give ~0.0
aligned_l = loss_align(delta_text, delta_text)
print(f"  L_align (aligned): {aligned_l.item():.6f} (should be ~0.0)")
assert aligned_l.item() < 0.01

# L_lm
logits = torch.randn(B, S, V)
labels = torch.randint(0, V, (B, S))
loss_lm = LanguageModelingLoss(V, pad_token_id=0)
l4 = loss_lm(logits, labels)
print(f"  L_lm: {l4.item():.4f} (expected ~10.4 for random)")

print("  OK - All individual losses work")

# Test 2: Combined loss
print("\n[2/4] Testing combined loss...")
from src.training.losses import CausalWorldModelLoss

combined = CausalWorldModelLoss(vocab_size=V)

# Full batch with all loss data
model_outputs = {
    "logits": logits,
    "state_t1_pred": pred,
    "state_t1_target": target,
    "state_cf": cf_next,
    "causal_distance_labels": distance_labels,
    "delta_z_text": delta_text,
    "delta_z_image": delta_image,
}
result = combined(model_outputs, labels=labels)
print(f"  Total loss: {result['loss'].item():.4f}")
print(f"  Active losses: {int(result['losses_active'].item())}/4")
for k in ["loss_causal", "loss_cf", "loss_align", "loss_lm"]:
    if k in result:
        print(f"    {k}: {result[k].item():.4f}")
assert int(result["losses_active"].item()) == 4

# Partial batch (text-only, no CF or align data)
partial_outputs = {"logits": logits, "state_t1_pred": pred, "state_t1_target": target}
partial_result = combined(partial_outputs, labels=labels)
print(f"  Partial (LM + causal only): {partial_result['loss'].item():.4f}")
print(f"  Active losses: {int(partial_result['losses_active'].item())}/4")
assert int(partial_result["losses_active"].item()) == 2

print("  OK - Combined loss handles full and partial batches")

# Test 3: EMA scheduler
print("\n[3/4] Testing EMA scheduler...")
from src.training.ema import EMAScheduler

scheduler = EMAScheduler(momentum_start=0.99, momentum_end=0.999, anneal_steps=10000)
m0 = scheduler.get_momentum(0)
m5000 = scheduler.get_momentum(5000)
m10000 = scheduler.get_momentum(10000)
m20000 = scheduler.get_momentum(20000)
print(f"  Step     0: momentum={m0:.4f} (should be 0.99)")
print(f"  Step  5000: momentum={m5000:.4f} (between 0.99 and 0.999)")
print(f"  Step 10000: momentum={m10000:.4f} (should be 0.999)")
print(f"  Step 20000: momentum={m20000:.4f} (should stay 0.999)")
assert abs(m0 - 0.99) < 0.001
assert m5000 > 0.99 and m5000 < 0.999
assert abs(m10000 - 0.999) < 0.001
assert abs(m20000 - 0.999) < 0.001

# Test EMA update on small models
model_a = nn.Linear(4, 4, bias=False)
model_b = nn.Linear(4, 4, bias=False)
EMAScheduler.initialize_ema(model_b, model_a)
assert not any(p.requires_grad for p in model_b.parameters())

# After update, EMA should be close to original
model_a.weight.data.fill_(1.0)
scheduler.update(model_b, model_a, step=0)
print(f"  EMA weight after update: {model_b.weight.data.mean().item():.4f} (mix of old + 1.0)")

print("  OK - EMA scheduler works")

# Test 4: Trainer initialization
print("\n[4/4] Testing trainer initialization...")
from src.utils.config import WorldModelConfig
from src.model.world_model import CausalWorldModel
from src.training.trainer import CausalWorldModelTrainer, TrainingConfig

model_config = WorldModelConfig.tiny()
model = CausalWorldModel(model_config)

# Create a tiny dummy dataset
dummy_ids = torch.randint(1, 100, (32, 16))
dummy_labels = torch.randint(1, 100, (32, 16))
dummy_mask = torch.ones(32, 16, dtype=torch.long)
dummy_event = torch.randint(1, 100, (32, 8))
dummy_target = torch.randint(1, 100, (32, 16))

dataset = TensorDataset(dummy_ids, dummy_labels, dummy_mask, dummy_event, dummy_target)

# Custom collate that produces the expected dict format
def collate_fn(batch):
    ids, labels, mask, events, targets = zip(*batch)
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(mask),
        "event_ids": torch.stack(events),
        "target_ids": torch.stack(targets),
    }

train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

training_config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=2,
    num_epochs=1,
    warmup_steps=5,
    log_every_n_steps=1,
    use_fp16=False,  # CPU test
)

trainer = CausalWorldModelTrainer(
    model=model,
    model_config=model_config,
    training_config=training_config,
    train_loader=train_loader,
    device=torch.device("cpu"),
)

print(f"  Trainer created successfully")
print(f"  Effective batch size: {training_config.effective_batch_size}")
print(f"  Steps per epoch: {trainer.steps_per_epoch}")
print(f"  Total steps: {trainer.total_steps}")
print(f"  LR at step 0: {trainer.get_lr(0):.6f}")
print(f"  LR at step 5 (after warmup): {trainer.get_lr(5):.6f}")

# Run 1 step to verify the full pipeline connects
print("\n  Running 1 training step on CPU...")
batch = next(iter(train_loader))
step_losses = trainer.train_step(batch)
print(f"  Step losses: loss={step_losses['loss']:.4f}")
for k, v in step_losses.items():
    if k.startswith("loss_") and isinstance(v, float):
        print(f"    {k}={v:.4f}")

print("  OK - Full pipeline verified")

print("\n" + "=" * 60)
print("  ALL TRAINING TESTS PASSED")
print("=" * 60)
