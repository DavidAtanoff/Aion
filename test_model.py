"""Quick verification script for the Causal-JEPA model architecture."""
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("  Causal-JEPA World Model — Architecture Verification")
print("=" * 60)

# Test 1: Config presets
print("\n[1/4] Testing config presets...")
from src.utils.config import WorldModelConfig

for name in ["tiny", "base", "large", "xl", "research"]:
    config = getattr(WorldModelConfig, name)()
    est = config.estimate_params()
    trainable = est["total_trainable"]
    print(f"  {name:>10}: {trainable/1e6:>8.1f}M trainable params")
print("  ✓ Config OK")

# Test 2: Encoder forward pass
print("\n[2/4] Testing encoder forward pass...")
import torch
from src.model.encoder import UnifiedEncoder

config = WorldModelConfig.tiny()
encoder = UnifiedEncoder(config)

# Text-only
B, S = 2, 32
input_ids = torch.randint(0, config.vocab_size, (B, S))
out = encoder(input_ids=input_ids, use_causal_mask=True)
print(f"  Text-only:  input={input_ids.shape} → output={out.shape}")
assert out.shape == (B, S, config.hidden_dim), f"Expected {(B, S, config.hidden_dim)}, got {out.shape}"

# Image-only
images = torch.randn(B, 3, config.image_size, config.image_size)
out = encoder(images=images, use_causal_mask=False)
print(f"  Image-only: input={images.shape} → output={out.shape}")
assert out.shape == (B, config.num_patches, config.hidden_dim)

# Mixed
out = encoder(input_ids=input_ids, images=images, use_causal_mask=True)
expected_len = S + config.num_patches
print(f"  Mixed:      input=({input_ids.shape}, {images.shape}) → output={out.shape}")
assert out.shape == (B, expected_len, config.hidden_dim)

# With memory vectors
memory = torch.randn(B, 8, config.hidden_dim)
out = encoder(input_ids=input_ids, memory_vectors=memory, use_causal_mask=True)
print(f"  With memory: input={input_ids.shape}, mem={memory.shape} → output={out.shape}")
assert out.shape == (B, S, config.hidden_dim)

print("  ✓ Encoder OK")

# Test 3: Causal network
print("\n[3/4] Testing causal network...")
from src.model.causal_net import CausalStateCompressor, CausalTransitionNetwork

compressor = CausalStateCompressor(config)
transition = CausalTransitionNetwork(config)

# Compress encoder output to state
encoder_out = torch.randn(B, S, config.hidden_dim)
state = compressor(encoder_out)
print(f"  Compressor: input={encoder_out.shape} → state={state.shape}")
assert state.shape == (B, config.hidden_dim)

# Predict next state
event = torch.randn(B, config.hidden_dim)
next_state = transition(state, event)
print(f"  Transition: state={state.shape} + event={event.shape} → next={next_state.shape}")
assert next_state.shape == (B, config.hidden_dim)

# Counterfactual prediction
cf_event = torch.randn(B, config.hidden_dim)
cf_result = transition.predict_counterfactual(state, event, cf_event)
print(f"  Counterfactual: causal_distance={cf_result['causal_distance'].shape}")
assert cf_result["causal_distance"].shape == (B,)

print("  ✓ Causal network OK")

# Test 4: Full world model
print("\n[4/4] Testing full world model...")
from src.model.world_model import CausalWorldModel

model = CausalWorldModel(config)
trainable = model.count_parameters()
total = model.count_parameters(include_ema=True)
print(f"  Trainable params:     {trainable:>12,}")
print(f"  Total (incl. EMA):    {total:>12,}")

# Training forward pass
outputs = model(
    input_ids=input_ids,
    event_ids=torch.randint(0, config.vocab_size, (B, 16)),
    target_ids=torch.randint(0, config.vocab_size, (B, S)),
)
print(f"  Forward outputs: {list(outputs.keys())}")
assert "logits" in outputs
assert "state_t" in outputs
assert "state_t1_pred" in outputs
assert "state_t1_target" in outputs
print(f"  Logits shape:      {outputs['logits'].shape}")
print(f"  State shape:       {outputs['state_t'].shape}")
print(f"  Pred state shape:  {outputs['state_t1_pred'].shape}")
print(f"  Target state shape:{outputs['state_t1_target'].shape}")

# EMA update
momentum = model.update_ema(step=0)
print(f"  EMA momentum (step 0): {momentum:.4f}")
momentum = model.update_ema(step=10000)
print(f"  EMA momentum (step 10000): {momentum:.4f}")

# Memory gate check
for layer in model.encoder.layers:
    if layer.has_memory_cross_attn:
        gate_val = layer.memory_cross_attn.gate.item()
        print(f"  Memory gate init value: {gate_val:.4f} (should be 0.0)")
        assert gate_val == 0.0, f"Gate should be 0.0, got {gate_val}"

print("  ✓ Full world model OK")

# Print model summary
print()
print(model.summary())

print("\n✅ ALL TESTS PASSED — Architecture verified!")
