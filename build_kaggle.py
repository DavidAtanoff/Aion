"""
Build the all-in-one Kaggle notebook cell.
Combines causal_jepa.py + kaggle_train.py into a single executable file.
"""
import re

# Read compiled model
with open("causal_jepa.py", "r", encoding="utf-8") as f:
    model_code = f.read()

# Read training script
with open("kaggle_train.py", "r", encoding="utf-8") as f:
    train_code = f.read()

# Strip the causal_jepa imports from training script (already in same file)
train_code = re.sub(
    r'^from causal_jepa import \([\s\S]*?\)\s*$',
    '# [model code is above — no import needed]',
    train_code,
    flags=re.MULTILINE,
)

# Strip the module docstring from training script (we'll add our own header)
train_code = re.sub(
    r'^"""[\s\S]*?"""',
    '',
    train_code,
    count=1,
)

# Strip `if __name__ == "__main__":` block — replace with direct execution
# Find the if __name__ block and extract what's inside
main_match = re.search(r'if __name__ == "__main__":\s*\n([\s\S]*?)$', train_code)
if main_match:
    # Remove the if __name__ block
    train_code = train_code[:main_match.start()]
    
# Build the combined file
combined = f'''#!/usr/bin/env python3
# ==========================================================================
#  CAUSAL-JEPA WORLD MODEL — ALL-IN-ONE KAGGLE CELL
# ==========================================================================
#  Paste this ENTIRE file into a single Kaggle notebook cell and run it.
#  Requirements: GPU T4 x2, Internet ON
#
#  What this does:
#    1. Defines the complete Causal-JEPA architecture (~4800 lines)
#    2. Downloads yahma/alpaca-cleaned from HuggingFace (52K examples)
#    3. Trains base config (42M params) for 1 epochs on 2x T4 GPUs
#    4. Saves checkpoints to /kaggle/working/checkpoints/
#
#  Estimated training time: ~1.5 hours on 2x T4
# ==========================================================================

{model_code}

# ==========================================================================
#  TRAINING SCRIPT
# ==========================================================================

{train_code}

# ==========================================================================
#  AUTO-RUN TRAINING
# ==========================================================================

# You can change these settings:
SETTINGS = dict(
    model_scale="base",        # tiny=13M, base=42M, large=111M, xl=323M
    dataset_name="teknium/OpenHermes-2.5",  # auto-downloads from HuggingFace
    num_epochs=2,
    batch_size=16,             # per GPU (16 x 2 GPUs x 2 accum = 64 effective)
    learning_rate=3e-4,
    max_seq_length=256,
    save_dir="/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "./checkpoints",
)

print("=" * 60)
print("  Starting Causal-JEPA training with settings:")
for k, v in SETTINGS.items():
    print(f"    {{k}}: {{v}}")
print("=" * 60)

config = KaggleTrainingConfig(**SETTINGS)
train(config, use_ddp=False)
'''

# Write output
output_path = "kaggle_all_in_one.py"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(combined)

lines = combined.count("\\n") + 1
size_kb = len(combined.encode()) / 1024
print(f"Created {output_path}: {lines:,} lines, {size_kb:.1f} KB")
