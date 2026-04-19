"""
Causal-JEPA Comprehensive Testing Suite
=======================================

This script evaluates the trained Causal-JEPA World Model across four dimensions:
1. Fact Retrieval (Zero-shot knowledge)
2. Logical/Coding Reasoning (OpenHermes task proxy)
3. Conversational Fluency
4. Infinite Memory / Dynamic State Injection (The major architectural novelty)

Usage on Kaggle:
Paste this into a new cell AFTER your training cell has completed, or 
load an existing checkpoint.
"""

import os
import torch
from transformers import AutoTokenizer

try:
    # If pasted below the training cell, these are already in memory
    from __main__ import WorldModelConfig, CausalWorldModel
except ImportError:
    # Fallback if running as standalone script
    from causal_jepa import WorldModelConfig, CausalWorldModel

print("="*60)
print("  Evaluating Causal-JEPA World Model")
print("="*60)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Re-initialize the model to Base scale (must match what we trained!)
print("1. Initializing model architecture...")
config = WorldModelConfig.base()
config.vocab_size = tokenizer.vocab_size
config.pad_token_id = tokenizer.pad_token_id
test_model = CausalWorldModel(config)

# Load the weights we just trained
ckpt_path = "/kaggle/working/checkpoints/final.pt" 
print(f"2. Loading trained weights from {ckpt_path}...")

# Use weights_only=False to bypass PyTorch 2.6 security blocks on custom classes
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    test_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load memory database if it exists
    memory_dir = os.path.join(os.path.dirname(ckpt_path), "memory")
    if os.path.exists(memory_dir):
        test_model.memory.load(memory_dir)
        print(f"   Loaded Episodic Memory: {test_model.memory.episodic_store.size} items")
else:
    print(f"WARNING: Checkpoint not found at {ckpt_path}. Using random weights!")

test_model = test_model.to(device)
test_model.eval()

# =======================================================================
# Test Suite Defines
# =======================================================================

tests = [
    {
        "name": "Fact Retrieval",
        "prompt": "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "max_tokens": 20
    },
    {
        "name": "Coding/Logic (OpenHermes Style)",
        "prompt": "### Instruction:\nWrite a python function to add two numbers together, and explain how it works.\n\n### Response:\n",
        "max_tokens": 80
    },
    {
        "name": "Conversational Constraints",
        "prompt": "### Instruction:\nSpeak to me purely in pirate language. Tell me what your favorite food is.\n\n### Response:\n",
        "max_tokens": 50
    }
]

print("\n3. Running Standard Generation Tests...\n")

for t in tests:
    print(f"--- TEST: {t['name']} ---")
    print(f"PROMPT:\n{t['prompt'].strip()}")
    
    input_ids = tokenizer(t['prompt'], return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        out_ids = test_model.generate(
            input_ids=input_ids,
            max_new_tokens=t['max_tokens'],
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            use_memory=True,          # Active memory retrieval
            store_as_episode=False,   # Don't pollute memory during basic testing
        )
    
    new_tokens = out_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print(f"\n[GENERATED RESPONSE]\n{response.strip()}")
    print("=" * 60 + "\n")


# =======================================================================
# The "Infinite Memory" Test
# =======================================================================
print("\n4. Running Infinite Memory / State Injection Test...")
print("Concept: We will inject a completely novel, made-up causal state into the model.")
print("We will then clear its 'context window' and ask it about that state to prove")
print("that it can retrieve facts dynamically via its Causal Compressor & FAISS database.")

base_memory_count = test_model.memory.episodic_store.size

# Step 4a: Memorize novel fact
fact_prompt = "### Instruction:\nThe CEO of Centis Systems is a golden retriever named Max.\n\n### Response:\nUnderstood. Max the golden retriever is the CEO."
fact_ids = tokenizer(fact_prompt, return_tensors="pt").input_ids.to(device)

print(f"\n[PHASE 1] Prompting model with novel fact & storing recursively...")
with torch.no_grad():
    # store_as_episode=True actively writes this to the FAISS database
    # Passing max_new_tokens=2 just to trigger the generation loop which handles the storage
    test_model.generate(fact_ids, max_new_tokens=2, store_as_episode=True)

new_memory_count = test_model.memory.episodic_store.size
print(f"Episodic memory count increased: {base_memory_count} -> {new_memory_count}")

# Step 4b: Retrieve from empty context
question = "### Instruction:\nWho is the CEO of Centis Systems?\n\n### Response:\n"
q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

print("\n[PHASE 2] Asking the model purely from a completely empty context window...")
with torch.no_grad():
    # use_memory=True tells the Causal Compressors to scan FAISS 
    # and pull the "golden retriever" state directly into Layer 3
    out_ids = test_model.generate(
        q_ids, 
        max_new_tokens=40, 
        use_memory=True,         # CRITICAL TO RESOLVE THE NOVEL STATE
        store_as_episode=False
    )

new_tokens = out_ids[0, q_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\n[MODEL MEMORY RESPONSE]\n{response.strip()}")
print("\n(If it mentioned Max or a Golden Retriever, the Memory subsystem is 100% functional!)")
print("=" * 60)
