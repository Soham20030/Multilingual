# Checkpoint Inference Utilities

This directory contains utilities for saving and loading checkpoints for inference.

## Files

- `save_checkpoint_for_inference.py` - Saves checkpoint metadata to a pickle file
- `load_checkpoint_for_inference.py` - Loads model and tokenizer from checkpoint for inference

## Usage

### 1. Save Checkpoint Info (Already Done)

The pickle file has been created at:
```
checkpoints/checkpoint-125000/checkpoint_info.pkl
```

This file contains:
- Checkpoint path
- Base model name
- Adapter configuration
- Training step and epoch information

### 2. Load Model for Inference

#### Option A: Using the utility script

```python
from utils.load_checkpoint_for_inference import load_model_from_checkpoint, generate_response

# Load model and tokenizer (automatically uses latest checkpoint)
model, tokenizer = load_model_from_checkpoint()

# Or specify checkpoint path
model, tokenizer = load_model_from_checkpoint(checkpoint_path="checkpoints/checkpoint-125000")

# Or use pickle file
model, tokenizer = load_model_from_checkpoint(pickle_path="checkpoints/checkpoint-125000/checkpoint_info.pkl")

# Generate response
response = generate_response(
    model, 
    tokenizer, 
    "What is artificial intelligence?",
    max_new_tokens=256
)
print(response)
```

#### Option B: Using pickle file directly

```python
import pickle
from utils.load_checkpoint_for_inference import load_model_from_checkpoint

# Load checkpoint info from pickle
with open("checkpoints/checkpoint-125000/checkpoint_info.pkl", "rb") as f:
    checkpoint_info = pickle.load(f)

print(f"Checkpoint step: {checkpoint_info['step']}")
print(f"Base model: {checkpoint_info['base_model_name']}")

# Load model using the checkpoint path from pickle
model, tokenizer = load_model_from_checkpoint(
    checkpoint_path=checkpoint_info['checkpoint_path']
)
```

#### Option C: Direct loading (standard approach)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load adapter from checkpoint
checkpoint_path = "checkpoints/checkpoint-125000"
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()
```

## Command Line Usage

### Save checkpoint info:
```bash
python utils/save_checkpoint_for_inference.py
```

### Test loading and inference:
```bash
python utils/load_checkpoint_for_inference.py
```

## Notes

- The pickle file only contains metadata, not the actual model weights
- Model weights are stored in the checkpoint directory (`adapter_model.safetensors`)
- The pickle file makes it easy to track which checkpoint to use without hardcoding paths
- For production, you can use the checkpoint directory directly (no pickle needed)

## Checkpoint Information

**Latest Checkpoint:** `checkpoint-125000`
- **Step:** 125000
- **Epoch:** 5.0
- **Base Model:** `unsloth/llama-3.2-3b-instruct-bnb-4bit`
- **Location:** `checkpoints/checkpoint-125000/`

