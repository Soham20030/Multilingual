#!/usr/bin/env python3
"""
Unsloth Fine-Tuning Script for WSL
Trains Llama-3.2-3B-Instruct on Arabic instruction dataset (all 200k samples)
"""

import os

# Disable Unsloth zoo patches that rely on torch._inductor.config (not available in your Torch build)
os.environ["UNSLOTH_DISABLE_ZOO_PATCHES"] = "1"

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import json
from collections import Counter, defaultdict
import random
import glob
import logging
from datetime import datetime

# ============================================================================
# Setup Logging
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create log file with timestamp
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(logs_dir, log_filename)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("Unsloth Fine-Tuning - WSL")
logger.info("=" * 70)
logger.info(f"Logging to: {log_filepath}")

# ============================================================================
# Step 1: Load Dataset
# ============================================================================
logger.info("\n[Step 1] Loading Arabic dataset...")

# Dataset path (relative to project root)
# script_dir and project_root already defined above
arabic_path = os.path.join(project_root, "data", "Arabic.jsonl")

# Check if dataset exists
if not os.path.exists(arabic_path):
    raise FileNotFoundError(f"Arabic dataset not found: {arabic_path}")

logger.info(f"  Arabic dataset: {arabic_path}")

# Load dataset
dataset = load_dataset("json", data_files=arabic_path, split="train")

logger.info(f"  ✓ Loaded {len(dataset):,} Arabic samples")
logger.info(f"  ✓ Using all {len(dataset):,} samples for training (no sampling)")

# Show first example
if len(dataset) > 0:
    logger.info("\n  First example:")
    logger.info(json.dumps(dataset[0], ensure_ascii=False, indent=2)[:500])

# ============================================================================
# Step 3: Load Model
# ============================================================================
logger.info("\n[Step 3] Loading model...")

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True

logger.info("  Loading Llama-3.2-3B-Instruct model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
logger.info("  ✓ Model loaded successfully")

# ============================================================================
# Step 4: Format Dataset
# ============================================================================
logger.info("\n[Step 4] Formatting dataset...")

# Arabic prompt template
arabic_prompt = """You are a helpful Arabic language assistant capable of understanding and responding in Modern Standard Arabic and various Arabic dialects.

Please respond to the following in Arabic:

{}
"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    if 'instruction' in examples and 'output' in examples:
        for instruction, output in zip(examples['instruction'], examples['output']):
            text = arabic_prompt.format(f"{instruction}\n\n{output}") + EOS_TOKEN
            texts.append(text)
    elif 'text' in examples:
        for text in examples['text']:
            texts.append(text + EOS_TOKEN)
    else:
        raise ValueError(f"Unknown dataset format. Keys: {list(examples.keys())}")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
logger.info("  ✓ Dataset formatted successfully")

# ============================================================================
# Step 4.5: Setup Checkpoint Directory
# ============================================================================
logger.info("\n[Step 4.5] Setting up checkpoint directory...")

checkpoint_dir = os.path.join(project_root, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
logger.info(f"  Checkpoint directory: {checkpoint_dir}")

# Find latest checkpoint if it exists
def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the checkpoint directory."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number (extract from checkpoint-XXXXX)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    latest = checkpoints[-1]
    logger.info(f"  Found latest checkpoint: {latest}")
    return latest

latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    logger.info(f"  ✓ Will resume from: {latest_checkpoint}")
else:
    logger.info(f"  ✓ No existing checkpoint found. Starting fresh training.")

# ============================================================================
# Step 5: Configure LoRA
# ============================================================================
logger.info("\n[Step 5] Configuring LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
logger.info("  ✓ LoRA configuration applied")

# ============================================================================
# Step 6: Setup Trainer
# ============================================================================
logger.info("\n[Step 6] Setting up trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=125000,  # 5 epochs for ~200k samples (25000 * 5)
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        logging_dir=logs_dir,  # Save training logs
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=checkpoint_dir,  # Save checkpoints here
        save_strategy="steps",
        save_steps=2500,  # Save checkpoint every 2500 steps
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        report_to="none",
    ),
)
logger.info("  ✓ Trainer configured and ready")

# ============================================================================
# Step 7: Train Model
# ============================================================================
logger.info("\n" + "=" * 70)
logger.info("[Step 7] Starting training...")
logger.info("=" * 70)
logger.info("This may take 18-24 hours depending on your GPU (RTX 3060) for 3 epochs.")
logger.info(f"Checkpoints will be saved every 2500 steps to: {checkpoint_dir}")
logger.info(f"Training logs will be saved to: {log_filepath}")
if latest_checkpoint:
    logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
logger.info("=" * 70)

trainer.train(resume_from_checkpoint=latest_checkpoint if latest_checkpoint else None)

logger.info("=" * 70)
logger.info("  ✓ Training complete!")
logger.info("=" * 70)

# ============================================================================
# Step 8: Save Model
# ============================================================================
logger.info("\n[Step 8] Saving model...")

output_dir = os.path.join(project_root, "lora_model")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"  ✓ Model saved to: {output_dir}")

logger.info("\n" + "=" * 70)
logger.info("✅ All done! Model saved successfully!")
logger.info(f"Training log saved to: {log_filepath}")
logger.info("=" * 70)

