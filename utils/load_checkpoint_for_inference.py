#!/usr/bin/env python3
"""
Load checkpoint for inference - loads model and tokenizer from checkpoint or pickle info
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: Optional[str] = None,
    pickle_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from checkpoint for inference.
    
    Args:
        checkpoint_path: Direct path to checkpoint directory (optional if pickle_path provided)
        pickle_path: Path to checkpoint_info.pkl file (optional if checkpoint_path provided)
        load_in_4bit: Whether to load base model in 4-bit quantization
        device_map: Device mapping strategy ("auto", "cpu", "cuda", etc.)
        torch_dtype: Torch dtype (default: float16 if CUDA available, else float32)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine checkpoint path
    if checkpoint_path is None:
        if pickle_path is None:
            # Try to find latest checkpoint
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            checkpoint_dir = project_root / "checkpoints"
            
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            checkpoints = [c for c in checkpoints if c.is_dir()]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            checkpoint_path = str(checkpoints[-1])
            logger.info(f"📦 Using latest checkpoint: {checkpoint_path}")
        else:
            # Load from pickle
            with open(pickle_path, 'rb') as f:
                checkpoint_info = pickle.load(f)
            checkpoint_path = checkpoint_info["checkpoint_path"]
            logger.info(f"📦 Loaded checkpoint path from pickle: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Read adapter config to get base model
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {checkpoint_path}")
    
    import json
    with open(adapter_config_path, 'r', encoding='utf-8') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    
    logger.info(f"🔧 Base model: {base_model_name}")
    logger.info(f"🔧 Loading adapter from: {checkpoint_path}")
    
    # Import transformers
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(f"Required libraries not installed: {e}. Install with: pip install transformers peft")
    
    # Load tokenizer
    logger.info("📥 Loading tokenizer...")
    if (checkpoint_path / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
        logger.info("  ✓ Tokenizer loaded from checkpoint")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        logger.info("  ✓ Tokenizer loaded from base model")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set torch dtype
    if torch_dtype is None:
        if torch.cuda.is_available() and not load_in_4bit:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    # Load base model
    logger.info("📥 Loading base model...")
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    
    if load_in_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = device_map
            logger.info("  ✓ Using 4-bit quantization")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not set up 4-bit quantization: {e}")
            logger.info("  ↳ Falling back to standard loading")
            if device_map != "auto":
                model_kwargs["device_map"] = device_map
    else:
        if device_map != "auto":
            model_kwargs["device_map"] = device_map
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    logger.info("  ✓ Base model loaded")
    
    # Load LoRA adapter
    logger.info("📥 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()
    logger.info("  ✓ Adapter loaded")
    
    # Move to appropriate device if not using device_map
    if device_map == "auto" and not load_in_4bit:
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info(f"  ✓ Model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("  ✓ Model on CPU")
    
    logger.info("="*70)
    logger.info("✅ Model and tokenizer loaded successfully!")
    logger.info("="*70)
    
    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate response from the model.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
    
    Returns:
        Generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove prompt from output if present
    if full_text.startswith(prompt):
        return full_text[len(prompt):].lstrip()
    
    return full_text


if __name__ == "__main__":
    import sys
    
    # Load model
    try:
        model, tokenizer = load_model_from_checkpoint()
        
        # Test generation
        logger.info("\n" + "="*70)
        logger.info("🧪 Testing model inference...")
        logger.info("="*70)
        
        test_prompt = "What is artificial intelligence?"
        logger.info(f"Prompt: {test_prompt}")
        
        response = generate_response(model, tokenizer, test_prompt, max_new_tokens=100)
        logger.info(f"\nResponse:\n{response}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ Inference test complete!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)

