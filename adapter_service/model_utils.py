#!/usr/bin/env python3
"""
Model Utilities for Adapter Loading and Management

Provides helper functions for:
- Loading base models with adapters
- Merging LoRA adapters
- Model quantization (8-bit/FP16)
- Device management (CPU/GPU)

Note: Core functionality is currently integrated in standalone_api.py
This module provides a clean interface for future refactoring.
"""

import logging
from typing import Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def load_base_model(
    model_name: str = "bigscience/bloomz-560m",
    use_8bit: bool = True,
    device_map: str = "auto"
) -> Tuple[Any, Any]:
    """
    Load base language model with optional quantization
    
    Args:
        model_name: HuggingFace model identifier
        use_8bit: Whether to use 8-bit quantization
        device_map: Device mapping strategy ("auto", "cpu", "cuda")
        
    Returns:
        Tuple of (model, tokenizer)
        
    Example:
        >>> model, tokenizer = load_base_model("bigscience/bloomz-560m")
        >>> outputs = model.generate(...)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        logger.info(f"Loading base model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with optional quantization
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch.float16 if use_8bit else torch.float32,
        }
        
        if use_8bit and torch.cuda.is_available():
            load_kwargs["load_in_8bit"] = True
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        logger.info(f"Model loaded successfully on {device_map}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_adapter(
    model: Any,
    adapter_path: str,
    adapter_name: str = "default"
) -> Any:
    """
    Load and apply LoRA adapter to base model
    
    Args:
        model: Base model instance
        adapter_path: Path to adapter checkpoint
        adapter_name: Name for the adapter
        
    Returns:
        Model with adapter loaded
        
    Example:
        >>> model, tokenizer = load_base_model()
        >>> model = load_adapter(model, "adapters/gurukul_lite")
    """
    try:
        from peft import PeftModel
        
        logger.info(f"Loading adapter from: {adapter_path}")
        
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
            
        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            adapter_name=adapter_name
        )
        
        logger.info(f"Adapter '{adapter_name}' loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise


def merge_adapter(
    model: Any,
    save_path: Optional[str] = None
) -> Any:
    """
    Merge adapter weights into base model
    
    Args:
        model: Model with adapter loaded
        save_path: Optional path to save merged model
        
    Returns:
        Merged model
        
    Example:
        >>> model = load_adapter(model, "adapters/gurukul_lite")
        >>> merged = merge_adapter(model, "models/merged_bloomz")
    """
    try:
        logger.info("Merging adapter into base model...")
        
        # Merge adapter weights
        model = model.merge_and_unload()
        
        if save_path:
            logger.info(f"Saving merged model to: {save_path}")
            model.save_pretrained(save_path)
            
        logger.info("Adapter merged successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to merge adapter: {e}")
        raise


def get_model_info(model: Any) -> dict:
    """
    Get information about loaded model
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with model information
    """
    try:
        import torch
        
        info = {
            "dtype": str(model.dtype),
            "device": str(model.device),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "is_quantized": hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit,
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {}


def unload_model(model: Any) -> None:
    """
    Unload model from memory
    
    Args:
        model: Model instance to unload
    """
    try:
        import torch
        import gc
        
        logger.info("Unloading model from memory...")
        
        # Delete model
        del model
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Force garbage collection
        gc.collect()
        
        logger.info("Model unloaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")


# TODO: Implement these functions when adapter training is working
def prepare_model_for_training(model: Any, use_gradient_checkpointing: bool = True) -> Any:
    """Prepare model for LoRA training (TODO: implement when training works)"""
    raise NotImplementedError("Adapter training is currently not working. See standalone_api.py for generation.")


def get_peft_config(
    task_type: str = "CAUSAL_LM",
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list = None
) -> Any:
    """Get PEFT configuration for LoRA (TODO: implement when training works)"""
    raise NotImplementedError("Adapter training is currently not working. See standalone_api.py for generation.")


if __name__ == "__main__":
    # Test model loading
    logging.basicConfig(level=logging.INFO)
    
    print("Testing model_utils.py...")
    print("\n1. Loading base model...")
    
    try:
        model, tokenizer = load_base_model("bigscience/bloomz-560m", use_8bit=False)
        print(f"✅ Model loaded successfully")
        
        info = get_model_info(model)
        print(f"\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        print("\n2. Unloading model...")
        unload_model(model)
        print("✅ Model unloaded successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")

