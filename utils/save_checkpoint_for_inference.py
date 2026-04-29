#!/usr/bin/env python3
"""
Save checkpoint for inference - creates pickle file and inference utilities
"""

import os
import pickle
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_checkpoint_info(checkpoint_path: str, output_pickle_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Save checkpoint information to a pickle file for easy loading later.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        output_pickle_path: Path where to save the pickle file (default: checkpoint_path/checkpoint_info.pkl)
    
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Read adapter config
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {checkpoint_path}")
    
    with open(adapter_config_path, 'r', encoding='utf-8') as f:
        adapter_config = json.load(f)
    
    # Read trainer state to get step info
    trainer_state_path = checkpoint_path / "trainer_state.json"
    trainer_state = None
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read trainer_state.json: {e}")
    
    # Collect checkpoint information
    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path.absolute()),
        "base_model_name": adapter_config.get("base_model_name_or_path"),
        "adapter_config": adapter_config,
        "step": trainer_state.get("global_step") if trainer_state else None,
        "epoch": trainer_state.get("epoch") if trainer_state else None,
        "has_tokenizer": (checkpoint_path / "tokenizer.json").exists(),
        "has_adapter": (checkpoint_path / "adapter_model.safetensors").exists(),
    }
    
    # Save to pickle
    if output_pickle_path is None:
        output_pickle_path = checkpoint_path / "checkpoint_info.pkl"
    else:
        output_pickle_path = Path(output_pickle_path)
    
    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(checkpoint_info, f)
    
    logger.info(f"✅ Checkpoint info saved to: {output_pickle_path}")
    logger.info(f"   Step: {checkpoint_info['step']}")
    logger.info(f"   Base model: {checkpoint_info['base_model_name']}")
    logger.info(f"   Epoch: {checkpoint_info['epoch']}")
    
    return checkpoint_info


def load_checkpoint_info(pickle_path: str) -> Dict[str, Any]:
    """
    Load checkpoint information from pickle file.
    
    Args:
        pickle_path: Path to the pickle file
    
    Returns:
        Dictionary containing checkpoint information
    """
    with open(pickle_path, 'rb') as f:
        checkpoint_info = pickle.load(f)
    
    logger.info(f"✅ Checkpoint info loaded from: {pickle_path}")
    return checkpoint_info


if __name__ == "__main__":
    import sys
    
    # Default to latest checkpoint
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    checkpoint_dir = project_root / "checkpoints"
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    
    # Filter out .rar files and directories only
    checkpoints = [c for c in checkpoints if c.is_dir()]
    if not checkpoints:
        logger.error(f"No checkpoint directories found in {checkpoint_dir}")
        sys.exit(1)
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest_checkpoint = checkpoints[-1]
    
    logger.info(f"📦 Saving checkpoint info for: {latest_checkpoint.name}")
    
    try:
        checkpoint_info = save_checkpoint_info(str(latest_checkpoint))
        logger.info("\n" + "="*70)
        logger.info("✅ Successfully saved checkpoint info!")
        logger.info("="*70)
        logger.info(f"Pickle file: {latest_checkpoint / 'checkpoint_info.pkl'}")
        logger.info(f"You can now use this for inference with load_checkpoint_for_inference.py")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)

