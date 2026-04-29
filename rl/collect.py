#!/usr/bin/env python3
"""
RL episode collection scaffold

Purpose:
- Provide a lightweight, MCP-friendly hook to log interaction episodes
- Write JSONL episodes locally (NAS path) or to a cloud URL (pre-signed PUT)

Usage examples:
- Local JSONL logging:
  python rl/collect.py --out rl_runs/episodes.jsonl --env_name toy --max_episodes 3 \
    --prompt "Translate to Hindi: Hello friend." --adapter_path adapters/gurukul_lite

- Cloud (pre-signed URL) upload after run:
  python rl/collect.py --out rl_runs/episodes.jsonl --upload_url "https://...presigned..." --upload true

Notes:
- This is a scaffold: integrate your real reward function and environment later
"""

import os
import sys
import json
import time
import uuid
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_generator(base_model: str, adapter_path: Optional[str] = None):
    """Load a lightweight generator (base model + optional adapter)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )
    
    # Load adapter if provided
    if adapter_path and Path(adapter_path).exists():
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("Loaded adapter from: %s", adapter_path)
        except Exception as e:
            logger.warning("Failed to load adapter %s: %s", adapter_path, e)
            logger.info("Using base model without adapter")
    else:
        logger.info("Using base model without adapter")

    device = next(model.parameters()).device

    def generate(prompt: str, max_new_tokens: int = 64, temperature: float = 0.7) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if full_text.startswith(prompt):
            return full_text[len(prompt):].lstrip()
        return full_text

    return generate


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def upload_file(url: str, file_path: Path) -> bool:
    """Upload file to cloud storage (S3, NAS, etc.)"""
    try:
        with file_path.open("rb") as f:
            resp = requests.put(url, data=f, timeout=30)
        if resp.status_code // 100 == 2:
            logger.info("Upload successful to: %s", url)
            return True
        logger.warning("Upload failed: %s %s", resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        logger.error("Upload error: %s", e)
        return False


def upload_to_s3(file_path: Path, bucket: str, key: str, aws_access_key: str, aws_secret_key: str) -> bool:
    """Upload to S3 using boto3 (if available)"""
    try:
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        s3_client.upload_file(str(file_path), bucket, key)
        logger.info("Uploaded to S3: s3://%s/%s", bucket, key)
        return True
    except ImportError:
        logger.warning("boto3 not available, falling back to HTTP upload")
        return False
    except Exception as e:
        logger.error("S3 upload error: %s", e)
        return False


def calculate_reward(prompt: str, output: str, max_new_tokens: int) -> float:
    """Calculate reward for the generated output"""
    # Basic reward function - can be enhanced with more sophisticated metrics
    
    # Length reward (0-0.4)
    length_ratio = len(output) / max(1, max_new_tokens)
    length_reward = min(0.4, length_ratio * 0.4)
    
    # Quality reward (0-0.3) - basic heuristics
    quality_reward = 0.0
    if len(output.strip()) > 5:  # Not too short
        quality_reward += 0.1
    if not output.strip() == prompt.strip():  # Not just echoing
        quality_reward += 0.1
    if len(output.split()) > 2:  # Multiple words
        quality_reward += 0.1
    
    # Language diversity reward (0-0.3) - check for non-English characters
    diversity_reward = 0.0
    if any(ord(char) > 127 for char in output):  # Non-ASCII characters
        diversity_reward += 0.3
    
    total_reward = length_reward + quality_reward + diversity_reward
    return max(0.0, min(1.0, total_reward))


def collect(args: argparse.Namespace) -> Path:
    run_id = args.run_id or str(uuid.uuid4())
    out_path = Path(args.out)
    logger.info("Starting RL collection run_id=%s -> %s", run_id, out_path)

    # Load generator lazily
    generate = load_generator(args.base_model, args.adapter_path)

    # Sample prompts for different languages
    sample_prompts = [
        "Translate to Hindi: Hello friend.",
        "Write a short story in Bengali about a cat.",
        "Explain machine learning in Tamil.",
        "Tell me about the weather in Telugu.",
        "Write a poem in Gujarati about nature.",
        "Describe your day in Marathi.",
        "Translate to Urdu: Good morning everyone.",
        "Write a greeting in Punjabi.",
        "Explain technology in Nepali.",
        "Tell a joke in Odia."
    ]

    for ep_idx in range(args.max_episodes):
        t0 = time.time()
        
        # Use provided prompt or sample from multilingual prompts
        if args.prompt:
            prompt = args.prompt
        else:
            prompt = sample_prompts[ep_idx % len(sample_prompts)]
        
        output = generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

        # Calculate reward using enhanced function
        reward = calculate_reward(prompt, output, args.max_new_tokens)

        episode = {
            "run_id": run_id,
            "episode_index": ep_idx,
            "timestamp": time.time(),
            "env_name": args.env_name,
            "prompt": prompt,
            "output": output,
            "reward": reward,
            "latency_s": round(time.time() - t0, 3),
            "meta": {
                "adapter_path": args.adapter_path,
                "base_model": args.base_model,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "language": "multilingual",
                "model_type": "bloomz-560m"
            },
        }
        write_jsonl(out_path, episode)
        logger.info("Episode %d reward=%.3f len=%d output='%s'", 
                   ep_idx, reward, len(output), output[:50] + "..." if len(output) > 50 else output)

    # Upload to cloud if configured
    if args.upload and args.upload_url:
        logger.info("Uploading episodes to cloud...")
        upload_file(args.upload_url, out_path)
    
    # Upload to S3 if configured
    if args.s3_bucket and args.s3_key:
        logger.info("Uploading episodes to S3...")
        upload_to_s3(out_path, args.s3_bucket, args.s3_key, args.aws_access_key, args.aws_secret_key)

    logger.info("Collection complete: %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL episode collection scaffold")
    p.add_argument("--out", default="rl_runs/episodes.jsonl", help="Output JSONL path")
    p.add_argument("--env_name", default="mcp-lite", help="Environment name/label")
    p.add_argument("--run_id", default="", help="Optional run id (uuid if empty)")
    p.add_argument("--max_episodes", type=int, default=3, help="Number of episodes to collect")
    p.add_argument("--prompt", default="", help="Prompt to use for each episode")
    p.add_argument("--adapter_path", default="", help="Adapter directory path (optional)")
    p.add_argument("--base_model", default="bigscience/bloomz-560m", help="Base model identifier")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens for generation")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    # Cloud upload options
    p.add_argument("--upload", action="store_true", help="Upload output file to a pre-signed URL")
    p.add_argument("--upload_url", default="", help="Pre-signed PUT URL for upload")
    
    # S3 upload options
    p.add_argument("--s3_bucket", default="", help="S3 bucket name for upload")
    p.add_argument("--s3_key", default="", help="S3 key/path for upload")
    p.add_argument("--aws_access_key", default="", help="AWS access key")
    p.add_argument("--aws_secret_key", default="", help="AWS secret key")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(args)


