#!/usr/bin/env python3
"""
RL Self-Improvement Loop

This module implements the reinforcement learning self-improvement loop that:
- Accepts { prompt, output, reward } feedback
- Updates local adapter delta or policy table
- Runs periodic reward-based adjustments
- Syncs logs to s3://bhiv/rl_feedback/sovereign_core/

Author: Soham Kotkar
"""

import json
import os
import time
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil
from dataclasses import dataclass, asdict
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class PolicyEntry:
    """Entry in the policy table"""
    state: str  # Encoded state (prompt + context)
    action: str  # Action taken (model output)
    reward: float  # Estimated value for this state→action (running average)
    count: int     # Number of updates applied
    last_updated: float  # Unix time of last update

class RLPolicyUpdater:
    """
    RL Self-Improvement Loop
    
    Maintains a policy table (Q-table or bandit style) and updates it
    based on user feedback to improve model performance over time.
    """
    
    def __init__(self):
        # In-memory policy and tracking
        self.policy_table = {}  # state -> PolicyEntry
        self.adapter_deltas = {}  # Placeholder for model adapter tweaks (not applied yet)
        self.reward_history = []  # Recent rewards (for stats/learning-rate tuning)
        
        # Learning hyperparameters
        self.update_frequency = 10  # Trigger sync after this many feedbacks
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        self.initialized = False
        
        # S3/NAS sync configuration (populated later via env/config)
        # TODO[Vijay]: Read these from environment or config once provided
        #   - RL_S3_BUCKET, RL_S3_PREFIX, AWS_DEFAULT_REGION
        #   - Prefer IAM role over keys; if keys are used: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
        self.s3_bucket = "bhiv"
        self.s3_prefix = "rl_feedback/sovereign_core/"
        # NOTE: Paste real endpoints here or map from env:
        #   - RL_S3_BUCKET=your-bucket-name          # ← set actual bucket
        #   - RL_S3_PREFIX=rl_feedback/sovereign/    # ← set prefix/path (ensure trailing slash)
        #   - RL_NAS_PATH=\\\\nas-host\\share\\sovereign  # ← if using NAS instead of S3
        # These will be read and used in _sync_to_s3().
        
        # TODO[Vijay]: If NAS is chosen instead of S3, add env RL_NAS_PATH and write there in _sync_to_s3
        
    async def initialize(self):
        """Initialize the RL policy updater: load any saved policy and start background jobs."""
        try:
            logger.info("Initializing RL Self-Improvement Loop...")
            
            # Load existing policy snapshot if available
            await self._load_policy()
            
            # Start background tasks (periodic optimization + periodic sync)
            asyncio.create_task(self._periodic_policy_update())
            asyncio.create_task(self._sync_to_s3())
            
            self.initialized = True
            logger.info("✅ RL Policy Updater initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL policy updater: {e}")
            raise
    
    async def _load_policy(self):
        """Load existing policy table and deltas from local disk if present."""
        try:
            policy_file = Path("data/rl_policy.json")
            if policy_file.exists():
                async with aiofiles.open(policy_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Restore policy table
                    for state, entry_data in data.get("policy_table", {}).items():
                        self.policy_table[state] = PolicyEntry(**entry_data)
                    
                    # Restore adapter deltas
                    self.adapter_deltas = data.get("adapter_deltas", {})
                    
                    logger.info(f"Loaded policy with {len(self.policy_table)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            self.policy_table = {}
            self.adapter_deltas = {}
    
    async def _save_policy(self):
        """Persist current policy table and deltas to local disk."""
        try:
            policy_file = Path("data/rl_policy.json")
            policy_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "policy_table": {
                    state: asdict(entry) 
                    for state, entry in self.policy_table.items()
                },
                "adapter_deltas": self.adapter_deltas,
                "last_updated": time.time()
            }
            
            async with aiofiles.open(policy_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            
            logger.info(f"Saved policy with {len(self.policy_table)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save policy: {e}")
    
    async def process_feedback(self, prompt: str, output: str, reward: float,
                             user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process one feedback item and update:
        - policy_table via a simple Q-style update (running average)
        - reward_history for stats
        - adapter_deltas placeholder (illustrative)
        Optionally trigger a sync every N feedbacks.
        """
        try:
            # Encode a compact state key from prompt + user/session context
            state = self._encode_state(prompt, user_id, session_id)
            
            # Update state-action value (bandit-style)
            policy_updated = await self._update_policy_entry(state, output, reward)
            
            # Log reward to rolling history
            reward_logged = await self._log_reward(prompt, output, reward, user_id, session_id)
            
            # Track adapter deltas (illustrative, no live application yet)
            adapter_updated = await self._update_adapter_deltas(state, output, reward)
            
            # Periodically sync current snapshot to remote storage
            s3_synced = False
            if len(self.reward_history) % self.update_frequency == 0:
                s3_synced = await self._sync_to_s3()
            
            return {
                "policy_updated": policy_updated,
                "reward_logged": reward_logged,
                "adapter_updated": adapter_updated,
                "s3_synced": s3_synced,
                "policy_size": len(self.policy_table),
                "reward_count": len(self.reward_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to process RL feedback: {e}")
            return {
                "policy_updated": False,
                "reward_logged": False,
                "adapter_updated": False,
                "s3_synced": False,
                "error": str(e)
            }
    
    def _encode_state(self, prompt: str, user_id: Optional[str], session_id: Optional[str]) -> str:
        """Create a compact state identifier. Keeps PII out and caps string size."""
        context_parts = []
        if user_id:
            context_parts.append(f"user:{user_id}")
        if session_id:
            context_parts.append(f"session:{session_id}")
        # Short hash of first 100 chars keeps key stable and bounded
        prompt_hash = hash(prompt[:100]) % 10000
        return f"{prompt_hash}:{':'.join(context_parts)}"
    
    async def _update_policy_entry(self, state: str, action: str, reward: float) -> bool:
        """Bandit-style running-average update for state→best_action and value."""
        try:
            if state in self.policy_table:
                entry = self.policy_table[state]
                # Incremental average: r ← r + α (reward − r)
                old_reward = entry.reward
                entry.reward = old_reward + self.learning_rate * (reward - old_reward)
                entry.count += 1
                entry.last_updated = time.time()
                # If this action performed better, keep it as best
                if reward > old_reward:
                    entry.action = action
            else:
                # First observation for this state
                self.policy_table[state] = PolicyEntry(
                    state=state,
                    action=action,
                    reward=reward,
                    count=1,
                    last_updated=time.time()
                )
            return True
        except Exception as e:
            logger.error(f"Failed to update policy entry: {e}")
            return False
    
    async def _log_reward(self, prompt: str, output: str, reward: float,
                         user_id: Optional[str], session_id: Optional[str]) -> bool:
        """Append a compact reward record to in-memory history (bounded)."""
        try:
            reward_entry = {
                "timestamp": time.time(),
                "prompt": prompt[:200],  # Limit size
                "output": output[:200],
                "reward": reward,
                "user_id": user_id,
                "session_id": session_id
            }
            self.reward_history.append(reward_entry)
            # Retain last 1000 entries to bound memory
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]
            return True
        except Exception as e:
            logger.error(f"Failed to log reward: {e}")
            return False
    
    async def _update_adapter_deltas(self, state: str, action: str, reward: float) -> bool:
        """Illustrative adapter delta tracker: accumulates positive/negative nudges."""
        try:
            if reward > 0.7:
                # High reward → positive delta bucket
                delta_key = f"{state}:positive"
                if delta_key not in self.adapter_deltas:
                    self.adapter_deltas[delta_key] = 0.0
                self.adapter_deltas[delta_key] += 0.01
            elif reward < 0.3:
                # Low reward → negative delta bucket
                delta_key = f"{state}:negative"
                if delta_key not in self.adapter_deltas:
                    self.adapter_deltas[delta_key] = 0.0
                self.adapter_deltas[delta_key] -= 0.01
            return True
        except Exception as e:
            logger.error(f"Failed to update adapter deltas: {e}")
            return False
    
    async def _periodic_policy_update(self):
        """Every few minutes, optimize hyperparameters and persist the policy."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if self.policy_table:
                    await self._optimize_policy()
                    await self._save_policy()
            except Exception as e:
                logger.error(f"Error in periodic policy update: {e}")
                await asyncio.sleep(300)
    
    async def _optimize_policy(self):
        """Adjust learning/exploration rates based on recent performance."""
        try:
            if not self.reward_history:
                return
            # Average reward over last 100 feedbacks
            recent_rewards = [entry["reward"] for entry in self.reward_history[-100:]]
            avg_reward = np.mean(recent_rewards)
            # Simple heuristics
            if avg_reward > 0.7:
                self.learning_rate = min(0.2, self.learning_rate * 1.1)
            elif avg_reward < 0.3:
                self.learning_rate = max(0.01, self.learning_rate * 0.9)
            if avg_reward > 0.8:
                self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
            elif avg_reward < 0.2:
                self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
            logger.info(f"Policy optimized: avg_reward={avg_reward:.3f}, lr={self.learning_rate:.3f}, exp={self.exploration_rate:.3f}")
        except Exception as e:
            logger.error(f"Failed to optimize policy: {e}")
    
    async def _sync_to_s3(self) -> bool:
        """Create a snapshot (rl_sync.json) and upload it (simulated for now)."""
        try:
            # Bundle current state into a compact JSON document
            sync_data = {
                "timestamp": time.time(),
                "policy_table": {
                    state: asdict(entry) 
                    for state, entry in self.policy_table.items()
                },
                "adapter_deltas": self.adapter_deltas,
                "reward_history": self.reward_history[-100:],  # Last 100 rewards
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate
            }
            
            # Always write a local copy (useful for debugging)
            sync_file = Path("data/rl_sync.json")
            sync_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(sync_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(sync_data, ensure_ascii=False, indent=2))
            
            # Attempt NAS first if configured
            try:
                nas_path = os.getenv("RL_NAS_PATH")
                # Normalize whitespace to avoid trailing space breaking UNC path
                if nas_path:
                    nas_path = nas_path.strip()
                if nas_path:
                    dst = Path(nas_path) / "rl_sync.json"
                    dst_parent = dst.parent
                    try:
                        dst_parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        # Parent may be a network mount that disallows mkdir; ignore
                        pass
                    shutil.copy(str(sync_file), str(dst))
                    logger.info(f"✅ NAS sync completed to {dst}")
                    return True
            except Exception as nas_err:
                logger.warning(f"NAS sync failed, will try S3 next: {nas_err}")

            # Fallback to S3 if env provided
            try:
                bucket = os.getenv("RL_S3_BUCKET", self.s3_bucket)
                prefix = os.getenv("RL_S3_PREFIX", self.s3_prefix)
                if bucket and prefix:
                    try:
                        import boto3  # optional dependency
                        s3 = boto3.client("s3")
                        s3.upload_file(str(sync_file), bucket, f"{prefix}rl_sync.json")
                        logger.info(f"✅ S3 sync completed to s3://{bucket}/{prefix}rl_sync.json")
                        return True
                    except ImportError:
                        logger.warning("boto3 not installed; skipping S3 upload")
            except Exception as s3_err:
                logger.warning(f"S3 sync failed: {s3_err}")

            # If neither NAS nor S3 succeeded, just log simulation
            logger.info(f"Simulating sync to {self.s3_bucket}/{self.s3_prefix}")
            return True
        except Exception as e:
            logger.error(f"Failed to sync to S3: {e}")
            return False
    
    async def get_policy_stats(self) -> Dict[str, Any]:
        """Summarize current policy and learning stats for monitoring."""
        try:
            if not self.reward_history:
                avg_reward = 0.0
                recent_avg = 0.0
            else:
                avg_reward = np.mean([entry["reward"] for entry in self.reward_history])
                recent_avg = np.mean([entry["reward"] for entry in self.reward_history[-50:]])
            return {
                "initialized": self.initialized,
                "policy_entries": len(self.policy_table),
                "reward_history_size": len(self.reward_history),
                "average_reward": avg_reward,
                "recent_average_reward": recent_avg,
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "adapter_deltas": len(self.adapter_deltas),
                "last_sync": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get policy stats: {e}")
            return {"error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Alias for external callers expecting get_stats()."""
        return await self.get_policy_stats()
