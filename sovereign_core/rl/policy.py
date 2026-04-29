import json
import os
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

# Local data files used by the policy
DATA_DIR = Path("data")
FEEDBACK_PATH = DATA_DIR / "feedback_stream.jsonl"  # Appended by MCP stream
POLICY_PATH = DATA_DIR / "rl_policy.json"          # Saved by RLPolicyUpdater


class PolicyManager:
	"""Lightweight policy manager exposing alignment actions for KSML."""
	def __init__(self, feedback_path: Path = FEEDBACK_PATH, policy_path: Path = POLICY_PATH) -> None:
		# File locations for recent feedback and saved policy snapshot
		self.feedback_path = feedback_path
		self.policy_path = policy_path
		# Simple cache to avoid re-reading files too frequently
		self._last_load_ts: float = 0.0
		self._cache: Dict[str, Any] = {}

	def _load_recent_feedback(self, max_lines: int = 200) -> List[Dict[str, Any]]:
		"""Read the tail of feedback_stream.jsonl (last max_lines entries)."""
		entries: List[Dict[str, Any]] = []
		if not self.feedback_path.exists():
			return entries
		# Efficient tail-read
		try:
			with open(self.feedback_path, "r", encoding="utf-8") as f:
				lines = f.readlines()[-max_lines:]
			for line in lines:
				line = line.strip()
				if not line:
					continue
				try:
					entries.append(json.loads(line))
				except Exception:
					continue
		except Exception:
			pass
		return entries

	def _load_policy(self) -> Dict[str, Any]:
		"""Load the saved policy snapshot (optional for future advanced nudges)."""
		if not self.policy_path.exists():
			return {}
		try:
			with open(self.policy_path, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception:
			return {}

	def get_alignment_actions(self, input_text: str, source_lang: Optional[str]) -> Dict[str, Any]:
		"""
		Compute simple nudges for KSML based on recent rewards and saved policy.
		Returns keys used by KSML aligner to adjust outputs:
		- target_lang_override (Optional[str])
		- karma_bias (Optional[str]) in {"sattva","rajas","tamas"}
		- intent_bias (Optional[str])
		- confidence_delta (float)
		- tone (Optional[str])
		"""
		# Refresh cache every 5 seconds
		now = time.time()
		if now - self._last_load_ts > 5:
			recent = self._load_recent_feedback()
			policy = self._load_policy()
			self._cache = {"recent": recent, "policy": policy}
			self._last_load_ts = now
		recent: List[Dict[str, Any]] = self._cache.get("recent", [])
		# Average reward over recent feedbacks (very simple signal)
		avg_reward = 0.0
		if recent:
			valid = [e for e in recent if isinstance(e.get("reward", None), (int, float))]
			if valid:
				avg_reward = sum(e.get("reward", 0.0) for e in valid) / max(1, len(valid))
		# Heuristics mapping average reward to small KSML nudges
		confidence_delta = 0.05 if avg_reward >= 0.6 else (-0.05 if avg_reward and avg_reward < 0.3 else 0.0)
		target_lang_override: Optional[str] = None
		karma_bias: Optional[str] = None
		intent_bias: Optional[str] = None
		tone: Optional[str] = None
		# If many recent prompts are translation-heavy, bias target language to English
		if recent:
			lower_prompts = " \n".join(str(e.get("prompt", "")).lower() for e in recent)
			if any(kw in lower_prompts for kw in ["translate", "in english", "english"]):
				target_lang_override = "en"
		# Reward shapes basic tone/karma bias
		if avg_reward > 0.75:
			karma_bias = "sattva"
			tone = "calm"
		elif 0.4 <= avg_reward <= 0.75:
			karma_bias = "rajas"
			tone = "energetic"
		else:
			karma_bias = None  # keep existing
			# tone remains None
		return {
			"target_lang_override": target_lang_override,
			"karma_bias": karma_bias,
			"intent_bias": intent_bias,
			"confidence_delta": confidence_delta,
			"tone": tone,
		}


# Singleton accessor to avoid repeatedly reading files on hot paths
_policy_manager: Optional[PolicyManager] = None

def get_policy_manager() -> PolicyManager:
	global _policy_manager
	if _policy_manager is None:
		_policy_manager = PolicyManager()
	return _policy_manager
