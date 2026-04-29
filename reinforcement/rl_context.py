import uuid
from datetime import datetime
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class RLContext:
    """Centralized RL context for logging actions and rewards."""
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.task_logs = []

    def log_action(self, task_id: str, agent: str, model: str, action: str, metadata: Dict[str, Any] = None):
        """Log an RL action (e.g., agent selection, model choice)."""
        action_entry = {
            "task_id": task_id,
            "agent": agent,
            "model": model,
            "action": action,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.actions.append(action_entry)
        logger.info(f"Logged RL action: {action_entry}")
        return action_entry

    def log_reward(self, task_id: str, reward: float, metrics: Dict[str, Any] = None):
        """Log a reward for a task."""
        reward_entry = {
            "task_id": task_id,
            "reward": reward,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        self.rewards.append(reward_entry)
        logger.info(f"Logged RL reward: {reward_entry}")
        return reward_entry

    def log_task(self, task_id: str, input_data: str, output_data: Dict[str, Any], agent: str, model: str):
        """Log task details for RL analysis."""
        task_entry = {
            "task_id": task_id,
            "input": input_data,
            "output": output_data,
            "agent": agent,
            "model": model,
            "timestamp": datetime.now().isoformat()
        }
        self.task_logs.append(task_entry)
        logger.info(f"Logged RL task: {task_entry}")
        return task_entry

rl_context = RLContext()