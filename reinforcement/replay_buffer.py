from datetime import datetime
import json
from typing import Dict, Any, List
from utils.logger import get_logger
from reinforcement.rl_context import rl_context
import os

logger = get_logger(__name__)

class ReplayBuffer:
    """Stores past runs for RL training."""
    def __init__(self, buffer_file: str = "logs/learning_log.json"):
        self.buffer_file = buffer_file
        self.buffer = []
        self.load_buffer()

    def load_buffer(self):
        """Load existing buffer from file."""
        try:
            if os.path.exists(self.buffer_file):
                with open(self.buffer_file, 'r') as f:
                    self.buffer = json.load(f)
                logger.info(f"Loaded replay buffer from {self.buffer_file}")
        except Exception as e:
            logger.error(f"Error loading replay buffer: {str(e)}")
            self.buffer = []

    def save_buffer(self):
        """Save buffer to file."""
        try:
            with open(self.buffer_file, 'w') as f:
                json.dump(self.buffer, f, indent=2)
            logger.info(f"Saved replay buffer to {self.buffer_file}")
        except Exception as e:
            logger.error(f"Error saving replay buffer: {str(e)}")

    def add_run(self, task_id: str, input_data: str, output_data: Dict[str, Any], agent: str, model: str, reward: float):
        """Add a run to the buffer."""
        run_entry = {
            "task_id": task_id,
            "input": input_data,
            "output": output_data,
            "agent": agent,
            "model": model,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }
        self.buffer.append(run_entry)
        self.save_buffer()
        logger.info(f"Added run to replay buffer: {task_id}")
        rl_context.log_task(task_id, input_data, output_data, agent, model)

replay_buffer = ReplayBuffer()