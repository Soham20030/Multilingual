import random
import math
import uuid
import threading
from typing import Dict, Any, List, Optional
from utils.logger import get_logger
from reinforcement.rl_context import rl_context
from config.settings import MODEL_CONFIG, RL_CONFIG

logger = get_logger(__name__)

class ModelSelector:
    """Enhanced RL-based model selector with UCB and dynamic exploration."""
    def __init__(self, exploration_rate: Optional[float] = None):
        self.base_exploration_rate = exploration_rate or RL_CONFIG.get('exploration_rate', 0.2)
        self.exploration_rate = self.base_exploration_rate
        self.history = {}  # Tracks model performance: {model: {'rewards': [], 'count': int, 'avg_reward': float}}
        self.models = list(MODEL_CONFIG.keys())
        self.total_selections = 0
        self.exploration_decay = 0.995
        self.task_type_weights = {
            'text': {'edumentor_agent': 1.2, 'vedas_agent': 1.0, 'wellness_agent': 1.0},
            'pdf': {'edumentor_agent': 1.5, 'vedas_agent': 0.8, 'wellness_agent': 0.8},
            'image': {'edumentor_agent': 1.0, 'vedas_agent': 0.7, 'wellness_agent': 0.7},
            'audio': {'edumentor_agent': 1.0, 'vedas_agent': 0.7, 'wellness_agent': 0.7},
            'multi': {'edumentor_agent': 1.3, 'vedas_agent': 0.9, 'wellness_agent': 0.9}
        }
        self._lock = threading.RLock()  # Thread-safe operations

    def calculate_dynamic_exploration_rate(self, task_type: str = 'text') -> float:
        """Calculate exploration rate based on task type and experience."""
        # Decay exploration over time
        decayed_rate = self.base_exploration_rate * (self.exploration_decay ** self.total_selections)
        min_exploration = 0.05  # Minimum 5% exploration

        # Adjust for task complexity
        complexity_multipliers = {
            'text': 1.0,
            'pdf': 1.2,
            'image': 1.5,
            'audio': 1.8,
            'multi': 2.0
        }

        complexity_factor = complexity_multipliers.get(task_type, 1.0)
        dynamic_rate = max(min_exploration, decayed_rate * complexity_factor)

        return min(dynamic_rate, 0.7)  # Cap at 70%

    def get_model_weights(self, task_type: str) -> Dict[str, float]:
        """Get task-specific model weights."""
        return self.task_type_weights.get(task_type, {model: 1.0 for model in self.models})

    def calculate_ucb_score(self, model: str, task_type: str = 'text') -> float:
        """Calculate Upper Confidence Bound score for a model."""
        if model not in self.history:
            return float('inf')  # Unexplored models get highest priority

        model_data = self.history[model]
        avg_reward = model_data.get('avg_reward', 0)
        count = model_data.get('count', 1)

        # UCB formula with task-specific weights
        confidence_interval = math.sqrt(2 * math.log(max(self.total_selections, 1)) / count)
        task_weight = self.get_model_weights(task_type).get(model, 1.0)

        ucb_score = (avg_reward * task_weight) + confidence_interval
        return ucb_score

    def select_model(self, task: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Enhanced model selection using UCB and dynamic exploration."""
        task_id = task.get('task_id', str(uuid.uuid4()))
        task_type = task.get('input_type', 'text')

        with self._lock:
            self.total_selections += 1

        # Calculate dynamic exploration rate
        current_exploration_rate = self.calculate_dynamic_exploration_rate(task_type)

        # Epsilon-greedy with UCB
        if random.random() < current_exploration_rate:
            # Exploration: weighted random selection based on task type
            model_weights = self.get_model_weights(task_type)
            available_models = [m for m in self.models if m in model_weights]

            if available_models:
                # Weighted random selection
                weights = [model_weights[m] for m in available_models]
                model = random.choices(available_models, weights=weights)[0]
            else:
                model = random.choice(self.models)

            logger.info(f"Exploration (rate={current_exploration_rate:.3f}): Selected model {model} for {task_type} task {task_id}")
            rl_context.log_action(task_id, task.get('agent', 'unknown'), model, "model_selection",
                                {"reason": "exploration", "exploration_rate": current_exploration_rate})
            return model

        else:
            # Exploitation: UCB selection
            ucb_scores = {}
            for model in self.models:
                ucb_scores[model] = self.calculate_ucb_score(model, task_type)

            # Select model with highest UCB score
            best_model = max(ucb_scores.keys(), key=lambda x: ucb_scores[x])
            ucb_score = ucb_scores[best_model]
            ucb_display = "∞" if ucb_score == float('inf') else f"{ucb_score:.3f}"

            logger.info(f"UCB Exploitation: Selected model {best_model} (UCB={ucb_display}) for {task_type} task {task_id}")
            rl_context.log_action(task_id, task.get('agent', 'unknown'), best_model, "model_selection",
                                {"reason": "exploitation", "ucb_score": ucb_score})
            return best_model

    def update_history(self, task_id: str, model: str, reward: float):
        """Update model performance history with enhanced tracking."""
        with self._lock:
            if model not in self.history:
                self.history[model] = {'rewards': [], 'count': 0, 'avg_reward': 0.0, 'total_reward': 0.0}

            # Update model statistics
            self.history[model]['rewards'].append(reward)
            self.history[model]['count'] += 1
            self.history[model]['total_reward'] += reward
            self.history[model]['avg_reward'] = self.history[model]['total_reward'] / self.history[model]['count']

            # Keep only recent rewards (sliding window of 100)
            if len(self.history[model]['rewards']) > 100:
                old_reward = self.history[model]['rewards'].pop(0)
                self.history[model]['total_reward'] -= old_reward
                self.history[model]['count'] = len(self.history[model]['rewards'])
                self.history[model]['avg_reward'] = self.history[model]['total_reward'] / self.history[model]['count']

        logger.info(f"Updated model {model} history: reward={reward:.3f}, avg_reward={self.history[model]['avg_reward']:.3f}, count={self.history[model]['count']}")

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        summary = {}
        for model, data in self.history.items():
            summary[model] = {
                'avg_reward': data['avg_reward'],
                'count': data['count'],
                'confidence': self.calculate_confidence(model),
                'recent_performance': data['rewards'][-10:] if len(data['rewards']) >= 10 else data['rewards']
            }
        return summary

    def calculate_confidence(self, model: str) -> float:
        """Calculate confidence in model performance."""
        if model not in self.history:
            return 0.0

        data = self.history[model]
        count = data['count']

        if count < 5:
            return 0.2  # Low confidence for few samples
        elif count < 20:
            return 0.6  # Medium confidence
        else:
            return 0.9  # High confidence for many samples

model_selector = ModelSelector()