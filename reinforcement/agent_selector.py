import random
import math
import logging
from typing import Dict, Any, Optional
from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from config.settings import RL_CONFIG

logger = get_logger(__name__)

class AgentSelector:
    def __init__(self):
        self.base_exploration_rate = RL_CONFIG.get('exploration_rate', 0.2)
        self.exploration_rate = self.base_exploration_rate
        self.agent_scores = {}  # Track performance of agents
        self.task_complexity_weights = {
            'text': 1.0,
            'pdf': 1.5,
            'image': 2.0,
            'audio': 2.5,
            'multi': 3.0
        }
        self.total_tasks = 0
        self.exploration_decay = 0.995  # Decay exploration over time

    def calculate_dynamic_exploration_rate(self, input_type: str) -> float:
        """Calculate exploration rate based on task complexity and experience."""
        # Adjust exploration based on task complexity
        complexity_multiplier = self.task_complexity_weights.get(input_type, 1.0)

        # Decay exploration over time but maintain minimum
        decayed_rate = self.base_exploration_rate * (self.exploration_decay ** self.total_tasks)
        min_exploration = 0.05  # Minimum 5% exploration

        # Higher exploration for complex tasks
        dynamic_rate = max(min_exploration, decayed_rate * complexity_multiplier)

        return min(dynamic_rate, 0.8)  # Cap at 80%

    def get_agent_confidence(self, agent_id: str) -> float:
        """Get confidence score for an agent based on historical performance."""
        if agent_id not in self.agent_scores:
            return 0.5  # Neutral confidence for new agents

        agent_data = self.agent_scores[agent_id]
        avg_reward = agent_data.get('avg_reward', 0)
        task_count = agent_data.get('count', 0)

        # Confidence increases with both performance and experience
        if task_count == 0:
            return 0.5

        # Normalize reward to 0-1 range (assuming rewards are typically 0-1)
        normalized_reward = max(0, min(1, avg_reward))

        # Confidence bonus for experience (diminishing returns)
        experience_bonus = math.log(task_count + 1) / 10

        return min(1.0, normalized_reward + experience_bonus)

    def select_agent(self, payload: Dict[str, Any]) -> Optional[str]:
        task_id = payload.get('task_id', '')
        input_data = payload.get('input', '')
        input_type = payload.get('input_type', 'text')

        # Use RL if enabled
        if not RL_CONFIG.get('use_rl', True):
            return None

        self.total_tasks += 1

        # Calculate dynamic exploration rate
        current_exploration_rate = self.calculate_dynamic_exploration_rate(input_type)

        # Get available agents from registry
        from agents.agent_registry import agent_registry  # Lazy import
        available_agents = [
            agent_id for agent_id, config in agent_registry.agent_configs.items()
            if input_type in config.get('input_types', []) and config.get('enabled', True)
        ]

        if not available_agents:
            # Fallback agents if registry doesn't have appropriate agents
            agent_mapping = {
                'text': ['text_agent'],
                'pdf': ['archive_agent'],
                'image': ['image_agent'],
                'audio': ['audio_agent'],
                'multi': ['text_agent', 'archive_agent', 'image_agent', 'audio_agent']
            }
            available_agents = agent_mapping.get(input_type, ['text_agent'])

        # Epsilon-greedy with Upper Confidence Bound (UCB) for exploitation
        if random.random() < current_exploration_rate:
            # Exploration: random selection from appropriate agents
            selected = random.choice(available_agents)
            logger.info(f"Exploration (rate={current_exploration_rate:.3f}): selected {selected} for {input_type} task {task_id}")
            return selected
        else:
            # Exploitation: UCB selection
            if not self.agent_scores:
                selected = random.choice(available_agents)
                logger.info(f"No history: selected {selected} for task {task_id}")
                return selected

            # Calculate UCB scores for available agents
            ucb_scores = {}
            for agent in available_agents:
                if agent in self.agent_scores:
                    agent_data = self.agent_scores[agent]
                    avg_reward = agent_data.get('avg_reward', 0)
                    count = agent_data.get('count', 1)

                    # UCB formula: avg_reward + sqrt(2 * ln(total_tasks) / count)
                    confidence_interval = math.sqrt(2 * math.log(max(self.total_tasks, 1)) / count)
                    ucb_scores[agent] = avg_reward + confidence_interval
                else:
                    # High UCB for unexplored agents
                    ucb_scores[agent] = float('inf')

            # Select agent with highest UCB score
            best_agent = max(ucb_scores.keys(), key=lambda x: ucb_scores[x])
            ucb_score = ucb_scores[best_agent]
            ucb_display = "∞" if ucb_score == float('inf') else f"{ucb_score:.3f}"
            logger.info(f"UCB Exploitation: selected {best_agent} (UCB={ucb_display}) for task {task_id}")
            return best_agent

    def update_history(self, task_id: str, agent_id: str, reward: float) -> None:
        if agent_id not in self.agent_scores:
            self.agent_scores[agent_id] = {'total_reward': 0, 'count': 0}
        self.agent_scores[agent_id]['total_reward'] += reward
        self.agent_scores[agent_id]['count'] += 1
        self.agent_scores[agent_id]['avg_reward'] = (
            self.agent_scores[agent_id]['total_reward'] / self.agent_scores[agent_id]['count']
        )
        logger.info(f"Updated agent {agent_id} with reward {reward}, avg_reward: {self.agent_scores[agent_id]['avg_reward']}")
        replay_buffer.add_selection(task_id, agent_id, reward)

agent_selector = AgentSelector()