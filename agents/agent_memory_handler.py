import logging
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)

class AgentMemoryHandler:
    """Enhanced agent memory handler with caching and persistence."""

    def __init__(self, max_memory_per_agent: int = 100, memory_file: str = "logs/agent_memory.json"):
        self.max_memory_per_agent = max_memory_per_agent
        self.memory_file = memory_file
        self.agent_memories = defaultdict(lambda: deque(maxlen=max_memory_per_agent))
        self.agent_stats = defaultdict(lambda: {
            'total_tasks': 0,
            'successful_tasks': 0,
            'avg_response_time': 0.0,
            'last_activity': None,
            'preferred_models': defaultdict(int)
        })
        self._lock = threading.RLock()  # Thread-safe operations
        self.load_memory()

    def log_memory(self, agent_name: str, data: str):
        """Legacy method for backward compatibility."""
        logger.info(f"Logging memory for {agent_name}: {data}")
        self.add_memory(agent_name, {"input": data, "timestamp": datetime.now().isoformat()})

    def add_memory(self, agent_name: str, memory_entry: Dict[str, Any]) -> None:
        """Add a memory entry for an agent."""
        with self._lock:
            # Ensure timestamp is present
            if 'timestamp' not in memory_entry:
                memory_entry['timestamp'] = datetime.now().isoformat()

            # Add to memory
            self.agent_memories[agent_name].append(memory_entry)

            # Update stats
            self.agent_stats[agent_name]['total_tasks'] += 1
            self.agent_stats[agent_name]['last_activity'] = memory_entry['timestamp']

            if memory_entry.get('status') == 200:
                self.agent_stats[agent_name]['successful_tasks'] += 1

            # Track model usage
            if 'model' in memory_entry:
                self.agent_stats[agent_name]['preferred_models'][memory_entry['model']] += 1

            # Update average response time
            if 'response_time' in memory_entry:
                current_avg = self.agent_stats[agent_name]['avg_response_time']
                total_tasks = self.agent_stats[agent_name]['total_tasks']
                new_avg = ((current_avg * (total_tasks - 1)) + memory_entry['response_time']) / total_tasks
                self.agent_stats[agent_name]['avg_response_time'] = new_avg

            logger.debug(f"Added memory for {agent_name}: {len(self.agent_memories[agent_name])} entries")

    def get_recent_memories(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories for an agent."""
        memories = list(self.agent_memories[agent_name])
        return memories[-limit:] if memories else []

    def get_agent_context(self, agent_name: str, task_type: str = None) -> Dict[str, Any]:
        """Get contextual information for an agent."""
        recent_memories = self.get_recent_memories(agent_name, 5)
        stats = self.agent_stats[agent_name]

        # Filter memories by task type if specified
        if task_type:
            recent_memories = [m for m in recent_memories if m.get('input_type') == task_type]

        # Calculate success rate
        success_rate = (stats['successful_tasks'] / stats['total_tasks']) if stats['total_tasks'] > 0 else 0.0

        # Get preferred model
        preferred_model = max(stats['preferred_models'].items(), key=lambda x: x[1])[0] if stats['preferred_models'] else None

        return {
            'agent_name': agent_name,
            'recent_memories': recent_memories,
            'total_tasks': stats['total_tasks'],
            'success_rate': success_rate,
            'avg_response_time': stats['avg_response_time'],
            'last_activity': stats['last_activity'],
            'preferred_model': preferred_model,
            'memory_count': len(self.agent_memories[agent_name])
        }

    def get_similar_tasks(self, agent_name: str, current_input: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar past tasks for an agent."""
        memories = list(self.agent_memories[agent_name])

        # Simple similarity based on common words
        current_words = set(current_input.lower().split())
        similar_tasks = []

        for memory in memories:
            if 'input' in memory:
                memory_words = set(memory['input'].lower().split())
                similarity = len(current_words.intersection(memory_words)) / len(current_words.union(memory_words))
                if similarity > 0.2:  # Threshold for similarity
                    memory['similarity'] = similarity
                    similar_tasks.append(memory)

        # Sort by similarity and return top results
        similar_tasks.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_tasks[:limit]

    def cleanup_old_memories(self, days_to_keep: int = 30) -> None:
        """Remove memories older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for agent_name in self.agent_memories:
            original_count = len(self.agent_memories[agent_name])

            # Filter out old memories
            filtered_memories = deque(maxlen=self.max_memory_per_agent)
            for memory in self.agent_memories[agent_name]:
                memory_date = datetime.fromisoformat(memory['timestamp'])
                if memory_date > cutoff_date:
                    filtered_memories.append(memory)

            self.agent_memories[agent_name] = filtered_memories
            removed_count = original_count - len(filtered_memories)

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old memories for {agent_name}")

    def save_memory(self) -> None:
        """Save memory to persistent storage."""
        try:
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

            with self._lock:
                # Convert deques to lists for JSON serialization
                memory_data = {
                    'agent_memories': {
                        agent: list(memories) for agent, memories in self.agent_memories.items()
                    },
                    'agent_stats': dict(self.agent_stats),
                    'last_saved': datetime.now().isoformat()
                }

            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2, default=str)

            logger.debug(f"Saved agent memory to {self.memory_file}")
        except Exception as e:
            logger.error(f"Error saving agent memory: {e}")

    def load_memory(self) -> None:
        """Load memory from persistent storage."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)

                with self._lock:
                    # Restore agent memories
                    for agent, memories in memory_data.get('agent_memories', {}).items():
                        self.agent_memories[agent] = deque(memories, maxlen=self.max_memory_per_agent)

                    # Restore agent stats
                    for agent, stats in memory_data.get('agent_stats', {}).items():
                        self.agent_stats[agent].update(stats)
                        # Convert preferred_models back to defaultdict
                        if 'preferred_models' in stats:
                            self.agent_stats[agent]['preferred_models'] = defaultdict(int, stats['preferred_models'])

                logger.info(f"Loaded agent memory from {self.memory_file}")
        except Exception as e:
            logger.warning(f"Could not load agent memory: {e}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all agent memories."""
        summary = {
            'total_agents': len(self.agent_memories),
            'total_memories': sum(len(memories) for memories in self.agent_memories.values()),
            'agents': {}
        }

        for agent_name in self.agent_memories:
            context = self.get_agent_context(agent_name)
            summary['agents'][agent_name] = {
                'memory_count': context['memory_count'],
                'success_rate': context['success_rate'],
                'total_tasks': context['total_tasks'],
                'preferred_model': context['preferred_model']
            }

        return summary

# Global instance
agent_memory_handler = AgentMemoryHandler()