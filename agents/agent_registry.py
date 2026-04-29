import json
import os
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class AgentRegistry:
    """Registry for managing agent configurations and routing."""
    
    def __init__(self, config_file: str = "config/agent_configs.json"):
        self.config_file = config_file
        self.agents = {}
        self.load_agents()
    
    def load_agents(self):
        """Load agent configurations from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.agents = json.load(f)
                logger.info(f"Loaded {len(self.agents)} agents from {self.config_file}")
            else:
                logger.warning(f"Agent config file not found: {self.config_file}")
                self.agents = {}
        except Exception as e:
            logger.error(f"Error loading agent configs: {e}")
            self.agents = {}
    
    def find_agent(self, task_context: Dict[str, Any]) -> str:
        """Find appropriate agent based on task context."""
        if isinstance(task_context, str):
            # If passed a string, treat it as agent name
            return task_context
        
        # Extract agent from task context
        agent_name = task_context.get('model', task_context.get('agent', 'edumentor_agent'))
        input_type = task_context.get('input_type', 'text')
        
        # Route based on input type if no specific agent
        if agent_name == 'edumentor_agent' and input_type != 'text':
            type_mapping = {
                'pdf': 'archive_agent',
                'image': 'image_agent', 
                'audio': 'audio_agent'
            }
            agent_name = type_mapping.get(input_type, 'edumentor_agent')
        
        return agent_name
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name."""
        return self.agents.get(agent_name)
    
    def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name (alias for get_agent_config)."""
        return self.get_agent_config(agent_name)
    
    def list_agents(self) -> Dict[str, Any]:
        """List all available agents."""
        return self.agents
    
    def register_agent(self, agent_name: str, config: Dict[str, Any]):
        """Register a new agent configuration."""
        self.agents[agent_name] = config
        logger.info(f"Registered agent: {agent_name}")
    
    def is_agent_available(self, agent_name: str) -> bool:
        """Check if an agent is available."""
        return agent_name in self.agents

# Global agent registry instance
agent_registry = AgentRegistry()


