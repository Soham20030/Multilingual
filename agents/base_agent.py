import logging
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent:
    """Base agent with memory scaffold."""
    def __init__(self):
        self.memory = {}  # Placeholder for short-term JSON cache
    
    def store_memory(self, key: str, value: str):
        logger.info(f"Storing memory: {key}")
        self.memory[key] = value