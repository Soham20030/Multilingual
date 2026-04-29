from typing import Dict, Any
from utils.logger import get_logger
from reinforcement.rl_context import rl_context
from datetime import datetime

logger = get_logger(__name__)

def get_reward_from_output(output: Dict[str, Any], task_id: str) -> float:
    """Compute reward based on output quality."""
    try:
        result = output.get('result') or output.get('response', '')
        status = output.get('status', 200)  # Use status from API response
        keywords = output.get('keywords') or output.get('sources', [])
        
        reward = 1.0 if status == 200 else 0.0
        clarity_score = 0.0
        tag_count = 0
        if result:
            word_count = len(result.split())
            clarity_score = min(word_count / 100.0, 1.0)
            reward += clarity_score * 0.5
        if keywords:
            tag_count = len(keywords)
            reward += tag_count * 0.1
            
        metrics = {
            "clarity_score": clarity_score,
            "tag_count": tag_count,
            "status": status
        }
        logger.info(f"Computed reward {reward} for task {task_id}")
        rl_context.log_reward(task_id, reward, metrics)
        return reward
    except Exception as e:
        logger.error(f"Error computing reward for task {task_id}: {e}")
        metrics = {"error": str(e), "status": output.get('status', 500)}
        rl_context.log_reward(task_id, 0.0, metrics)
        return 0.0

# from typing import Dict, Any
# from utils.logger import get_logger
# from reinforcement.rl_context import rl_context

# logger = get_logger(__name__)

# def get_reward_from_output(output: Dict[str, Any], task_id: str) -> float:
#     """Compute reward based on output quality."""
#     try:
#         # Handle different output structures (e.g., vedas_agent uses 'response' instead of 'result')
#         result = output.get('result') or output.get('response', '')
#         status = output.get('status', 500)
#         keywords = output.get('keywords') or output.get('sources', [])  # Fallback to sources if keywords missing
        
#         # Base reward: 1.0 for success, 0.0 for failure
#         reward = 1.0 if status == 200 else 0.0
        
#         # Bonus for content quality
#         clarity_score = 0.0
#         tag_count = 0
#         if result:
#             word_count = len(result.split())
#             clarity_score = min(word_count / 100.0, 1.0)  # Reward based on output length
#             reward += clarity_score * 0.5
#         if keywords:
#             tag_count = len(keywords)
#             reward += tag_count * 0.1  # Weight tags
            
#         metrics = {
#             "clarity_score": clarity_score,
#             "tag_count": tag_count,
#             "status": status
#         }
#         logger.info(f"Computed reward {reward} for task {task_id}")
#         rl_context.log_reward(task_id, reward, metrics)
#         return reward
#     except Exception as e:
#         logger.error(f"Error computing reward for task {task_id}: {str(e)}")
#         metrics = {"error": str(e), "status": output.get('status', 500)}
#         rl_context.log_reward(task_id, 0.0, metrics)
#         return 0.0