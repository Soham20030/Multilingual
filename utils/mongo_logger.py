#!/usr/bin/env python3
"""
MongoDB Logger - Enhanced logging for token/cost data and RL tracking.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import motor.motor_asyncio
from config.settings import MONGO_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class MongoLogger:
    """Enhanced MongoDB logger for comprehensive tracking."""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collections = {}
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize MongoDB connection."""
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
            self.db = self.client[MONGO_CONFIG['database']]
            
            # Initialize collections
            self.collections = {
                'task_logs': self.db.task_logs,
                'token_costs': self.db.token_costs,
                'rl_actions': self.db.rl_actions,
                'model_performance': self.db.model_performance,
                'agent_performance': self.db.agent_performance,
                'fallback_logs': self.db.fallback_logs,
                'nlo_collection': self.db.nlo_collection
            }
            
            logger.info("MongoDB logger initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB logger: {e}")
    
    async def log_task_execution(self, task_data: Dict[str, Any]) -> Optional[str]:
        """Log comprehensive task execution data."""
        try:
            # Enhance task data with metadata
            enhanced_data = {
                **task_data,
                'logged_at': datetime.now(),
                'log_type': 'task_execution'
            }
            
            result = await self.collections['task_logs'].insert_one(enhanced_data)
            logger.debug(f"Logged task execution: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging task execution: {e}")
            return None
    
    async def log_token_cost(self, cost_data: Dict[str, Any]) -> Optional[str]:
        """Log token usage and cost information."""
        try:
            cost_entry = {
                'task_id': cost_data.get('task_id'),
                'model': cost_data.get('model'),
                'agent': cost_data.get('agent'),
                'tokens_used': cost_data.get('tokens_used', 0),
                'cost_estimate': cost_data.get('cost_estimate', 0.0),
                'input_tokens': cost_data.get('input_tokens', 0),
                'output_tokens': cost_data.get('output_tokens', 0),
                'processing_time': cost_data.get('processing_time', 0.0),
                'timestamp': datetime.now(),
                'log_type': 'token_cost'
            }
            
            result = await self.collections['token_costs'].insert_one(cost_entry)
            logger.debug(f"Logged token cost: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging token cost: {e}")
            return None
    
    async def log_rl_action(self, rl_data: Dict[str, Any]) -> Optional[str]:
        """Log RL actions and decisions."""
        try:
            rl_entry = {
                'task_id': rl_data.get('task_id'),
                'agent': rl_data.get('agent'),
                'model': rl_data.get('model'),
                'action_type': rl_data.get('action_type'),  # 'model_selection', 'agent_selection', 'fallback'
                'decision_reason': rl_data.get('decision_reason'),  # 'exploration', 'exploitation', 'fallback'
                'confidence_score': rl_data.get('confidence_score', 0.0),
                'exploration_rate': rl_data.get('exploration_rate', 0.0),
                'reward': rl_data.get('reward', 0.0),
                'metadata': rl_data.get('metadata', {}),
                'timestamp': datetime.now(),
                'log_type': 'rl_action'
            }
            
            result = await self.collections['rl_actions'].insert_one(rl_entry)
            logger.debug(f"Logged RL action: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging RL action: {e}")
            return None
    
    async def log_model_performance(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """Log model performance metrics."""
        try:
            perf_entry = {
                'model': performance_data.get('model'),
                'task_id': performance_data.get('task_id'),
                'reward': performance_data.get('reward', 0.0),
                'success': performance_data.get('success', False),
                'response_time': performance_data.get('response_time', 0.0),
                'error_type': performance_data.get('error_type'),
                'input_type': performance_data.get('input_type'),
                'task_complexity': performance_data.get('task_complexity', 'medium'),
                'timestamp': datetime.now(),
                'log_type': 'model_performance'
            }
            
            result = await self.collections['model_performance'].insert_one(perf_entry)
            logger.debug(f"Logged model performance: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging model performance: {e}")
            return None
    
    async def log_fallback_action(self, fallback_data: Dict[str, Any]) -> Optional[str]:
        """Log fallback actions and their outcomes."""
        try:
            fallback_entry = {
                'task_id': fallback_data.get('task_id'),
                'original_model': fallback_data.get('original_model'),
                'failed_model': fallback_data.get('failed_model'),
                'fallback_model': fallback_data.get('fallback_model'),
                'error_reason': fallback_data.get('error_reason'),
                'retry_count': fallback_data.get('retry_count', 0),
                'fallback_success': fallback_data.get('fallback_success', False),
                'total_attempts': fallback_data.get('total_attempts', 1),
                'timestamp': datetime.now(),
                'log_type': 'fallback_action'
            }
            
            result = await self.collections['fallback_logs'].insert_one(fallback_entry)
            logger.debug(f"Logged fallback action: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging fallback action: {e}")
            return None
    
    async def get_cost_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get cost analytics for specified time range."""
        try:
            start_time = datetime.now() - timedelta(hours=time_range_hours)
            
            pipeline = [
                {'$match': {'timestamp': {'$gte': start_time}}},
                {'$group': {
                    '_id': '$model',
                    'total_cost': {'$sum': '$cost_estimate'},
                    'total_tokens': {'$sum': '$tokens_used'},
                    'task_count': {'$sum': 1},
                    'avg_cost_per_task': {'$avg': '$cost_estimate'},
                    'avg_tokens_per_task': {'$avg': '$tokens_used'}
                }},
                {'$sort': {'total_cost': -1}}
            ]
            
            results = await self.collections['token_costs'].aggregate(pipeline).to_list(None)
            
            total_cost = sum(r['total_cost'] for r in results)
            total_tokens = sum(r['total_tokens'] for r in results)
            total_tasks = sum(r['task_count'] for r in results)
            
            return {
                'time_range_hours': time_range_hours,
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'total_tasks': total_tasks,
                'avg_cost_per_task': total_cost / total_tasks if total_tasks > 0 else 0,
                'model_breakdown': results,
                'generated_at': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting cost analytics: {e}")
            return {}
    
    async def get_fallback_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get fallback frequency analytics."""
        try:
            start_time = datetime.now() - timedelta(hours=time_range_hours)
            
            # Get fallback frequency by model
            pipeline = [
                {'$match': {'timestamp': {'$gte': start_time}}},
                {'$group': {
                    '_id': '$failed_model',
                    'fallback_count': {'$sum': 1},
                    'success_rate': {'$avg': {'$cond': ['$fallback_success', 1, 0]}},
                    'avg_retry_count': {'$avg': '$retry_count'}
                }},
                {'$sort': {'fallback_count': -1}}
            ]
            
            fallback_stats = await self.collections['fallback_logs'].aggregate(pipeline).to_list(None)
            
            # Get total task count for comparison
            total_tasks = await self.collections['task_logs'].count_documents({
                'logged_at': {'$gte': start_time}
            })
            
            total_fallbacks = sum(stat['fallback_count'] for stat in fallback_stats)
            
            return {
                'time_range_hours': time_range_hours,
                'total_tasks': total_tasks,
                'total_fallbacks': total_fallbacks,
                'fallback_rate': total_fallbacks / total_tasks if total_tasks > 0 else 0,
                'model_fallback_stats': fallback_stats,
                'generated_at': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting fallback analytics: {e}")
            return {}
    
    async def get_rl_performance_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get RL performance summary."""
        try:
            start_time = datetime.now() - timedelta(hours=time_range_hours)
            
            # Get RL action statistics
            pipeline = [
                {'$match': {'timestamp': {'$gte': start_time}}},
                {'$group': {
                    '_id': {
                        'action_type': '$action_type',
                        'decision_reason': '$decision_reason'
                    },
                    'count': {'$sum': 1},
                    'avg_reward': {'$avg': '$reward'},
                    'avg_confidence': {'$avg': '$confidence_score'}
                }},
                {'$sort': {'count': -1}}
            ]
            
            rl_stats = await self.collections['rl_actions'].aggregate(pipeline).to_list(None)
            
            return {
                'time_range_hours': time_range_hours,
                'rl_action_stats': rl_stats,
                'generated_at': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting RL performance summary: {e}")
            return {}
    
    async def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Global instance
mongo_logger = MongoLogger()

# Convenience functions for backward compatibility
async def log_to_mongo(collection_name: str, data: Dict[str, Any]) -> Optional[str]:
    """Log data to specified MongoDB collection."""
    if collection_name in mongo_logger.collections:
        try:
            result = await mongo_logger.collections[collection_name].insert_one({
                **data,
                'logged_at': datetime.now()
            })
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error logging to {collection_name}: {e}")
    return None
