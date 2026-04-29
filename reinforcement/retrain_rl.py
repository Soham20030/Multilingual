#!/usr/bin/env python3
"""
RL Retraining Script - Periodically retrain RL models using replay buffer data.
"""

import json
import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any
from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from reinforcement.agent_selector import agent_selector
from reinforcement.model_selector import model_selector
from reinforcement.reward_functions import get_reward_from_output
from config.settings import RL_CONFIG

logger = get_logger(__name__)

class RLRetrainer:
    """Handles retraining of RL components using historical data."""
    
    def __init__(self):
        self.min_samples_for_retraining = 50
        self.retraining_interval_hours = 24
        self.performance_threshold = 0.1  # Minimum improvement to update models
        self.backup_dir = "logs/rl_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def load_replay_data(self) -> List[Dict[str, Any]]:
        """Load data from replay buffer."""
        try:
            replay_buffer.load_buffer()
            logger.info(f"Loaded {len(replay_buffer.buffer)} samples from replay buffer")
            return replay_buffer.buffer
        except Exception as e:
            logger.error(f"Error loading replay data: {str(e)}")
            return []
    
    def analyze_agent_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agent performance from historical data."""
        agent_stats = {}
        
        for entry in data:
            agent = entry.get('agent', 'unknown')
            reward = entry.get('reward', 0.0)
            input_type = entry.get('input_type', 'text')
            
            if agent not in agent_stats:
                agent_stats[agent] = {
                    'total_reward': 0.0,
                    'count': 0,
                    'rewards': [],
                    'input_types': {},
                    'recent_performance': []
                }
            
            agent_stats[agent]['total_reward'] += reward
            agent_stats[agent]['count'] += 1
            agent_stats[agent]['rewards'].append(reward)
            
            # Track performance by input type
            if input_type not in agent_stats[agent]['input_types']:
                agent_stats[agent]['input_types'][input_type] = {'rewards': [], 'count': 0}
            agent_stats[agent]['input_types'][input_type]['rewards'].append(reward)
            agent_stats[agent]['input_types'][input_type]['count'] += 1
        
        # Calculate averages and recent performance
        for agent, stats in agent_stats.items():
            stats['avg_reward'] = stats['total_reward'] / stats['count'] if stats['count'] > 0 else 0
            stats['recent_performance'] = stats['rewards'][-20:]  # Last 20 tasks
            stats['recent_avg'] = sum(stats['recent_performance']) / len(stats['recent_performance']) if stats['recent_performance'] else 0
            
            # Calculate input type averages
            for input_type, type_stats in stats['input_types'].items():
                type_stats['avg_reward'] = sum(type_stats['rewards']) / type_stats['count'] if type_stats['count'] > 0 else 0
        
        return agent_stats
    
    def analyze_model_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze model performance from historical data."""
        model_stats = {}
        
        for entry in data:
            model = entry.get('model', 'unknown')
            reward = entry.get('reward', 0.0)
            input_type = entry.get('input_type', 'text')
            
            if model not in model_stats:
                model_stats[model] = {
                    'total_reward': 0.0,
                    'count': 0,
                    'rewards': [],
                    'input_types': {},
                    'recent_performance': []
                }
            
            model_stats[model]['total_reward'] += reward
            model_stats[model]['count'] += 1
            model_stats[model]['rewards'].append(reward)
            
            # Track performance by input type
            if input_type not in model_stats[model]['input_types']:
                model_stats[model]['input_types'][input_type] = {'rewards': [], 'count': 0}
            model_stats[model]['input_types'][input_type]['rewards'].append(reward)
            model_stats[model]['input_types'][input_type]['count'] += 1
        
        # Calculate averages
        for model, stats in model_stats.items():
            stats['avg_reward'] = stats['total_reward'] / stats['count'] if stats['count'] > 0 else 0
            stats['recent_performance'] = stats['rewards'][-20:]
            stats['recent_avg'] = sum(stats['recent_performance']) / len(stats['recent_performance']) if stats['recent_performance'] else 0
            
            for input_type, type_stats in stats['input_types'].items():
                type_stats['avg_reward'] = sum(type_stats['rewards']) / type_stats['count'] if type_stats['count'] > 0 else 0
        
        return model_stats
    
    def backup_current_models(self):
        """Backup current RL model states."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup agent selector state
        agent_backup = {
            'agent_scores': agent_selector.agent_scores,
            'exploration_rate': agent_selector.exploration_rate,
            'total_tasks': agent_selector.total_tasks,
            'timestamp': timestamp
        }
        
        with open(f"{self.backup_dir}/agent_selector_{timestamp}.json", 'w') as f:
            json.dump(agent_backup, f, indent=2)
        
        # Backup model selector state
        model_backup = {
            'history': model_selector.history,
            'exploration_rate': model_selector.exploration_rate,
            'total_selections': model_selector.total_selections,
            'timestamp': timestamp
        }
        
        with open(f"{self.backup_dir}/model_selector_{timestamp}.json", 'w') as f:
            json.dump(model_backup, f, indent=2)
        
        logger.info(f"Backed up RL models to {self.backup_dir} with timestamp {timestamp}")
    
    def update_agent_selector(self, agent_stats: Dict[str, Any]) -> bool:
        """Update agent selector with new performance data."""
        updated = False
        
        for agent, stats in agent_stats.items():
            if stats['count'] >= 5:  # Minimum samples for reliable update
                current_avg = agent_selector.agent_scores.get(agent, {}).get('avg_reward', 0)
                new_avg = stats['avg_reward']
                
                # Update if significant improvement or new agent
                if abs(new_avg - current_avg) > self.performance_threshold or agent not in agent_selector.agent_scores:
                    agent_selector.agent_scores[agent] = {
                        'total_reward': stats['total_reward'],
                        'count': stats['count'],
                        'avg_reward': new_avg
                    }
                    logger.info(f"Updated agent {agent}: avg_reward {current_avg:.3f} -> {new_avg:.3f}")
                    updated = True
        
        return updated
    
    def update_model_selector(self, model_stats: Dict[str, Any]) -> bool:
        """Update model selector with new performance data."""
        updated = False
        
        for model, stats in model_stats.items():
            if stats['count'] >= 5:  # Minimum samples for reliable update
                current_data = model_selector.history.get(model, {})
                current_avg = current_data.get('avg_reward', 0)
                new_avg = stats['avg_reward']
                
                # Update if significant improvement or new model
                if abs(new_avg - current_avg) > self.performance_threshold or model not in model_selector.history:
                    model_selector.history[model] = {
                        'rewards': stats['rewards'][-50:],  # Keep recent 50 rewards
                        'count': min(stats['count'], 50),
                        'avg_reward': new_avg,
                        'total_reward': sum(stats['rewards'][-50:])
                    }
                    logger.info(f"Updated model {model}: avg_reward {current_avg:.3f} -> {new_avg:.3f}")
                    updated = True
        
        return updated
    
    def adjust_exploration_rates(self, agent_stats: Dict[str, Any], model_stats: Dict[str, Any]):
        """Dynamically adjust exploration rates based on performance variance."""
        # Calculate performance variance for agents
        agent_variances = []
        for stats in agent_stats.values():
            if len(stats['rewards']) > 1:
                mean = stats['avg_reward']
                variance = sum((r - mean) ** 2 for r in stats['rewards']) / len(stats['rewards'])
                agent_variances.append(variance)
        
        # Calculate performance variance for models
        model_variances = []
        for stats in model_stats.values():
            if len(stats['rewards']) > 1:
                mean = stats['avg_reward']
                variance = sum((r - mean) ** 2 for r in stats['rewards']) / len(stats['rewards'])
                model_variances.append(variance)
        
        # Adjust exploration rates based on variance
        if agent_variances:
            avg_agent_variance = sum(agent_variances) / len(agent_variances)
            # Higher variance suggests need for more exploration
            new_agent_exploration = min(0.5, max(0.05, agent_selector.base_exploration_rate * (1 + avg_agent_variance)))
            agent_selector.exploration_rate = new_agent_exploration
            logger.info(f"Adjusted agent exploration rate to {new_agent_exploration:.3f} based on variance {avg_agent_variance:.3f}")
        
        if model_variances:
            avg_model_variance = sum(model_variances) / len(model_variances)
            new_model_exploration = min(0.5, max(0.05, model_selector.base_exploration_rate * (1 + avg_model_variance)))
            model_selector.exploration_rate = new_model_exploration
            logger.info(f"Adjusted model exploration rate to {new_model_exploration:.3f} based on variance {avg_model_variance:.3f}")
    
    def retrain(self, force: bool = False) -> Dict[str, Any]:
        """Main retraining function."""
        logger.info("Starting RL retraining process")
        
        # Load replay data
        data = self.load_replay_data()
        
        if len(data) < self.min_samples_for_retraining and not force:
            logger.info(f"Insufficient data for retraining: {len(data)} < {self.min_samples_for_retraining}")
            return {"status": "skipped", "reason": "insufficient_data", "samples": len(data)}
        
        # Backup current models
        self.backup_current_models()
        
        # Analyze performance
        agent_stats = self.analyze_agent_performance(data)
        model_stats = self.analyze_model_performance(data)
        
        # Update models
        agent_updated = self.update_agent_selector(agent_stats)
        model_updated = self.update_model_selector(model_stats)
        
        # Adjust exploration rates
        self.adjust_exploration_rates(agent_stats, model_stats)
        
        # Generate report
        report = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "samples_processed": len(data),
            "agent_updated": agent_updated,
            "model_updated": model_updated,
            "agent_performance": {agent: stats['avg_reward'] for agent, stats in agent_stats.items()},
            "model_performance": {model: stats['avg_reward'] for model, stats in model_stats.items()},
            "exploration_rates": {
                "agent_selector": agent_selector.exploration_rate,
                "model_selector": model_selector.exploration_rate
            }
        }
        
        # Save report
        report_path = f"logs/retraining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Retraining completed. Report saved to {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description="RL Retraining Script")
    parser.add_argument("--force", action="store_true", help="Force retraining even with insufficient data")
    parser.add_argument("--continuous", action="store_true", help="Run continuously with periodic retraining")
    parser.add_argument("--interval", type=int, default=24, help="Retraining interval in hours (for continuous mode)")
    
    args = parser.parse_args()
    
    retrainer = RLRetrainer()
    retrainer.retraining_interval_hours = args.interval
    
    if args.continuous:
        logger.info(f"Starting continuous retraining with {args.interval} hour intervals")
        while True:
            try:
                report = retrainer.retrain(force=args.force)
                logger.info(f"Retraining cycle completed: {report['status']}")
                
                # Sleep until next retraining
                sleep_seconds = args.interval * 3600
                logger.info(f"Sleeping for {args.interval} hours until next retraining")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Continuous retraining stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous retraining: {str(e)}")
                time.sleep(300)  # Sleep 5 minutes before retrying
    else:
        # Single retraining run
        report = retrainer.retrain(force=args.force)
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
