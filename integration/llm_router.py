import requests
from typing import Dict, Any
from config.settings import MODEL_CONFIG, TIMEOUT_CONFIG
from utils.logger import get_logger
from reinforcement.rl_context import rl_context
from reinforcement.model_selector import model_selector
from reinforcement.reward_functions import get_reward_from_output
from utils.calculator import Calculator
from langchain_groq import ChatGroq
import os
import uuid

logger = get_logger(__name__)

class TransformerAdapter:
    def __init__(self):
        self.model_config = MODEL_CONFIG
        self.calculator = Calculator()
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )
        # Fallback hierarchy for models
        self.fallback_hierarchy = {
            'llama': ['edumentor_agent', 'vedas_agent', 'wellness_agent'],
            'edumentor_agent': ['vedas_agent', 'wellness_agent', 'llama'],
            'vedas_agent': ['edumentor_agent', 'wellness_agent', 'llama'],
            'wellness_agent': ['edumentor_agent', 'vedas_agent', 'llama']
        }

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """Estimate cost based on tokens and model (mocked for now)."""
        rates = {
            'llama': 0.0001,
            'vedas_agent': 0.0002,
            'edumentor_agent': 0.0002,
            'wellness_agent': 0.0002
        }
        return tokens_used * rates.get(model, 0.0001)

    def run_with_model(self, model: str, query: str, live_feed: str = "", task_id: str = None, retry_count: int = 0, max_retries: int = 2) -> Dict[str, Any]:
        logger.info(f"Routing query to model: {model}, query: {query[:50]}... (retry: {retry_count})")
        task_id = task_id or str(uuid.uuid4())
        original_model = model

        # Check if query is a calculation
        if any(op in query for op in ['+', '-', '*', '/']):
            calc_result = self.calculator.evaluate(query)
            if calc_result.get('status') == 200:
                logger.info(f"Calculation result: {calc_result['result']}")
                reward = get_reward_from_output(calc_result, task_id)
                model_selector.update_history(task_id, model, reward)
                return calc_result

        try:
            # Use RL model selector if available (only on first attempt)
            if retry_count == 0:
                selected_model = model_selector.select_model({"task_id": task_id, "model": model, "query": query})
                if selected_model != model:
                    logger.info(f"RL model selector chose {selected_model} over {model}")
                    model = selected_model

            result = self._execute_model(model, query, live_feed, task_id)

            # Add retry metadata
            result['retry_count'] = retry_count
            result['fallback_used'] = retry_count > 0
            result['final_model'] = model
            result['original_model'] = original_model

            reward = get_reward_from_output(result, task_id)
            model_selector.update_history(task_id, model, reward)
            return result

        except (requests.exceptions.RequestException, ValueError, Exception) as e:
            logger.error(f"Error running model {model} (attempt {retry_count + 1}): {str(e)}")

            # Try fallback if retries available
            if retry_count < max_retries:
                fallback_models = self.fallback_hierarchy.get(model, ['edumentor_agent'])
                if fallback_models:
                    fallback_model = fallback_models[min(retry_count, len(fallback_models) - 1)]
                    logger.info(f"Attempting fallback from {model} to {fallback_model}")

                    # Log fallback action
                    rl_context.log_action(task_id, "llm_router", fallback_model, "fallback",
                                        {"original_model": original_model, "failed_model": model,
                                         "error": str(e), "retry_count": retry_count + 1})

                    return self.run_with_model(fallback_model, query, live_feed, task_id, retry_count + 1, max_retries)

            # All retries exhausted
            error_output = {
                'error': f"All models failed. Last error: {str(e)}",
                'status': 500,
                'keywords': [],
                'retry_count': retry_count,
                'fallback_used': retry_count > 0,
                'final_model': model,
                'original_model': original_model
            }
            reward = get_reward_from_output(error_output, task_id)
            model_selector.update_history(task_id, model, reward)
            return error_output

    def _execute_model(self, model: str, query: str, live_feed: str, task_id: str) -> Dict[str, Any]:
        """Execute a specific model with proper error handling."""
        if model in ['llama', 'llama_summarization_agent']:
            response = self.llm.invoke(query)
            result = response.content
            tokens_used = len(query.split()) + len(result.split())  # Approximate
            cost_estimate = self.estimate_cost(tokens_used, model)
            return {
                'result': result,
                'model': model,
                'tokens_used': tokens_used,
                'cost_estimate': cost_estimate,
                'status': 200,
                'keywords': []
            }
        elif model in ['vedas_agent', 'edumentor_agent', 'wellness_agent']:
            config = self.model_config.get(model, {})
            endpoint = config.get('endpoint')
            headers = config.get('headers', {'Content-Type': 'application/json'})
            if config.get('api_key'):
                headers['Authorization'] = f"Bearer {config['api_key']}"
            payload = {'query': f"{query} {live_feed}".strip(), 'user_id': 'bhiv_core', 'task_id': task_id}
            timeout = TIMEOUT_CONFIG.get('llm_timeout', 120)
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return {
                'result': result.get('response', 'No response generated'),
                'model': model,
                'tokens_used': 0,  # simple_api.py doesn't return token usage
                'cost_estimate': 0.0,
                'sources': result.get('sources', []),
                'status': 200,
                'keywords': []
            }
        else:
            raise ValueError(f"Unsupported model: {model}")

if __name__ == "__main__":
    adapter = TransformerAdapter()
    test_query = "5 + 3"
    test_model = "edumentor_agent"
    result = adapter.run_with_model(test_model, test_query)
    print(result)

# import requests
# from typing import Dict, Any
# from config.settings import MODEL_CONFIG
# from utils.logger import get_logger
# from reinforcement.rl_context import rl_context
# from reinforcement.model_selector import model_selector
# from reinforcement.reward_functions import get_reward_from_output

# logger = get_logger(__name__)

# class TransformerAdapter:
#     def __init__(self):
#         self.model_config = MODEL_CONFIG

#     def estimate_cost(self, tokens_used: int, model: str) -> float:
#         """Estimate cost based on tokens and model (mocked for now)."""
#         rates = {
#             'llama': 0.0001,
#             'vedas_agent': 0.0002,
#             'edumentor_agent': 0.0002,
#             'wellness_agent': 0.0002
#         }
#         return tokens_used * rates.get(model, 0.0001)

#     def run_with_model(self, model: str, query: str, live_feed: str = "", task_id: str = None) -> Dict[str, Any]:
#         logger.info(f"Routing query to model: {model}, query: {query[:50]}...")
#         task_id = task_id or str(uuid.uuid4())
#         try:
#             # Use RL model selector if available
#             selected_model = model_selector.select_model({"task_id": task_id, "model": model, "query": query})
#             if selected_model != model:
#                 logger.info(f"RL model selector chose {selected_model} over {model}")
#                 model = selected_model

#             if model in ['llama', 'llama_summarization_agent']:
#                 config = self.model_config.get('llama', {})
#                 api_url = config.get('api_url', 'http://localhost:1234/v1/chat/completions')
#                 headers = {
#                     'Content-Type': 'application/json',
#                     'Authorization': f"Bearer {config.get('api_key')}" if config.get('api_key') else None
#                 }
#                 headers = {k: v for k, v in headers.items() if v is not None}
#                 payload = {
#                     'model': config.get('model_name', 'llama-3.1-8b-instruct'),
#                     'messages': [{'role': 'user', 'content': query}]
#                 }
#                 response = requests.post(api_url, json=payload, headers=headers, timeout=15)
#                 response.raise_for_status()
#                 result = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No summary generated')
#                 tokens_used = response.json().get('usage', {}).get('total_tokens', 0)
#                 cost_estimate = self.estimate_cost(tokens_used, model)
#                 output = {
#                     'result': result,
#                     'model': model,
#                     'tokens_used': tokens_used,
#                     'cost_estimate': cost_estimate,
#                     'status': 200,
#                     'keywords': []  # Add default keywords for reward calculation
#                 }
#                 reward = get_reward_from_output(output, task_id)
#                 model_selector.update_history(task_id, model, reward)
#                 return output
#             elif model in ['vedas_agent', 'edumentor_agent', 'wellness_agent']:
#                 config = self.model_config.get(model, {})
#                 endpoint = config.get('endpoint')
#                 headers = config.get('headers', {'Content-Type': 'application/json'})
#                 if config.get('api_key'):
#                     headers['Authorization'] = f"Bearer {config['api_key']}"
#                 payload = {'query': f"{query} {live_feed}".strip(), 'user_id': 'bhiv_core', 'task_id': task_id}
#                 response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
#                 response.raise_for_status()
#                 result = response.json()
#                 output = {
#                     'result': result.get('response', 'No response generated'),
#                     'model': model,
#                     'tokens_used': 0,  # simple_api.py doesn't return token usage
#                     'cost_estimate': 0.0,
#                     'sources': result.get('sources', []),
#                     'status': 200,
#                     'keywords': []  # Add default keywords for reward calculation
#                 }
#                 reward = get_reward_from_output(output, task_id)
#                 model_selector.update_history(task_id, model, reward)
#                 return output
#             else:
#                 raise ValueError(f"Unsupported model: {model}")
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error running model {model}: {str(e)}")
#             output = {'error': f"Model execution failed: {str(e)}", 'status': 500, 'keywords': []}
#             reward = get_reward_from_output(output, task_id)
#             model_selector.update_history(task_id, model, reward)
#             return output
#         except Exception as e:
#             logger.error(f"Unexpected error in model {model}: {str(e)}")
#             output = {'error': f"Unexpected error: {str(e)}", 'status': 500, 'keywords': []}
#             reward = get_reward_from_output(output, task_id)
#             model_selector.update_history(task_id, model, reward)
#             return output

# if __name__ == "__main__":
#     adapter = TransformerAdapter()
#     test_query = "Summarize this text: AI advancements in 2025."
#     test_model = "edumentor_agent"
#     result = adapter.run_with_model(test_model, test_query)
#     print(result)

