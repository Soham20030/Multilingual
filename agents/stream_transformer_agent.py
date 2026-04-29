import importlib
import base64
from typing import Dict, Any
import uuid
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from agents.agent_registry import agent_registry

logger = get_logger(__name__)

class StreamTransformerAgent:
    """Delegates to specialized agents based on input type."""
    def __init__(self):
        self.agent_map = {
            "pdf": "archive_agent",
            "image": "image_agent",
            "audio": "audio_agent",
            "text": "text_agent"
        }

    def run(self, input_path: str, live_feed: str = "", model: str = "edumentor_agent", input_type: str = "text", task_id: str = None) -> Dict[str, Any]:
        task_id = task_id or str(uuid.uuid4())
        logger.info(f"StreamTransformerAgent processing task {task_id}, input: {input_path}, type: {input_type}")
        
        agent_id = self.agent_map.get(input_type, "text_agent")
        agent_config = agent_registry.get_agent_config(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found for input type {input_type}")
            output = {"error": f"Agent {agent_id} not found", "status": 400, "keywords": []}
            reward = get_reward_from_output(output, task_id)
            replay_buffer.add_run(task_id, input_path, output, agent_id, model, reward)
            return output

        try:
            module_path = agent_config['module_path']
            class_name = agent_config['class_name']
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            agent = agent_class()
            result = agent.run(input_path, live_feed, model, input_type, task_id)
            reward = get_reward_from_output(result, task_id)
            replay_buffer.add_run(task_id, input_path, result, agent_id, model, reward)
            return result
        except Exception as e:
            logger.error(f"Error processing task with agent {agent_id}: {str(e)}")
            output = {"error": f"Task processing failed: {str(e)}", "status": 500, "keywords": []}
            reward = get_reward_from_output(output, task_id)
            replay_buffer.add_run(task_id, input_path, output, agent_id, model, reward)
            return output

if __name__ == "__main__":
    agent = StreamTransformerAgent()
    test_input = "test.pdf"
    test_feed = "Live feed: AI advancements in 2025."
    result = agent.run(test_input, test_feed, "edumentor_agent", "pdf")
    print(result)

# import PyPDF2
# import base64
# from typing import Dict, Any
# from integration.llm_router import TransformerAdapter
# from utils.logger import get_logger
# from reinforcement.reward_functions import get_reward_from_output
# from reinforcement.replay_buffer import replay_buffer

# logger = get_logger(__name__)

# class StreamTransformerAgent:
#     def __init__(self):
#         self.llm_adapter = TransformerAdapter()

#     def extract_pdf_text(self, pdf_path: str) -> str:
#         """Extract text from a PDF file."""
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 text = ""
#                 for page in reader.pages:
#                     text += page.extract_text() or ""
#                 logger.info(f"Extracted {len(text)} characters from PDF: {pdf_path}")
#                 return text.strip()
#         except Exception as e:
#             logger.error(f"Error extracting PDF text: {str(e)}")
#             return ""

#     def process_image(self, image_path: str) -> str:
#         """Process an image file (stub for Vision API)."""
#         try:
#             with open(image_path, 'rb') as image_file:
#                 encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
#             logger.info(f"Processed image: {image_path}")
#             return f"Image content: {encoded_image[:50]}..."  # Stub for Vision API
#         except Exception as e:
#             logger.error(f"Error processing image: {str(e)}")
#             return ""

#     def process_audio(self, audio_path: str) -> str:
#         """Process an audio file (stub for Whisper API)."""
#         logger.info(f"Processing audio: {audio_path}")
#         return "Audio transcript: Sample transcript"  # Stub for Whisper API

#     def run(self, input_path: str, live_feed: str = "", model: str = "edumentor_agent", input_type: str = "text", task_id: str = None) -> Dict[str, Any]:
#         task_id = task_id or str(uuid.uuid4())
#         logger.info(f"Processing task with model: {model}, Input: {input_path}, Type: {input_type}, Live Feed: {len(live_feed)} chars")
        
#         content = ""
#         if input_type == "pdf":
#             content = self.extract_pdf_text(input_path)
#         elif input_type == "image":
#             content = self.process_image(input_path)
#         elif input_type == "audio":
#             content = self.process_audio(input_path)
#         else:
#             content = input_path  # Treat as text

#         if not content:
#             logger.error(f"No content extracted for input type {input_type}")
#             output = {"error": f"Failed to extract {input_type} content", "status": 400}
#             reward = get_reward_from_output(output, task_id)
#             replay_buffer.add_run(task_id, input_path, output, model, model, reward)
#             return output

#         merged_content = f"Summarize this: {content} {live_feed}".strip()
#         try:
#             result = self.llm_adapter.run_with_model(model, merged_content, task_id=task_id)
#             logger.info(f"Task processed successfully: {result}")
#             reward = get_reward_from_output(result, task_id)
#             replay_buffer.add_run(task_id, input_path, result, model, model, reward)
#             return result
#         except Exception as e:
#             logger.error(f"Error processing task: {str(e)}")
#             output = {"error": f"Task processing failed: {str(e)}", "status": 500}
#             reward = get_reward_from_output(output, task_id)
#             replay_buffer.add_run(task_id, input_path, output, model, model, reward)
#             return output

# if __name__ == "__main__":
#     agent = StreamTransformerAgent()
#     test_input = "test.pdf"  # Replace with real input path
#     test_feed = "Live feed: AI advancements in 2025."
#     result = agent.run(test_input, test_feed, "edumentor_agent", "pdf")
#     print(result)


