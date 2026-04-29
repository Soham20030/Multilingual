import logging
from typing import Dict, Any
import uuid
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from config.settings import MODEL_CONFIG

logger = get_logger(__name__)

class ImageAgent:
    """Agent for processing image inputs using BLIP model."""
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model_config = MODEL_CONFIG.get("edumentor_agent", {})

    def process_image(self, image_path: str, task_id: str) -> Dict[str, Any]:
        """Generate caption for an image using BLIP."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated caption for {image_path}: {caption}")
            return {
                "result": caption,
                "model": "image_agent",
                "tokens_used": len(caption.split()),  # Approximate tokens
                "cost_estimate": 0.0,  # Local model, no cost
                "status": 200,
                "keywords": ["image", "caption"]
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {"error": f"Image processing failed: {str(e)}", "status": 500, "keywords": []}

    def run(self, input_path: str, live_feed: str = "", model: str = "edumentor_agent", input_type: str = "image", task_id: str = None) -> Dict[str, Any]:
        task_id = task_id or str(uuid.uuid4())
        logger.info(f"ImageAgent processing task {task_id}, input: {input_path}")
        result = self.process_image(input_path, task_id)
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, input_path, result, "image_agent", model, reward)
        return result

if __name__ == "__main__":
    agent = ImageAgent()
    test_input = "test_image.jpg"
    result = agent.run(test_input, input_type="image")
    print(result)