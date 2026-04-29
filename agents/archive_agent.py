import logging
import time
import PyPDF2
from typing import Dict, Any
import uuid
from langchain_groq import ChatGroq
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from config.settings import MODEL_CONFIG
import os

logger = get_logger(__name__)

class ArchiveAgent:
    """Agent for processing PDF archives using Groq."""
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"
        )
        self.model_config = MODEL_CONFIG.get("edumentor_agent", {})

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                logger.info(f"Extracted {len(text)} characters from PDF: {pdf_path}")
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""

    def process_pdf(self, text: str, task_id: str, retries: int = 3) -> Dict[str, Any]:
        """Summarize PDF text using Groq's API with retry logic."""
        start_time = time.time()

        for attempt in range(retries):
            try:
                logger.info(f"Processing PDF (attempt {attempt + 1}/{retries}) for task {task_id}")

                # Limit text for API (Groq has token limits)
                limited_text = text[:2000]
                prompt = f"Summarize the following text in 50-100 words: {limited_text}"

                response = self.llm.invoke(prompt)
                summary = response.content

                processing_time = time.time() - start_time

                logger.info(f"PDF processing successful for task {task_id} in {processing_time:.2f}s")
                logger.debug(f"PDF text length: {len(text)}, Summary: {summary[:100]}...")

                return {
                    "result": summary,
                    "model": "archive_agent",
                    "tokens_used": len(text.split()) + len(summary.split()),  # Approximate
                    "cost_estimate": 0.0,  # Free tier for development
                    "status": 200,
                    "keywords": ["pdf", "summary"],
                    "processing_time": processing_time,
                    "inference_time": processing_time,
                    "attempts": attempt + 1,
                    "text_length": len(text),
                    "text_truncated": len(text) > 2000
                }

            except Exception as e:
                processing_time = time.time() - start_time
                logger.warning(f"PDF processing attempt {attempt + 1}/{retries} failed for task {task_id}: {str(e)}")

                if attempt < retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    logger.error(f"PDF processing failed after {retries} attempts for task {task_id}: {str(e)}")
                    return {
                        "error": f"PDF processing failed after {retries} attempts: {str(e)}",
                        "status": 500,
                        "keywords": [],
                        "processing_time": processing_time,
                        "attempts": retries,
                        "text_length": len(text) if text else 0
                    }

    def run(self, input_path: str, live_feed: str = "", model: str = "edumentor_agent", input_type: str = "pdf", task_id: str = None) -> Dict[str, Any]:
        task_id = task_id or str(uuid.uuid4())
        logger.info(f"ArchiveAgent starting task {task_id} with input_type: {input_type}, model: {model}")
        logger.debug(f"Task {task_id} - Processing PDF file: {input_path}")

        # Extract PDF content
        content = self.extract_pdf_text(input_path)
        if not content:
            logger.error(f"Task {task_id} - Failed to extract content from PDF: {input_path}")
            output = {
                "error": "Failed to extract PDF content",
                "status": 400,
                "keywords": [],
                "agent": "archive_agent",
                "input_type": input_type,
                "file_path": input_path
            }
            reward = get_reward_from_output(output, task_id)
            replay_buffer.add_run(task_id, input_path, output, "archive_agent", model, reward)
            return output

        logger.info(f"Task {task_id} - Extracted {len(content)} characters from PDF")

        # Process the extracted content
        result = self.process_pdf(content, task_id)

        # Add metadata to result
        result['agent'] = 'archive_agent'
        result['input_type'] = input_type
        result['file_path'] = input_path
        result['content_length'] = len(content)

        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, input_path, result, "archive_agent", model, reward)

        logger.info(f"ArchiveAgent completed task {task_id} with status: {result.get('status', 'unknown')}")
        return result

if __name__ == "__main__":
    agent = ArchiveAgent()
    test_input = "test.pdf"
    result = agent.run(test_input, input_type="pdf")
    print(result)