"""
Complete Multilingual Pipeline Integration

This module provides the main integration class for connecting
the Multilingual Tokenization Model with Indigenous NLP and Vaani TTS.
"""

import asyncio
import aiohttp
import requests
import json
import time
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CompleteMultilingualPipeline:
    """
    Complete integration pipeline for multilingual text processing
    
    This class provides end-to-end integration:
    User Input → Language Detection → NLP Processing → Response Generation → TTS
    """
    
    def __init__(self, 
                 api_endpoint: str = "http://localhost:8000",
                 tts_endpoint: str = "http://localhost:8001", 
                 nlp_endpoint: str = "http://localhost:8002",
                 kb_endpoint: str = "http://localhost:8003"):
        """
        Initialize the complete multilingual pipeline
        
        Args:
            api_endpoint: Multilingual API endpoint
            tts_endpoint: Vaani TTS service endpoint
            nlp_endpoint: Indigenous NLP service endpoint
            kb_endpoint: Knowledge Base service endpoint
        """
        self.api_endpoint = api_endpoint
        self.tts_endpoint = tts_endpoint
        self.nlp_endpoint = nlp_endpoint
        self.kb_endpoint = kb_endpoint
        
        # Initialize sub-integrations
        self.tts = VaaniTTSIntegration(tts_endpoint)
        self.nlp = IndigenousNLPIntegration(nlp_endpoint)
        
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of input text"""
        try:
            response = requests.post(
                f"{self.api_endpoint}/language-detect",
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {"language": "english", "confidence": 0.0}
    
    def process_user_input(self, text: str, user_id: Optional[str] = None, 
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: Input → Processing → Response → Audio
        
        Args:
            text: Input text from user
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing processed response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Language detection
            lang_result = self.detect_language(text)
            detected_lang = lang_result['language']
            confidence = lang_result['confidence']
            
            logger.info(f"Detected language: {detected_lang} (confidence: {confidence})")
            
            # Step 2: Indigenous NLP preprocessing
            try:
                preprocessed = self.nlp.preprocess_text(text, detected_lang)
                processed_text = preprocessed.get('processed_text', text)
            except Exception as e:
                logger.warning(f"NLP preprocessing failed: {e}, using original text")
                processed_text = text
            
            # Step 3: Knowledge Base query and response generation
            try:
                kb_response = requests.post(
                    f"{self.api_endpoint}/multilingual-conversation",
                    json={
                        "text": processed_text,
                        "language": detected_lang,
                        "user_id": user_id,
                        "session_id": session_id,
                        "generate_response": True
                    },
                    timeout=30
                )
                kb_response.raise_for_status()
                kb_data = kb_response.json()
                
                # Use generated response if available, otherwise use KB answer
                if kb_data.get('generated_response'):
                    final_text = kb_data['generated_response']
                else:
                    final_text = kb_data.get('kb_answer', processed_text)
                    
            except Exception as e:
                logger.warning(f"KB integration failed: {e}, using processed text")
                final_text = processed_text
            
            # Step 4: Synthesize speech
            try:
                audio_result = self.tts.synthesize_speech(final_text, detected_lang)
                audio_url = audio_result.get('audio_url')
                duration = audio_result.get('duration', 0)
            except Exception as e:
                logger.warning(f"TTS synthesis failed: {e}")
                audio_url = None
                duration = 0
            
            processing_time = time.time() - start_time
            
            return {
                "user_input": text,
                "detected_language": detected_lang,
                "language_confidence": confidence,
                "text_response": final_text,
                "audio_url": audio_url,
                "audio_duration": duration,
                "session_id": session_id,
                "user_id": user_id,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                "user_input": text,
                "error": str(e),
                "success": False,
                "processing_time": time.time() - start_time
            }
    
    def handle_conversation(self, messages: List[Dict[str, str]], 
                          user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Handle multi-turn conversations
        
        Args:
            messages: List of message dictionaries with 'text' key
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        for i, message in enumerate(messages):
            logger.info(f"Processing message {i+1}/{len(messages)}")
            
            result = self.process_user_input(
                message['text'],
                user_id=user_id,
                session_id=session_id
            )
            
            responses.append(result)
            
            # Add small delay between messages for better user experience
            time.sleep(0.1)
        
        return responses
    
    def test_language_switching(self, texts: List[str], 
                               user_id: str = "test_user", 
                               session_id: str = "test_session") -> Dict[str, Any]:
        """
        Test mid-conversation language switching capability
        
        Args:
            texts: List of texts in different languages
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with switching test results
        """
        results = []
        detected_languages = set()
        
        for i, text in enumerate(texts):
            result = self.process_user_input(text, user_id, session_id)
            results.append(result)
            detected_languages.add(result.get('detected_language', 'unknown'))
        
        return {
            "test_results": results,
            "detected_languages": list(detected_languages),
            "switching_successful": len(detected_languages) > 1,
            "total_messages": len(texts),
            "unique_languages": len(detected_languages)
        }


class AsyncMultilingualPipeline:
    """
    Async version of the multilingual pipeline for better performance
    """
    
    def __init__(self, 
                 api_endpoint: str = "http://localhost:8000",
                 tts_endpoint: str = "http://localhost:8001",
                 nlp_endpoint: str = "http://localhost:8002",
                 kb_endpoint: str = "http://localhost:8003"):
        self.api_endpoint = api_endpoint
        self.tts_endpoint = tts_endpoint
        self.nlp_endpoint = nlp_endpoint
        self.kb_endpoint = kb_endpoint
        self.pipeline = CompleteMultilingualPipeline(
            api_endpoint, tts_endpoint, nlp_endpoint, kb_endpoint
        )
    
    async def process_user_input_async(self, text: str, 
                                     user_id: Optional[str] = None,
                                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """Async version of process_user_input"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.pipeline.process_user_input,
            text, user_id, session_id
        )
        return result
    
    async def batch_process(self, texts: List[str], 
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process multiple texts concurrently"""
        tasks = [
            self.process_user_input_async(text, user_id, session_id)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    async def handle_conversation_async(self, messages: List[Dict[str, str]], 
                                      user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Async version of handle_conversation"""
        tasks = [
            self.process_user_input_async(
                message['text'], user_id, session_id
            )
            for message in messages
        ]
        return await asyncio.gather(*tasks)


# Convenience functions for easy integration
def create_pipeline(api_endpoint: str = "http://localhost:8000",
                   tts_endpoint: str = "http://localhost:8001",
                   nlp_endpoint: str = "http://localhost:8002",
                   kb_endpoint: str = "http://localhost:8003") -> CompleteMultilingualPipeline:
    """Create a new multilingual pipeline instance"""
    return CompleteMultilingualPipeline(api_endpoint, tts_endpoint, nlp_endpoint, kb_endpoint)


def create_async_pipeline(api_endpoint: str = "http://localhost:8000",
                         tts_endpoint: str = "http://localhost:8001",
                         nlp_endpoint: str = "http://localhost:8002",
                         kb_endpoint: str = "http://localhost:8003") -> AsyncMultilingualPipeline:
    """Create a new async multilingual pipeline instance"""
    return AsyncMultilingualPipeline(api_endpoint, tts_endpoint, nlp_endpoint, kb_endpoint)
