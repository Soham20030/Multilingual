"""
Vaani TTS Integration

This module provides integration with Vaani TTS (Karthikeya) for
text-to-speech synthesis across multiple Indian languages.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Generator
import os

logger = logging.getLogger(__name__)

class VaaniTTSIntegration:
    """
    Integration class for Vaani TTS service
    
    Provides methods for text-to-speech synthesis, voice management,
    and audio streaming across multiple Indian languages.
    """
    
    def __init__(self, endpoint: str = "http://localhost:8001"):
        """
        Initialize Vaani TTS integration
        
        Args:
            endpoint: Vaani TTS service endpoint
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = 120  # 2 minutes timeout for TTS synthesis
        
    def health_check(self) -> Dict[str, Any]:
        """Check if Vaani TTS service is healthy"""
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_available_voices(self, language: str) -> List[Dict[str, Any]]:
        """
        Get available voices for a specific language
        
        Args:
            language: Language code (e.g., 'hindi', 'tamil', 'english')
            
        Returns:
            List of available voices with metadata
        """
        try:
            response = requests.get(
                f"{self.endpoint}/voices/{language}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get voices for {language}: {e}")
            return []
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        try:
            response = requests.get(f"{self.endpoint}/languages", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get supported languages: {e}")
            return ["hindi", "english"]  # Fallback
    
    def synthesize_speech(self, text: str, language: str, 
                         voice: Optional[str] = None,
                         format: str = "wav",
                         sample_rate: int = 22050,
                         speed: float = 1.0,
                         pitch: float = 1.0) -> Dict[str, Any]:
        """
        Convert text to speech using Vaani TTS
        
        Args:
            text: Text to synthesize
            language: Target language
            voice: Specific voice to use (optional)
            format: Audio format (wav, mp3, ogg)
            sample_rate: Audio sample rate
            speed: Speech speed multiplier
            pitch: Speech pitch multiplier
            
        Returns:
            Dictionary with audio metadata and URL
        """
        try:
            payload = {
                "text": text,
                "language": language,
                "format": format,
                "sample_rate": sample_rate,
                "speed": speed,
                "pitch": pitch
            }
            
            if voice:
                payload["voice"] = voice
            
            logger.info(f"Synthesizing speech for {language}: {text[:50]}...")
            
            response = requests.post(
                f"{self.endpoint}/synthesize",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"TTS synthesis completed: {result.get('duration', 0):.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "audio_url": None,
                "duration": 0
            }
    
    def stream_speech(self, text: str, language: str, 
                     chunk_size: int = 1024,
                     voice: Optional[str] = None) -> Generator[bytes, None, None]:
        """
        Stream speech synthesis for long texts
        
        Args:
            text: Text to synthesize
            language: Target language
            chunk_size: Chunk size for streaming
            voice: Specific voice to use (optional)
            
        Yields:
            Audio data chunks
        """
        try:
            payload = {
                "text": text,
                "language": language,
                "stream": True,
                "chunk_size": chunk_size
            }
            
            if voice:
                payload["voice"] = voice
            
            response = requests.post(
                f"{self.endpoint}/stream",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            yield b""  # Empty chunk on error
    
    def synthesize_to_file(self, text: str, language: str, 
                          output_path: str,
                          voice: Optional[str] = None) -> Dict[str, Any]:
        """
        Synthesize speech and save to file
        
        Args:
            text: Text to synthesize
            language: Target language
            output_path: Path to save audio file
            voice: Specific voice to use (optional)
            
        Returns:
            Dictionary with file metadata
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Synthesize speech
            result = self.synthesize_speech(text, language, voice)
            
            if not result.get('success', True):  # Handle both success field and error field
                return result
            
            # Download audio file if URL is provided
            if result.get('audio_url'):
                audio_response = requests.get(result['audio_url'], timeout=30)
                audio_response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(audio_response.content)
                
                result['file_path'] = output_path
                result['file_size'] = os.path.getsize(output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to synthesize to file: {e}")
            return {
                "error": str(e),
                "success": False,
                "file_path": None
            }
    
    def batch_synthesize(self, texts: List[str], language: str,
                        voice: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Synthesize multiple texts in batch
        
        Args:
            texts: List of texts to synthesize
            language: Target language
            voice: Specific voice to use (optional)
            
        Returns:
            List of synthesis results
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Synthesizing text {i+1}/{len(texts)}")
            
            result = self.synthesize_speech(text, language, voice)
            result['text_index'] = i
            result['text'] = text
            
            results.append(result)
            
            # Small delay between requests to avoid overwhelming the service
            time.sleep(0.1)
        
        return results
    
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Voice metadata
        """
        try:
            response = requests.get(
                f"{self.endpoint}/voice/{voice_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get voice info for {voice_id}: {e}")
            return {"error": str(e)}
    
    def test_voice_quality(self, text: str, language: str, 
                          voice: Optional[str] = None) -> Dict[str, Any]:
        """
        Test voice quality with a sample text
        
        Args:
            text: Sample text for testing
            language: Target language
            voice: Voice to test (optional)
            
        Returns:
            Quality test results
        """
        try:
            start_time = time.time()
            
            result = self.synthesize_speech(text, language, voice)
            
            end_time = time.time()
            synthesis_time = end_time - start_time
            
            return {
                "voice": voice or "default",
                "language": language,
                "synthesis_time": synthesis_time,
                "success": result.get('success', True),
                "duration": result.get('duration', 0),
                "quality_score": self._calculate_quality_score(result),
                "error": result.get('error')
            }
            
        except Exception as e:
            logger.error(f"Voice quality test failed: {e}")
            return {
                "voice": voice or "default",
                "language": language,
                "success": False,
                "error": str(e)
            }
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate a quality score for the synthesis result"""
        if not result.get('success', True):
            return 0.0
        
        # Simple quality scoring based on available metrics
        score = 1.0
        
        # Penalize if duration is too short or too long
        duration = result.get('duration', 0)
        if duration < 0.5 or duration > 30:
            score -= 0.2
        
        # Penalize if there are any warnings
        if result.get('warnings'):
            score -= 0.1
        
        return max(0.0, min(1.0, score))


# Convenience functions
def create_tts_client(endpoint: str = "http://localhost:8001") -> VaaniTTSIntegration:
    """Create a new Vaani TTS client"""
    return VaaniTTSIntegration(endpoint)


def quick_synthesize(text: str, language: str, 
                    endpoint: str = "http://localhost:8001") -> Dict[str, Any]:
    """Quick synthesis without creating a client instance"""
    client = VaaniTTSIntegration(endpoint)
    return client.synthesize_speech(text, language)
