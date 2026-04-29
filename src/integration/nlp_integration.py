"""
Indigenous NLP Integration

This module provides integration with Indigenous NLP (Nisarg) for
text preprocessing, analysis, and language-specific processing.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import re

logger = logging.getLogger(__name__)

class IndigenousNLPIntegration:
    """
    Integration class for Indigenous NLP (Nisarg) service
    
    Provides methods for text preprocessing, sentiment analysis,
    entity extraction, and language-specific processing.
    """
    
    def __init__(self, endpoint: str = "http://localhost:8002"):
        """
        Initialize Indigenous NLP integration
        
        Args:
            endpoint: Indigenous NLP service endpoint
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = 60  # 1 minute timeout for NLP processing
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Indigenous NLP service is healthy"""
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"NLP health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        try:
            response = requests.get(f"{self.endpoint}/languages", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get supported languages: {e}")
            return ["hindi", "english"]  # Fallback
    
    def preprocess_text(self, text: str, language: str,
                       normalize: bool = True,
                       clean: bool = True,
                       tokenize: bool = False) -> Dict[str, Any]:
        """
        Preprocess text using Indigenous NLP
        
        Args:
            text: Input text to preprocess
            language: Language code
            normalize: Whether to normalize Unicode
            clean: Whether to clean the text
            tokenize: Whether to tokenize the text
            
        Returns:
            Dictionary with preprocessing results
        """
        try:
            payload = {
                "text": text,
                "language": language,
                "normalize": normalize,
                "clean": clean,
                "tokenize": tokenize
            }
            
            logger.info(f"Preprocessing {language} text: {text[:50]}...")
            
            response = requests.post(
                f"{self.endpoint}/preprocess",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"NLP preprocessing completed")
            
            return result
            
        except Exception as e:
            logger.error(f"NLP preprocessing failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "processed_text": text,
                "tokens": []
            }
    
    def analyze_sentiment(self, text: str, language: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Sentiment analysis results
        """
        try:
            payload = {
                "text": text,
                "language": language
            }
            
            response = requests.post(
                f"{self.endpoint}/sentiment",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "sentiment": "neutral",
                "confidence": 0.0
            }
    
    def extract_entities(self, text: str, language: str) -> Dict[str, Any]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Entity extraction results
        """
        try:
            payload = {
                "text": text,
                "language": language
            }
            
            response = requests.post(
                f"{self.endpoint}/entities",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "entities": []
            }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language of text using Indigenous NLP
        
        Args:
            text: Text to analyze
            
        Returns:
            Language detection results
        """
        try:
            payload = {"text": text}
            
            response = requests.post(
                f"{self.endpoint}/detect-language",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "language": "unknown",
                "confidence": 0.0
            }
    
    def transliterate(self, text: str, source_lang: str, 
                     target_lang: str) -> Dict[str, Any]:
        """
        Transliterate text between languages
        
        Args:
            text: Text to transliterate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Transliteration results
        """
        try:
            payload = {
                "text": text,
                "source_language": source_lang,
                "target_language": target_lang
            }
            
            response = requests.post(
                f"{self.endpoint}/transliterate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Transliteration failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "transliterated_text": text
            }
    
    def analyze_grammar(self, text: str, language: str) -> Dict[str, Any]:
        """
        Analyze grammar of text
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Grammar analysis results
        """
        try:
            payload = {
                "text": text,
                "language": language
            }
            
            response = requests.post(
                f"{self.endpoint}/grammar",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Grammar analysis failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "grammar_score": 0.0,
                "errors": []
            }
    
    def extract_keywords(self, text: str, language: str, 
                        max_keywords: int = 10) -> Dict[str, Any]:
        """
        Extract keywords from text
        
        Args:
            text: Text to analyze
            language: Language code
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            Keyword extraction results
        """
        try:
            payload = {
                "text": text,
                "language": language,
                "max_keywords": max_keywords
            }
            
            response = requests.post(
                f"{self.endpoint}/keywords",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "keywords": []
            }
    
    def summarize_text(self, text: str, language: str, 
                      max_sentences: int = 3) -> Dict[str, Any]:
        """
        Summarize text
        
        Args:
            text: Text to summarize
            language: Language code
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Text summarization results
        """
        try:
            payload = {
                "text": text,
                "language": language,
                "max_sentences": max_sentences
            }
            
            response = requests.post(
                f"{self.endpoint}/summarize",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "summary": text
            }
    
    def batch_process(self, texts: List[str], language: str,
                     operations: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of texts to process
            language: Language code
            operations: List of operations to perform
            
        Returns:
            List of processing results
        """
        if operations is None:
            operations = ["preprocess", "sentiment", "entities"]
        
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            
            result = {"text": text, "operations": {}}
            
            for operation in operations:
                try:
                    if operation == "preprocess":
                        op_result = self.preprocess_text(text, language)
                    elif operation == "sentiment":
                        op_result = self.analyze_sentiment(text, language)
                    elif operation == "entities":
                        op_result = self.extract_entities(text, language)
                    elif operation == "keywords":
                        op_result = self.extract_keywords(text, language)
                    elif operation == "grammar":
                        op_result = self.analyze_grammar(text, language)
                    else:
                        op_result = {"error": f"Unknown operation: {operation}"}
                    
                    result["operations"][operation] = op_result
                    
                except Exception as e:
                    result["operations"][operation] = {"error": str(e)}
            
            results.append(result)
            
            # Small delay between requests
            time.sleep(0.1)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics from the service"""
        try:
            response = requests.get(f"{self.endpoint}/stats", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get processing stats: {e}")
            return {"error": str(e)}


# Convenience functions
def create_nlp_client(endpoint: str = "http://localhost:8002") -> IndigenousNLPIntegration:
    """Create a new Indigenous NLP client"""
    return IndigenousNLPIntegration(endpoint)


def quick_preprocess(text: str, language: str,
                    endpoint: str = "http://localhost:8002") -> Dict[str, Any]:
    """Quick preprocessing without creating a client instance"""
    client = IndigenousNLPIntegration(endpoint)
    return client.preprocess_text(text, language)
