"""
Cached Multilingual Pipeline

This module provides a cached version of the multilingual pipeline
for improved performance and reduced API calls.
"""

import json
import time
import hashlib
import redis
import logging
from typing import Dict, List, Optional, Any
from .multilingual_pipeline import CompleteMultilingualPipeline

logger = logging.getLogger(__name__)

class CachedMultilingualPipeline(CompleteMultilingualPipeline):
    """
    Cached version of the multilingual pipeline
    
    Provides caching for frequently used operations to improve
    performance and reduce API calls to external services.
    """
    
    def __init__(self, 
                 api_endpoint: str = "http://localhost:8000",
                 tts_endpoint: str = "http://localhost:8001",
                 nlp_endpoint: str = "http://localhost:8002",
                 kb_endpoint: str = "http://localhost:8003",
                 redis_url: str = "redis://localhost:6379",
                 cache_ttl: int = 3600):
        """
        Initialize cached multilingual pipeline
        
        Args:
            api_endpoint: Multilingual API endpoint
            tts_endpoint: Vaani TTS service endpoint
            nlp_endpoint: Indigenous NLP service endpoint
            kb_endpoint: Knowledge Base service endpoint
            redis_url: Redis connection URL
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(api_endpoint, tts_endpoint, nlp_endpoint, kb_endpoint)
        
        self.redis_client = redis.from_url(redis_url)
        self.cache_ttl = cache_ttl
        self.cache_prefix = "multilingual_pipeline"
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _get_cache_key(self, operation: str, text: str, 
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      **kwargs) -> str:
        """Generate cache key for an operation"""
        key_data = {
            "operation": operation,
            "text": text,
            "user_id": user_id,
            "session_id": session_id,
            **kwargs
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{self.cache_prefix}:{operation}:{key_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        
        return None
    
    def _set_cache(self, cache_key: str, data: Dict[str, Any], 
                   ttl: Optional[int] = None) -> bool:
        """Set data in cache"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.cache_ttl
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Cached language detection"""
        cache_key = self._get_cache_key("detect_language", text)
        
        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.debug("Language detection cache hit")
            return cached_result
        
        # Process and cache
        result = super().detect_language(text)
        self._set_cache(cache_key, result, ttl=1800)  # 30 minutes
        
        return result
    
    def process_user_input(self, text: str, user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """Cached user input processing"""
        cache_key = self._get_cache_key(
            "process_user_input", text, user_id, session_id
        )
        
        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.debug("User input processing cache hit")
            return cached_result
        
        # Process and cache
        result = super().process_user_input(text, user_id, session_id)
        
        # Only cache successful results
        if result.get('success', True):
            self._set_cache(cache_key, result)
        
        return result
    
    def synthesize_speech_cached(self, text: str, language: str,
                                voice: Optional[str] = None) -> Dict[str, Any]:
        """Cached speech synthesis"""
        cache_key = self._get_cache_key(
            "synthesize_speech", text, voice=voice, language=language
        )
        
        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.debug("Speech synthesis cache hit")
            return cached_result
        
        # Process and cache
        result = self.tts.synthesize_speech(text, language, voice)
        
        # Only cache successful results
        if result.get('success', True):
            self._set_cache(cache_key, result, ttl=7200)  # 2 hours
        
        return result
    
    def preprocess_text_cached(self, text: str, language: str) -> Dict[str, Any]:
        """Cached text preprocessing"""
        cache_key = self._get_cache_key(
            "preprocess_text", text, language=language
        )
        
        # Try to get from cache
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            logger.debug("Text preprocessing cache hit")
            return cached_result
        
        # Process and cache
        result = self.nlp.preprocess_text(text, language)
        
        # Only cache successful results
        if result.get('success', True):
            self._set_cache(cache_key, result, ttl=3600)  # 1 hour
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            info = self.redis_client.info()
            return {
                "connected_clients": info.get('connected_clients', 0),
                "used_memory": info.get('used_memory_human', '0B'),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "total_commands_processed": info.get('total_commands_processed', 0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_cache(self, pattern: str = None) -> int:
        """Clear cache entries"""
        if not self.redis_client:
            return 0
        
        try:
            if pattern is None:
                pattern = f"{self.cache_prefix}:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    def warm_cache(self, texts: List[str], languages: List[str]) -> Dict[str, Any]:
        """
        Warm up cache with common texts and languages
        
        Args:
            texts: List of common texts to cache
            languages: List of languages to process
            
        Returns:
            Warming results
        """
        results = {
            "cached_items": 0,
            "errors": 0,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        for text in texts:
            for language in languages:
                try:
                    # Cache language detection
                    self.detect_language(text)
                    
                    # Cache preprocessing
                    self.preprocess_text_cached(text, language)
                    
                    results["cached_items"] += 2
                    
                except Exception as e:
                    logger.warning(f"Cache warming failed for {text[:50]}... in {language}: {e}")
                    results["errors"] += 1
        
        results["processing_time"] = time.time() - start_time
        
        return results


# Convenience functions
def create_cached_pipeline(redis_url: str = "redis://localhost:6379",
                          cache_ttl: int = 3600) -> CachedMultilingualPipeline:
    """Create a new cached multilingual pipeline"""
    return CachedMultilingualPipeline(redis_url=redis_url, cache_ttl=cache_ttl)
