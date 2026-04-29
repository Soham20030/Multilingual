"""
Output Validation & Retry Logic
=================================

Validates generated outputs and retries with adjusted parameters if quality is poor.
"""

import logging
from typing import Dict, Tuple, Optional, Callable
from src.utils.post_processing import process_output

logger = logging.getLogger(__name__)


class OutputValidator:
    """
    Validates generated outputs and implements retry logic
    """
    
    def __init__(self, min_quality_threshold: float = 0.6):
        """
        Initialize validator
        
        Args:
            min_quality_threshold: Minimum quality score to accept (0.0 to 1.0)
        """
        self.min_quality_threshold = min_quality_threshold
        self.max_retries = 2
        logger.info(f"OutputValidator initialized with threshold: {min_quality_threshold}")
    
    def validate_and_retry(
        self,
        generation_func: Callable,
        text: str,
        language: str,
        max_length: Optional[int] = None,
        **generation_kwargs
    ) -> Tuple[str, Dict]:
        """
        Validate output and retry if quality is poor
        
        Args:
            generation_func: Function that generates text (should return raw text)
            text: Input text/prompt
            language: Target language
            max_length: Maximum output length
            **generation_kwargs: Additional kwargs for generation function
        
        Returns:
            Tuple of (best_output, metadata)
        """
        attempts = []
        best_output = None
        best_quality = 0.0
        
        for attempt_num in range(self.max_retries + 1):
            logger.info(f"Generation attempt {attempt_num + 1}/{self.max_retries + 1}")
            
            # Adjust parameters for retry attempts
            if attempt_num > 0:
                generation_kwargs = self._adjust_parameters_for_retry(
                    generation_kwargs, 
                    attempt_num,
                    previous_quality=best_quality
                )
                logger.info(f"Retry with adjusted params: temp={generation_kwargs.get('temperature', 'N/A')}")
            
            try:
                # Generate output
                raw_output = generation_func(text, **generation_kwargs)
                
                if not raw_output:
                    logger.warning(f"Attempt {attempt_num + 1}: Empty output")
                    continue
                
                # Post-process and score
                processed_output, post_metadata = process_output(
                    raw_output,
                    language,
                    max_length
                )
                
                quality_score = post_metadata.get('quality_score', 0.0)
                
                # Validate
                validation_result = self._validate_output(
                    processed_output,
                    language,
                    quality_score
                )
                
                attempts.append({
                    'attempt': attempt_num + 1,
                    'quality_score': quality_score,
                    'validation_passed': validation_result['passed'],
                    'issues': validation_result['issues'],
                    'output_length': len(processed_output),
                    'params': generation_kwargs.copy()
                })
                
                logger.info(
                    f"Attempt {attempt_num + 1}: Quality={quality_score:.2f}, "
                    f"Passed={validation_result['passed']}, Issues={validation_result['issues']}"
                )
                
                # Track best output
                if quality_score > best_quality:
                    best_quality = quality_score
                    best_output = processed_output
                
                # If quality is good enough, stop trying
                if validation_result['passed'] and quality_score >= self.min_quality_threshold:
                    logger.info(f"✅ Acceptable quality achieved on attempt {attempt_num + 1}")
                    break
                
            except Exception as e:
                logger.error(f"Attempt {attempt_num + 1} failed: {e}")
                attempts.append({
                    'attempt': attempt_num + 1,
                    'error': str(e),
                    'quality_score': 0.0,
                    'validation_passed': False
                })
        
        # Return best attempt
        metadata = {
            'total_attempts': len(attempts),
            'best_quality': best_quality,
            'attempts': attempts,
            'used_retry': len(attempts) > 1,
            'final_validation_passed': best_quality >= self.min_quality_threshold
        }
        
        if best_output:
            logger.info(
                f"Returning best output: Quality={best_quality:.2f}, "
                f"Attempts={len(attempts)}, Passed={metadata['final_validation_passed']}"
            )
        else:
            logger.warning("No valid output generated after all attempts")
        
        return best_output or "", metadata
    
    def _validate_output(self, text: str, language: str, quality_score: float) -> Dict:
        """
        Validate output quality
        
        Returns:
            Dict with 'passed' (bool) and 'issues' (list)
        """
        issues = []
        
        # Check 1: Minimum length
        if len(text) < 30:
            issues.append("too_short")
        
        # Check 2: Quality score
        if quality_score < self.min_quality_threshold:
            issues.append("low_quality_score")
        
        # Check 3: Too much repetition
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                issues.append("high_repetition")
        
        # Check 4: Check for wrong language indicators (basic)
        if language != "english":
            # Count English words (very basic check)
            english_words = sum(1 for word in words if word.isascii() and len(word) > 3)
            if english_words > len(words) * 0.5:  # More than 50% English
                issues.append("wrong_language")
        
        # Check 5: Check for empty after processing
        if not text.strip():
            issues.append("empty_output")
        
        passed = len(issues) == 0
        
        return {
            'passed': passed,
            'issues': issues,
            'quality_score': quality_score
        }
    
    def _adjust_parameters_for_retry(
        self,
        params: Dict,
        attempt_num: int,
        previous_quality: float
    ) -> Dict:
        """
        Adjust generation parameters for retry attempts
        
        Strategy:
        - Attempt 1 (retry 1): Lower temperature, more focused
        - Attempt 2 (retry 2): Even lower temperature, very focused
        """
        adjusted = params.copy()
        
        if attempt_num == 1:
            # First retry: Be more conservative
            adjusted['temperature'] = max(0.4, params.get('temperature', 0.6) - 0.2)
            adjusted['top_p'] = max(0.75, params.get('top_p', 0.85) - 0.1)
            adjusted['top_k'] = max(30, params.get('top_k', 40) - 10)
            logger.debug(f"Retry 1: Lowered temperature to {adjusted['temperature']}")
        
        elif attempt_num == 2:
            # Second retry: Very conservative, greedy-ish
            adjusted['temperature'] = 0.3
            adjusted['top_p'] = 0.7
            adjusted['top_k'] = 25
            adjusted['do_sample'] = True  # Keep sampling but very focused
            logger.debug(f"Retry 2: Very conservative params")
        
        return adjusted


# Global instance
output_validator = OutputValidator(min_quality_threshold=0.6)


# Convenience function
def validate_and_retry_generation(
    generation_func: Callable,
    text: str,
    language: str,
    max_length: Optional[int] = None,
    **generation_kwargs
) -> Tuple[str, Dict]:
    """
    Validate output and retry if needed
    
    Args:
        generation_func: Function that generates text
        text: Input text/prompt
        language: Target language
        max_length: Maximum output length
        **generation_kwargs: Additional kwargs for generation
    
    Returns:
        Tuple of (best_output, metadata)
    """
    return output_validator.validate_and_retry(
        generation_func,
        text,
        language,
        max_length,
        **generation_kwargs
    )

