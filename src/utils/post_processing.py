"""
Output Post-Processing Module
==============================

Cleans and refines model-generated text for all 21 supported languages.

Features:
1. Script Validation - Ensure correct Unicode ranges
2. Deduplication - Remove repeated phrases/sentences
3. Length Control - Trim incomplete sentences
4. Mixed Language Filter - Remove text in wrong language
5. Format Cleanup - Fix spacing, punctuation
6. Quality Scoring - Rate output quality
"""

import re
import logging
from typing import Tuple, Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# Unicode script ranges for each language
SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F)],  # Hindi, Sanskrit, Marathi, Nepali
    "bengali": [(0x0980, 0x09FF)],     # Bengali, Assamese
    "tamil": [(0x0B80, 0x0BFF)],       # Tamil
    "telugu": [(0x0C00, 0x0C7F)],      # Telugu
    "kannada": [(0x0C80, 0x0CFF)],     # Kannada
    "malayalam": [(0x0D00, 0x0D7F)],   # Malayalam
    "gujarati": [(0x0A80, 0x0AFF)],    # Gujarati
    "oriya": [(0x0B00, 0x0B7F)],       # Odia
    "punjabi": [(0x0A00, 0x0A7F)],     # Punjabi
    "urdu": [(0x0600, 0x06FF), (0x0750, 0x077F)],  # Arabic script
    "kashmiri": [(0x0600, 0x06FF)],    # Arabic script
    "sindhi": [(0x0600, 0x06FF)],      # Arabic script
    "santali": [(0x1C50, 0x1C7F)],     # Ol Chiki script
    "meitei": [(0xABC0, 0xABFF)],      # Meetei Mayek script
    "bodo": [(0x0900, 0x097F)],        # Devanagari
    "maithili": [(0x0900, 0x097F)],    # Devanagari
    "english": [(0x0020, 0x007F)],     # Basic Latin
}

# Language to script mapping
LANGUAGE_TO_SCRIPT = {
    "hindi": "devanagari",
    "sanskrit": "devanagari",
    "marathi": "devanagari",
    "nepali": "devanagari",
    "bodo": "devanagari",
    "maithili": "devanagari",
    "bengali": "bengali",
    "assamese": "bengali",
    "tamil": "tamil",
    "telugu": "telugu",
    "kannada": "kannada",
    "malyalam": "malayalam",
    "gujarati": "gujarati",
    "gujurati": "gujarati",  # Handle spelling variant
    "odia": "oriya",
    "punjabi": "punjabi",
    "urdu": "urdu",
    "kashmiri": "kashmiri",
    "sindhi": "sindhi",
    "santali": "santali",
    "meitei": "meitei",
    "english": "english",
}


class OutputPostProcessor:
    """Main class for post-processing generated text"""
    
    def __init__(self, enable_all: bool = True):
        """
        Initialize post-processor
        
        Args:
            enable_all: Enable all post-processing features by default
        """
        self.config = {
            "script_validation": enable_all,
            "deduplication": enable_all,
            "length_control": enable_all,
            "mixed_language_filter": enable_all,
            "format_cleanup": enable_all,
            "quality_scoring": enable_all,
        }
        logger.info(f"PostProcessor initialized with config: {self.config}")
    
    def process(
        self,
        text: str,
        language: str,
        max_length: Optional[int] = None
    ) -> Tuple[str, Dict]:
        """
        Process generated text through all enabled post-processing steps
        
        Args:
            text: Generated text to process
            language: Language of the text
            max_length: Maximum allowed length (characters)
        
        Returns:
            Tuple of (processed_text, metadata)
        """
        if not text or not text.strip():
            return text, {"quality_score": 0.0, "processed": False}
        
        original_text = text
        metadata = {
            "original_length": len(text),
            "steps_applied": [],
        }
        
        # Step 1: Format cleanup (do this first)
        if self.config["format_cleanup"]:
            text = self._cleanup_format(text)
            metadata["steps_applied"].append("format_cleanup")
        
        # Step 2: Deduplication
        if self.config["deduplication"]:
            text = self._remove_duplicates(text)
            metadata["steps_applied"].append("deduplication")
        
        # Step 3: Mixed language filter
        if self.config["mixed_language_filter"]:
            text = self._filter_mixed_languages(text, language)
            metadata["steps_applied"].append("mixed_language_filter")
        
        # Step 4: Script validation
        if self.config["script_validation"]:
            text = self._validate_script(text, language)
            metadata["steps_applied"].append("script_validation")
        
        # Step 5: Length control
        if self.config["length_control"]:
            text = self._control_length(text, max_length)
            metadata["steps_applied"].append("length_control")
        
        # Step 6: Quality scoring
        if self.config["quality_scoring"]:
            quality_score = self._score_quality(text, language, original_text)
            metadata["quality_score"] = quality_score
        
        metadata["final_length"] = len(text)
        metadata["processed"] = True
        metadata["improvement"] = self._calculate_improvement(original_text, text)
        
        logger.debug(f"Post-processed {language} text: {len(original_text)} → {len(text)} chars")
        
        return text, metadata
    
    def _cleanup_format(self, text: str) -> str:
        """Clean up formatting issues (spacing, punctuation, line breaks)"""
        if not text:
            return text
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        # Remove spaces before newlines
        text = re.sub(r'\s+\n', '\n', text)
        text = re.sub(r'\n\s+', '\n', text)
        
        # Limit consecutive newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_duplicates(self, text: str) -> str:
        """Remove repeated phrases and sentences"""
        if not text:
            return text
        
        # Split into sentences (basic approach)
        sentences = re.split(r'([.!?।॥\n]+)', text)
        
        seen = set()
        result = []
        
        for i in range(0, len(sentences), 2):
            if i >= len(sentences):
                break
                
            sentence = sentences[i].strip()
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ''
            
            if not sentence:
                continue
            
            # Normalize for comparison (lowercase, remove extra spaces)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            
            # Check for exact duplicates
            if normalized not in seen:
                seen.add(normalized)
                result.append(sentence + delimiter)
            else:
                logger.debug(f"Removed duplicate: {sentence[:50]}...")
        
        return ''.join(result)
    
    def _filter_mixed_languages(self, text: str, language: str) -> str:
        """Filter out text in wrong language/script - ENHANCED"""
        if not text or language not in LANGUAGE_TO_SCRIPT:
            return text
        
        expected_script = LANGUAGE_TO_SCRIPT[language]
        
        # 🎯 STRICTER: For non-English languages, limit English content
        if language != "english":
            # Filter English-heavy lines
            lines = text.split('\n')
            filtered_lines = []
            
            for line in lines:
                if not line.strip():
                    filtered_lines.append(line)
                    continue
                
                # Count English vs native script
                english_chars = sum(1 for c in line if c.isalpha() and ord(c) < 128)
                native_chars = sum(1 for c in line if c.isalpha() and ord(c) >= 128)
                total_alpha = english_chars + native_chars
                
                if total_alpha == 0:
                    filtered_lines.append(line)
                    continue
                
                english_ratio = english_chars / total_alpha if total_alpha > 0 else 0
                
                # 🎯 BALANCED FILTER: Remove only extremely English-heavy lines
                if english_ratio > 0.7 and len(line) > 30:  # Changed from 0.4 to 0.7
                    logger.debug(f"Filtered English-heavy line ({english_ratio:.1%} English): {line[:50]}...")
                    continue
                
                # Check for wrong scripts
                scripts_present = self._detect_scripts(line)
                
                # Keep if it uses expected script (even with some English)
                if expected_script in scripts_present or len(line) < 20:
                    filtered_lines.append(line)
                else:
                    logger.debug(f"Filtered wrong script line: {line[:50]}...")
            
            return '\n'.join(filtered_lines)
        
        # For English, just do basic script check
        return text
    
    def _detect_scripts(self, text: str) -> List[str]:
        """Detect which scripts are present in text"""
        script_counts = Counter()
        
        for char in text:
            if not char.isalpha():
                continue
            
            char_code = ord(char)
            
            for script_name, ranges in SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= char_code <= end:
                        script_counts[script_name] += 1
                        break
        
        # Return scripts sorted by frequency
        return [script for script, _ in script_counts.most_common()]
    
    def _validate_script(self, text: str, language: str) -> str:
        """Validate that text uses correct script for language"""
        if not text or language not in LANGUAGE_TO_SCRIPT:
            return text
        
        expected_script = LANGUAGE_TO_SCRIPT[language]
        
        # Build allowed character set
        allowed_ranges = SCRIPT_RANGES.get(expected_script, [])
        # Always allow English characters and common punctuation
        allowed_ranges.extend(SCRIPT_RANGES["english"])
        allowed_ranges.append((0x0000, 0x007F))  # Basic Latin + ASCII
        
        # Filter characters
        result = []
        for char in text:
            char_code = ord(char)
            
            # Always allow whitespace, punctuation, numbers
            if char.isspace() or char in '.,!?;:।॥""\'()[]{}/-—–…':
                result.append(char)
                continue
            
            # Check if character is in allowed ranges
            is_allowed = False
            for start, end in allowed_ranges:
                if start <= char_code <= end:
                    is_allowed = True
                    break
            
            if is_allowed:
                result.append(char)
        
        return ''.join(result)
    
    def _control_length(self, text: str, max_length: Optional[int]) -> str:
        """Control text length and trim incomplete sentences"""
        if not text:
            return text
        
        # Trim to max length if specified
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        # Find last complete sentence
        # Look for sentence-ending punctuation
        sentence_ends = ['.', '!', '?', '।', '॥']
        
        last_end = -1
        for i in range(len(text) - 1, max(len(text) - 200, 0), -1):
            if text[i] in sentence_ends:
                last_end = i
                break
        
        # If found a sentence end, trim there
        if last_end > len(text) // 2:  # Only trim if we're not cutting too much
            text = text[:last_end + 1].strip()
        
        return text
    
    def _score_quality(self, text: str, language: str, original_text: str) -> float:
        """Score output quality (0.0 to 1.0) - ENHANCED"""
        if not text:
            return 0.0
        
        score = 1.0
        
        # Penalty for very short outputs - BALANCED
        if len(text) < 20:
            score -= 0.5  # Very short
        elif len(text) < 40:
            score -= 0.2  # Somewhat short
        
        # Penalty for excessive length (> 1000 chars without structure)
        if len(text) > 1000 and '\n' not in text:
            score -= 0.2
        
        # 🎯 NEW: Penalty for too much repetition - MORE SENSITIVE
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.6:  # More than 40% repeated words (stricter)
                score -= 0.4
        
        # 🎯 ENHANCED: Stronger penalty for mixed scripts
        if language != "english":
            scripts = self._detect_scripts(text)
            if len(scripts) > 2:  # More than 2 scripts
                score -= 0.3  # Increased from 0.2
            elif len(scripts) > 1 and "english" not in scripts:
                # Mixed non-English scripts (e.g., Chinese + Bengali)
                score -= 0.5  # Heavy penalty
        
        # Bonus for proper sentence structure
        if any(char in text for char in '.!?।॥'):
            score += 0.1
        
        # Penalty for too much text removed
        if len(original_text) > 0:
            removal_ratio = 1 - (len(text) / len(original_text))
            if removal_ratio > 0.5:  # Removed more than 50%
                score -= 0.2
        
        # 🎯 NEW: Check for expected script dominance
        expected_script = LANGUAGE_TO_SCRIPT.get(language)
        if expected_script and expected_script != "english":
            scripts = self._detect_scripts(text)
            if scripts and scripts[0] != expected_script:
                # Wrong primary script
                score -= 0.4
            
            # 🎯 ENHANCED: Check English mixing ratio
            english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
            native_chars = sum(1 for c in text if c.isalpha() and ord(c) >= 128)
            total_alpha = english_chars + native_chars
            
            if total_alpha > 0:
                english_ratio = english_chars / total_alpha
                
                # Balanced penalty for English mixing
                if english_ratio > 0.7:  # More than 70% English - very bad
                    score -= 0.5
                elif english_ratio > 0.5:  # 50-70% English - bad
                    score -= 0.3
                elif english_ratio > 0.4:  # 40-50% English - warning
                    score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_improvement(self, original: str, processed: str) -> Dict:
        """Calculate improvement metrics"""
        return {
            "length_change": len(processed) - len(original),
            "length_change_percent": ((len(processed) - len(original)) / len(original) * 100) if len(original) > 0 else 0,
            "characters_removed": len(original) - len(processed),
        }


# Global instance
post_processor = OutputPostProcessor(enable_all=True)


# Convenience function
def process_output(
    text: str,
    language: str = "english",
    max_length: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    Process generated output text
    
    Args:
        text: Generated text to process
        language: Language of the text
        max_length: Maximum allowed length
    
    Returns:
        Tuple of (processed_text, metadata)
    """
    return post_processor.process(text, language, max_length)

