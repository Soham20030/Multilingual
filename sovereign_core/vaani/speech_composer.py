#!/usr/bin/env python3
"""
Vaani Compatibility Layer

This module implements the Vaani compatibility layer that converts aligned text
to prosody-optimized JSON for Karthikeya's TTS engine.

Features:
- /compose.speech_ready endpoint
- Prosody-optimized JSON conversion
- Tone + prosody_hint field mapping
- Integration with existing Vaani TTS

Author: Soham Kotkar
"""

import json
import time
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ToneType(Enum):
    """Types of speech tones"""
    CALM = "calm"
    EXCITED = "excited"
    SERIOUS = "serious"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    GENTLE = "gentle"
    ENERGETIC = "energetic"
    SOOTHING = "soothing"

class ProsodyHint(Enum):
    """Prosody hints for TTS"""
    GENTLE_LOW = "gentle_low"
    CONFIDENT_MID = "confident_mid"
    ENERGETIC_HIGH = "energetic_high"
    CALM_STEADY = "calm_steady"
    QUESTIONING_RISE = "questioning_rise"
    STATEMENT_FALL = "statement_fall"
    EMPHATIC_STRESS = "emphatic_stress"
    SOFT_WHISPER = "soft_whisper"

@dataclass
class SpeechMetadata:
    """Metadata for speech composition"""
    language: str
    tone: str
    prosody_hint: str
    speed: float
    pitch: float
    volume: float
    pauses: List[float]
    emphasis: List[str]

class VaaniSpeechComposer:
    """
    Vaani Compatibility Layer
    
    Converts aligned text to prosody-optimized JSON for Karthikeya's TTS engine.
    Provides tone and prosody hint mapping for optimal speech synthesis.
    """
    
    def __init__(self):
        self.tone_patterns = {}
        self.prosody_mappings = {}
        self.language_settings = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the Vaani speech composer"""
        try:
            logger.info("Initializing Vaani Speech Composer...")
            
            # Initialize tone detection patterns
            self._initialize_tone_patterns()
            
            # Initialize prosody mappings
            self._initialize_prosody_mappings()
            
            # Initialize language-specific settings
            self._initialize_language_settings()
            
            self.initialized = True
            logger.info("✅ Vaani Speech Composer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vaani speech composer: {e}")
            raise
    
    def _initialize_tone_patterns(self):
        """Initialize patterns for tone detection"""
        self.tone_patterns = {
            ToneType.CALM: [
                r'\b(peaceful|calm|quiet|serene|tranquil)\b',
                r'\b(शांत|शांति|सुखद|आरामदायक)\b',
                r'\b(அமைதி|அமைதியான|சாந்தம்)\b',
                r'\b(শান্ত|শান্তি|সুন্দর|আরামদায়ক)\b'
            ],
            ToneType.EXCITED: [
                r'\b(excited|thrilled|amazing|wonderful|fantastic|great)\b',
                r'\b(उत्साहित|रोमांचक|शानदार|अद्भुत)\b',
                r'\b(உற்சாகம்|உற்சாகமான|அற்புதம்)\b',
                r'\b(উত্তেজিত|রোমাঞ্চকর|চমৎকার)\b'
            ],
            ToneType.SERIOUS: [
                r'\b(important|serious|critical|urgent|warning)\b',
                r'\b(महत्वपूर्ण|गंभीर|जरूरी|चेतावनी)\b',
                r'\b(முக்கியம்|கடுமையான|அவசியம்|எச்சரிக்கை)\b',
                r'\b(গুরুত্বপূর্ণ|গুরুতর|জরুরি|সতর্কতা)\b'
            ],
            ToneType.FRIENDLY: [
                r'\b(hello|hi|welcome|nice|good|pleased)\b',
                r'\b(नमस्ते|स्वागत|अच्छा|खुशी)\b',
                r'\b(வணக்கம்|வரவேற்பு|நல்ல|மகிழ்ச்சி)\b',
                r'\b(হ্যালো|স্বাগতম|ভাল|খুশি)\b'
            ],
            ToneType.AUTHORITATIVE: [
                r'\b(must|should|required|mandatory|obligatory)\b',
                r'\b(अवश्य|चाहिए|आवश्यक|अनिवार्य)\b',
                r'\b(கண்டிப்பாக|வேண்டும்|தேவை|கட்டாயம்)\b',
                r'\b(অবশ্যই|চাই|প্রয়োজন|বাধ্যতামূলক)\b'
            ],
            ToneType.GENTLE: [
                r'\b(gentle|soft|kind|caring|loving)\b',
                r'\b(कोमल|नरम|दयालु|प्यार|स्नेह)\b',
                r'\b(மென்மையான|மென்மை|கருணை|அன்பு)\b',
                r'\b(নম্র|নরম|দয়ালু|ভালবাসা)\b'
            ]
        }
    
    def _initialize_prosody_mappings(self):
        """Initialize prosody hint mappings"""
        self.prosody_mappings = {
            ToneType.CALM: ProsodyHint.CALM_STEADY,
            ToneType.EXCITED: ProsodyHint.ENERGETIC_HIGH,
            ToneType.SERIOUS: ProsodyHint.CONFIDENT_MID,
            ToneType.FRIENDLY: ProsodyHint.GENTLE_LOW,
            ToneType.AUTHORITATIVE: ProsodyHint.EMPHATIC_STRESS,
            ToneType.GENTLE: ProsodyHint.SOFT_WHISPER
        }
    
    def _initialize_language_settings(self):
        """Initialize language-specific TTS settings"""
        self.language_settings = {
            "hindi": {
                "default_speed": 1.0,
                "default_pitch": 1.0,
                "default_volume": 0.8,
                "pause_duration": 0.3,
                "emphasis_markers": ["!", "?"]
            },
            "sanskrit": {
                "default_speed": 0.9,
                "default_pitch": 1.1,
                "default_volume": 0.9,
                "pause_duration": 0.4,
                "emphasis_markers": ["।", "॥"]
            },
            "tamil": {
                "default_speed": 1.0,
                "default_pitch": 1.0,
                "default_volume": 0.8,
                "pause_duration": 0.3,
                "emphasis_markers": ["!", "?"]
            },
            "bengali": {
                "default_speed": 1.0,
                "default_pitch": 1.0,
                "default_volume": 0.8,
                "pause_duration": 0.3,
                "emphasis_markers": ["!", "?"]
            },
            "english": {
                "default_speed": 1.0,
                "default_pitch": 1.0,
                "default_volume": 0.8,
                "pause_duration": 0.3,
                "emphasis_markers": ["!", "?"]
            }
        }
    
    async def compose_speech(self, text: str, language: str, tone: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compose speech-ready text with prosody optimization
        
        Args:
            text: Aligned text to convert
            language: Target language
            tone: Specific tone (auto-detect if None)
            context: Additional context
            
        Returns:
            Dictionary with speech-ready data
        """
        try:
            # Detect tone if not provided
            if not tone:
                tone = self._detect_tone(text)
            
            # Get prosody hint for the tone
            prosody_hint = self._get_prosody_hint(tone)
            
            # Get language-specific settings
            lang_settings = self.language_settings.get(language, self.language_settings["english"])
            
            # Adjust settings based on tone
            adjusted_settings = self._adjust_settings_for_tone(lang_settings, tone)
            
            # Ensure all required keys exist
            if "speed" not in adjusted_settings:
                adjusted_settings["speed"] = 1.0
            if "pitch" not in adjusted_settings:
                adjusted_settings["pitch"] = 1.0
            if "volume" not in adjusted_settings:
                adjusted_settings["volume"] = 0.8
            
            # Add prosody markers
            prosody_text = self._add_prosody_markers(text, language, tone)
            
            # Create speech metadata
            speech_metadata = SpeechMetadata(
                language=language,
                tone=tone,
                prosody_hint=prosody_hint.value,
                speed=adjusted_settings["speed"],
                pitch=adjusted_settings["pitch"],
                volume=adjusted_settings["volume"],
                pauses=self._calculate_pauses(text, language),
                emphasis=self._extract_emphasis(text, language)
            )
            
            # Create audio metadata for TTS
            audio_metadata = {
                "text": prosody_text,
                "language": language,
                "voice": self._select_voice(language, tone),
                "speed": adjusted_settings["speed"],
                "pitch": adjusted_settings["pitch"],
                "volume": adjusted_settings["volume"],
                "prosody_hint": prosody_hint.value,
                "format": "wav",
                "sample_rate": 22050
            }
            
            result = {
                "text": prosody_text,
                "tone": tone,
                "lang": language,
                "prosody_hint": prosody_hint.value,
                "audio_metadata": audio_metadata,
                "speech_metadata": {
                    "speed": speech_metadata.speed,
                    "pitch": speech_metadata.pitch,
                    "volume": speech_metadata.volume,
                    "pauses": speech_metadata.pauses,
                    "emphasis": speech_metadata.emphasis
                }
            }
            
            logger.info(f"Composed speech: {tone} tone, {prosody_hint.value} prosody for {language}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compose speech: {e}")
            raise
    
    def _detect_tone(self, text: str) -> str:
        """Detect tone from text content"""
        text_lower = text.lower()
        
        # Check each tone pattern
        tone_scores = {}
        
        for tone_type, patterns in self.tone_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            tone_scores[tone_type] = score
        
        # Return tone with highest score, default to calm
        if tone_scores and max(tone_scores.values()) > 0:
            detected_tone = max(tone_scores.items(), key=lambda x: x[1])[0]
            return detected_tone.value
        
        return ToneType.CALM.value
    
    def _get_prosody_hint(self, tone: str) -> ProsodyHint:
        """Get prosody hint for the given tone"""
        try:
            tone_enum = ToneType(tone)
            return self.prosody_mappings.get(tone_enum, ProsodyHint.CALM_STEADY)
        except ValueError:
            return ProsodyHint.CALM_STEADY
    
    def _adjust_settings_for_tone(self, base_settings: Dict[str, Any], tone: str) -> Dict[str, Any]:
        """Adjust TTS settings based on tone"""
        settings = base_settings.copy()
        
        # Ensure base settings have all required keys
        if "default_speed" not in settings:
            settings["default_speed"] = 1.0
        if "default_pitch" not in settings:
            settings["default_pitch"] = 1.0
        if "default_volume" not in settings:
            settings["default_volume"] = 0.8
        
        tone_adjustments = {
            ToneType.CALM.value: {"speed": 0.9, "pitch": 0.95, "volume": 0.8},
            ToneType.EXCITED.value: {"speed": 1.2, "pitch": 1.1, "volume": 0.9},
            ToneType.SERIOUS.value: {"speed": 0.95, "pitch": 1.05, "volume": 0.85},
            ToneType.FRIENDLY.value: {"speed": 1.05, "pitch": 1.0, "volume": 0.8},
            ToneType.AUTHORITATIVE.value: {"speed": 0.9, "pitch": 1.1, "volume": 0.9},
            ToneType.GENTLE.value: {"speed": 0.85, "pitch": 0.9, "volume": 0.75}
        }
        
        if tone in tone_adjustments:
            adjustments = tone_adjustments[tone]
            settings["speed"] = settings["default_speed"] * adjustments["speed"]
            settings["pitch"] = settings["default_pitch"] * adjustments["pitch"]
            settings["volume"] = settings["default_volume"] * adjustments["volume"]
        else:
            # Use default values if tone not found
            settings["speed"] = settings["default_speed"]
            settings["pitch"] = settings["default_pitch"]
            settings["volume"] = settings["default_volume"]
        
        return settings
    
    def _add_prosody_markers(self, text: str, language: str, tone: str) -> str:
        """Add prosody markers to text for better TTS"""
        # Add pauses after sentences
        if language in ["hindi", "sanskrit"]:
            # Add pause after Devanagari sentence endings
            text = re.sub(r'([।॥])\s*', r'\1<break time="0.5s"/> ', text)
        elif language in ["tamil", "bengali"]:
            # Add pause after sentence endings
            text = re.sub(r'([।.])\s*', r'\1<break time="0.5s"/> ', text)
        else:
            # Add pause after sentence endings
            text = re.sub(r'([.!?])\s*', r'\1<break time="0.5s"/> ', text)
        
        # Add emphasis markers based on tone
        if tone == ToneType.AUTHORITATIVE.value:
            # Emphasize important words
            text = re.sub(r'\b(must|should|required|अवश्य|चाहिए|கண்டிப்பாக|অবশ্যই)\b', 
                         r'<emphasis level="strong">\1</emphasis>', text, flags=re.IGNORECASE)
        
        elif tone == ToneType.GENTLE.value:
            # Soften the text
            text = re.sub(r'\b(please|kindly|कृपया|தயவு|দয়া)\b', 
                         r'<emphasis level="reduced">\1</emphasis>', text, flags=re.IGNORECASE)
        
        return text
    
    def _calculate_pauses(self, text: str, language: str) -> List[float]:
        """Calculate pause durations for the text"""
        pauses = []
        
        # Count sentences to determine pause points
        if language in ["hindi", "sanskrit"]:
            sentences = re.split(r'[।॥]+', text)
        elif language in ["tamil", "bengali"]:
            sentences = re.split(r'[।.]+', text)
        else:
            sentences = re.split(r'[.!?]+', text)
        
        # Add pause after each sentence (except the last one)
        for i in range(len(sentences) - 1):
            pauses.append(0.5)  # 500ms pause
        
        return pauses
    
    def _extract_emphasis(self, text: str, language: str) -> List[str]:
        """Extract words that should be emphasized"""
        emphasis_words = []
        
        # Look for words in caps or with special markers
        caps_words = re.findall(r'\b[A-Z][A-Z]+\b', text)
        emphasis_words.extend(caps_words)
        
        # Look for words with asterisks or other emphasis markers
        marked_words = re.findall(r'\*([^*]+)\*', text)
        emphasis_words.extend(marked_words)
        
        # Look for question words
        question_words = re.findall(r'\b(what|how|why|when|where|who|क्या|कैसे|क्यों|என்ன|எப்படி|কী|কীভাবে)\b', 
                                   text, flags=re.IGNORECASE)
        emphasis_words.extend(question_words)
        
        return emphasis_words
    
    def _select_voice(self, language: str, tone: str) -> str:
        """Select appropriate voice for language and tone"""
        voice_mapping = {
            "hindi": {
                ToneType.CALM.value: "hindi_female_calm",
                ToneType.EXCITED.value: "hindi_female_energetic",
                ToneType.SERIOUS.value: "hindi_male_serious",
                ToneType.FRIENDLY.value: "hindi_female_friendly",
                ToneType.AUTHORITATIVE.value: "hindi_male_authoritative",
                ToneType.GENTLE.value: "hindi_female_gentle"
            },
            "sanskrit": {
                ToneType.CALM.value: "sanskrit_male_calm",
                ToneType.EXCITED.value: "sanskrit_male_energetic",
                ToneType.SERIOUS.value: "sanskrit_male_serious",
                ToneType.FRIENDLY.value: "sanskrit_male_friendly",
                ToneType.AUTHORITATIVE.value: "sanskrit_male_authoritative",
                ToneType.GENTLE.value: "sanskrit_male_gentle"
            },
            "tamil": {
                ToneType.CALM.value: "tamil_female_calm",
                ToneType.EXCITED.value: "tamil_female_energetic",
                ToneType.SERIOUS.value: "tamil_male_serious",
                ToneType.FRIENDLY.value: "tamil_female_friendly",
                ToneType.AUTHORITATIVE.value: "tamil_male_authoritative",
                ToneType.GENTLE.value: "tamil_female_gentle"
            },
            "english": {
                ToneType.CALM.value: "english_female_calm",
                ToneType.EXCITED.value: "english_female_energetic",
                ToneType.SERIOUS.value: "english_male_serious",
                ToneType.FRIENDLY.value: "english_female_friendly",
                ToneType.AUTHORITATIVE.value: "english_male_authoritative",
                ToneType.GENTLE.value: "english_female_gentle"
            }
        }
        
        return voice_mapping.get(language, {}).get(tone, f"{language}_default")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the speech composer"""
        return {
            "initialized": self.initialized,
            "supported_tones": [tone.value for tone in ToneType],
            "supported_prosody_hints": [hint.value for hint in ProsodyHint],
            "supported_languages": list(self.language_settings.keys()),
            "tone_patterns_count": sum(len(patterns) for patterns in self.tone_patterns.values()),
            "prosody_mappings_count": len(self.prosody_mappings)
        }
