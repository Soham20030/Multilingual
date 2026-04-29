#!/usr/bin/env python3
"""
KSML Semantic Alignment Engine

This module implements the core KSML (Knowledge, Semantic, Multilingual, Language) 
semantic alignment engine that processes raw LM text and adds:

1. Intent Classification
2. Language Detection (source/target)
3. Karma State Classification (sattva/rajas/tamas)
4. Sanskrit Root Tagging via predefined lookup

Author: Soham Kotkar
"""

import json
import re
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
	from sovereign_core.rl.policy import get_policy_manager
except Exception:
	get_policy_manager = None  # type: ignore

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of intents the system can classify"""
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    INSTRUCTION = "instruction"
    GREETING = "greeting"
    EXPLANATION = "explanation"
    TRANSLATION = "translation"
    EDUCATIONAL = "educational"
    CONVERSATIONAL = "conversational"

class KarmaState(Enum):
    """Karma states based on Vedic philosophy"""
    SATTVIC = "sattva"  # Pure, harmonious, balanced
    RAJASIC = "rajas"   # Active, passionate, dynamic
    TAMASIC = "tamas"  # Inert, dark, destructive

@dataclass
class KSMLResult:
    """Result of KSML alignment processing"""
    intent: str
    source_lang: str
    target_lang: str
    karma_state: str
    semantic_roots: List[str]
    confidence: float
    metadata: Dict[str, Any]

class KSMLAligner:
    """
    =============================================================================
    KSML SEMANTIC ALIGNMENT ENGINE
    =============================================================================
    TODO: This processes text from Bhavesh's LM Core
    TODO: Confirm input format with Bhavesh
    TODO: Test with real Bhavesh responses
    TODO: Optimize performance for production
    """
    
    def __init__(self):
        self.ksml_roots = {}
        self.intent_patterns = {}
        self.karma_patterns = {}
        self.language_keywords = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize the KSML aligner with data and patterns"""
        try:
            logger.info("Initializing KSML Semantic Alignment Engine...")
            
            # Load Sanskrit roots lookup
            await self._load_ksml_roots()
            
            # Initialize intent classification patterns
            self._initialize_intent_patterns()
            
            # Initialize karma state patterns
            self._initialize_karma_patterns()
            
            # Initialize language detection patterns
            self._initialize_language_patterns()
            
            self.initialized = True
            logger.info("✅ KSML Aligner initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize KSML aligner: {e}")
            raise
    
    async def _load_ksml_roots(self):
        """Load Sanskrit roots lookup from JSON file"""
        try:
            roots_file = Path(__file__).parent / "ksml_roots.json"
            
            if roots_file.exists():
                with open(roots_file, 'r', encoding='utf-8') as f:
                    self.ksml_roots = json.load(f)
                logger.info(f"Loaded {len(self.ksml_roots)} Sanskrit roots")
            else:
                # Create default roots file if it doesn't exist
                await self._create_default_roots_file(roots_file)
                
        except Exception as e:
            logger.error(f"Failed to load KSML roots: {e}")
            self.ksml_roots = {}
    
    async def _create_default_roots_file(self, roots_file: Path):
        """Create default Sanskrit roots lookup file"""
        default_roots = {
            "धातु": {
                "meaning": "root, foundation",
                "category": "fundamental",
                "karma_state": "sattva",
                "intent": "educational"
            },
            "अर्थ": {
                "meaning": "meaning, purpose",
                "category": "semantic",
                "karma_state": "sattva",
                "intent": "explanation"
            },
            "भाव": {
                "meaning": "feeling, emotion",
                "category": "emotional",
                "karma_state": "rajas",
                "intent": "conversational"
            },
            "ज्ञान": {
                "meaning": "knowledge, wisdom",
                "category": "educational",
                "karma_state": "sattva",
                "intent": "educational"
            },
            "कर्म": {
                "meaning": "action, work",
                "category": "action",
                "karma_state": "rajas",
                "intent": "command"
            },
            "धर्म": {
                "meaning": "duty, righteousness",
                "category": "moral",
                "karma_state": "sattva",
                "intent": "explanation"
            },
            "मोक्ष": {
                "meaning": "liberation, freedom",
                "category": "spiritual",
                "karma_state": "sattva",
                "intent": "educational"
            },
            "योग": {
                "meaning": "union, discipline",
                "category": "spiritual",
                "karma_state": "sattva",
                "intent": "educational"
            },
            "भक्ति": {
                "meaning": "devotion, love",
                "category": "emotional",
                "karma_state": "rajas",
                "intent": "conversational"
            },
            "तपस्": {
                "meaning": "austerity, penance",
                "category": "spiritual",
                "karma_state": "sattva",
                "intent": "educational"
            }
        }
        
        with open(roots_file, 'w', encoding='utf-8') as f:
            json.dump(default_roots, f, ensure_ascii=False, indent=2)
        
        self.ksml_roots = default_roots
        logger.info(f"Created default KSML roots file with {len(default_roots)} entries")
    
    def _initialize_intent_patterns(self):
        """Initialize patterns for intent classification"""
        self.intent_patterns = {
            IntentType.QUESTION: [
                # English - Enhanced patterns
                r'\b(what|how|why|when|where|who|which|can|could|would|should|is|are|do|does|did)\b',
                r'\b(tell me about|what is|how does|explain|describe)\b',
                r'\b(what\'s|how\'s|why\'s|when\'s|where\'s|who\'s)\b',
                # Hindi
                r'\b(क्या|कैसे|क्यों|कब|कहाँ|कौन|कौनसा|क्या|है|हैं|कर|करता|करती)\b',
                # Sanskrit
                r'\b(किम्|कथम्|कुत्र|कदा|कः|का|किम्|अस्ति|भवति|करोति)\b',
                # Tamil
                r'\b(என்ன|எப்படி|ஏன்|எப்போது|எங்கே|யார்|எந்த|ஆகும்|இருக்கிறது)\b',
                # Bengali
                r'\b(কী|কীভাবে|কেন|কখন|কোথায়|কে|কোন|হয়|আছে|করে)\b'
            ],
            IntentType.COMMAND: [
                # English - Basic commands
                r'\b(please|do|go|come|give|take|show|tell|stop|start)\b',
                # Hindi
                r'\b(कृपया|करो|जाओ|आओ|दो|लो|दिखाओ|बताओ|रुको|शुरू)\b',
                # Sanskrit
                r'\b(कृपया|कुरु|गच्छ|आगच्छ|देहि|गृहाण|दर्शय|कथय|स्थगय|आरभ)\b'
            ],
            IntentType.INSTRUCTION: [
                # English - Action-oriented instructions
                r'\b(create|build|make|write|develop|generate|design|construct|produce)\b',
                r'\b(create a|build a|make a|write a|develop a|generate a|design a)\b',
                r'\b(create an|build an|make an|write an|develop an|generate an|design an)\b',
                # Hindi
                r'\b(बनाओ|निर्माण|लिखो|विकसित|उत्पन्न|डिजाइन|निर्माण करो)\b',
                # Sanskrit
                r'\b(निर्माण|रचय|लिख|विकस|उत्पादय|रूपय|निर्माण कुरु)\b'
            ],
            IntentType.GREETING: [
                # English
                r'\b(hello|hi|hey|good morning|good afternoon|good evening|namaste|namaskar)\b',
                # Hindi
                r'\b(नमस्ते|नमस्कार|सुप्रभात|शुभ संध्या|आदाब|सलाम)\b',
                # Sanskrit
                r'\b(नमस्ते|नमस्कारः|सुप्रभातम्|शुभसंध्या|आदाबः|सलामः)\b',
                # Tamil
                r'\b(வணக்கம்|வாழ்க|நமஸ்காரம்|காலை வணக்கம்)\b'
            ],
            IntentType.EXPLANATION: [
                # English - Enhanced patterns with higher priority
                r'^(explain|describe|define|clarify|elaborate)',
                r'\b(explain|describe|tell me about|what is|how does|means|definition)\b',
                r'\b(explanation|description|definition|clarification|elaboration)\b',
                # Hindi
                r'\b(समझाओ|वर्णन|बताओ|क्या है|कैसे|मतलब|परिभाषा)\b',
                # Sanskrit
                r'\b(व्याख्या|वर्णन|कथय|किम् अस्ति|कथम्|अर्थः|परिभाषा)\b'
            ],
            IntentType.TRANSLATION: [
                # English
                r'\b(translate|translation|in hindi|in english|in tamil|in sanskrit)\b',
                # Hindi
                r'\b(अनुवाद|हिंदी में|अंग्रेजी में|तमिल में|संस्कृत में)\b',
                # Sanskrit
                r'\b(अनुवादः|हिन्द्याम्|आङ्ग्ल्याम्|तमिळ्याम्|संस्कृते)\b'
            ]
        }
    
    def _initialize_karma_patterns(self):
        """Initialize patterns for karma state classification"""
        self.karma_patterns = {
            KarmaState.SATTVIC: [
                # Peaceful, harmonious, balanced
                r'\b(peace|calm|harmony|balance|wisdom|knowledge|truth|love|compassion|kindness)\b',
                r'\b(शांति|शांत|सामंजस्य|संतुलन|ज्ञान|सत्य|प्रेम|करुणा|दया)\b',
                r'\b(शान्तिः|शान्तः|सामञ्जस्यः|संतुलनम्|ज्ञानम्|सत्यम्|प्रेम|करुणा|दया)\b',
                r'\b(அமைதி|அமைதியான|ஒற்றுமை|சமநிலை|ஞானம்|உண்மை|அன்பு|கருணை)\b'
            ],
            KarmaState.RAJASIC: [
                # Active, passionate, dynamic - Enhanced patterns
                r'\b(action|passion|energy|dynamic|active|ambition|desire|work|effort|striving)\b',
                r'\b(create|build|make|write|develop|generate|design|construct|produce|work|do)\b',
                r'\b(create a|build a|make a|write a|develop a|generate a|design a|work on)\b',
                r'\b(कर्म|जुनून|ऊर्जा|गतिशील|सक्रिय|महत्वाकांक्षा|इच्छा|काम|प्रयास)\b',
                r'\b(बनाओ|निर्माण|लिखो|विकसित|उत्पन्न|डिजाइन|काम करो|करो)\b',
                r'\b(कर्म|जुनूनः|ऊर्जा|गतिशीलः|सक्रियः|महत्वाकाङ्क्षा|इच्छा|कामः|प्रयासः)\b',
                r'\b(निर्माण|रचय|लिख|विकस|उत्पादय|रूपय|कर्म कुरु|कुरु)\b',
                r'\b(செயல்|ஆர்வம்|ஆற்றல்|இயக்க|செயல்பாட்டு|ஆசை|வேலை|முயற்சி)\b'
            ],
            KarmaState.TAMASIC: [
                # Inert, dark, destructive
                r'\b(darkness|ignorance|destruction|violence|anger|hatred|fear|confusion|chaos)\b',
                r'\b(अंधकार|अज्ञान|विनाश|हिंसा|क्रोध|घृणा|भय|भ्रम|अराजकता)\b',
                r'\b(अन्धकारः|अज्ञानम्|विनाशः|हिंसा|क्रोधः|घृणा|भयः|भ्रमः|अराजकता)\b',
                r'\b(இருள்|அறியாமை|அழிவு|வன்முறை|கோபம்|வெறுப்பு|பயம்|குழப்பம்)\b'
            ]
        }
    
    def _initialize_language_patterns(self):
        """Initialize patterns for language detection"""
        self.language_keywords = {
            "hindi": ["है", "हैं", "था", "थी", "क्या", "कैसे", "क्यों", "कब", "कहाँ", "कौन"],
            "sanskrit": ["अस्ति", "भवति", "करोति", "गच्छति", "आगच्छति", "किम्", "कथम्", "कुत्र"],
            "tamil": ["ஆகும்", "இருக்கிறது", "செய்கிறது", "என்ன", "எப்படி", "ஏன்", "எப்போது"],
            "bengali": ["হয়", "আছে", "করে", "কী", "কীভাবে", "কেন", "কখন", "কোথায়"],
            "english": ["is", "are", "was", "were", "what", "how", "why", "when", "where", "who"],
            "telugu": ["అవుతుంది", "ఉంది", "చేస్తుంది", "ఏమి", "ఎలా", "ఎందుకు", "ఎప్పుడు"],
            "kannada": ["ಆಗುತ್ತದೆ", "ಇದೆ", "ಮಾಡುತ್ತದೆ", "ಏನು", "ಹೇಗೆ", "ಏಕೆ", "ಯಾವಾಗ"],
            "gujarati": ["છે", "છે", "કરે", "શું", "કેવી", "કેમ", "ક્યારે", "ક્યાં"],
            "marathi": ["आहे", "आहेत", "करते", "काय", "कसे", "का", "केव्हा", "कुठे"],
            "punjabi": ["ਹੈ", "ਹੈ", "ਕਰਦਾ", "ਕੀ", "ਕਿਵੇਂ", "ਕਿਉਂ", "ਕਦੋਂ", "ਕਿੱਥੇ"]
        }
    
    async def align_text(self, text: str, source_lang: Optional[str] = None, 
                        target_lang: str = "en", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        =============================================================================
        MAIN KSML SEMANTIC ALIGNMENT METHOD
        =============================================================================
        TODO: This processes Bhavesh's LM Core responses
        TODO: Confirm text format with Bhavesh
        TODO: Test with real Bhavesh data
        TODO: Optimize for production performance
        """
        if not self.initialized:
            raise RuntimeError("KSML aligner not initialized")
        
        start_time = time.time()
        
        try:
            # Step 1: Detect source language if not provided
            if not source_lang:
                source_lang = self._detect_language(text)
            
            # Step 2: Classify intent
            intent = self._classify_intent(text)
            
            # Step 3: Determine karma state
            karma_state = self._classify_karma_state(text)
            
            # Step 4: Extract Sanskrit roots
            semantic_roots = self._extract_sanskrit_roots(text)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(text, intent, karma_state, semantic_roots)

            # Step 6: Apply RL policy nudges (if available)
            tone = None
            if get_policy_manager is not None:
                try:
                    pm = get_policy_manager()
                    actions = pm.get_alignment_actions(input_text=text, source_lang=source_lang)
                    # target_lang override
                    if actions.get("target_lang_override"):
                        target_lang = actions["target_lang_override"]
                    # karma bias
                    karma_bias = actions.get("karma_bias")
                    if isinstance(karma_bias, str):
                        from enum import Enum as _E
                        try:
                            karma_state = KarmaState(karma_bias)  # type: ignore[arg-type]
                        except Exception:
                            pass
                    # intent bias (future use)
                    # confidence delta
                    confidence = max(0.0, min(1.0, confidence + float(actions.get("confidence_delta", 0.0))))
                    tone = actions.get("tone")
                except Exception:
                    pass
            
            processing_time = time.time() - start_time
            
            result = {
                "intent": intent.value,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "karma_state": karma_state.value,
                "semantic_roots": semantic_roots,
                "confidence": confidence,
                "processing_time": processing_time,
                "aligned_text": text,  # Add this key
                "metadata": {
                    "text_length": len(text),
                    "word_count": len(text.split()),
                    "context": context or {},
                    "tone": tone
                }
            }
            
            logger.info(f"KSML alignment completed: {intent.value} ({karma_state.value}) - {confidence:.2f} confidence")
            
            return result
            
        except Exception as e:
            logger.error(f"KSML alignment failed: {e}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        text_lower = text.lower()
        
        # Count matches for each language
        language_scores = {}
        
        for lang, keywords in self.language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            language_scores[lang] = score
        
        # Return language with highest score, default to English
        if language_scores:
            detected_lang = max(language_scores.items(), key=lambda x: x[1])[0]
            if language_scores[detected_lang] > 0:
                return detected_lang
        
        return "english"  # Default fallback
    
    def _classify_intent(self, text: str) -> IntentType:
        """Classify the intent of the input text"""
        text_lower = text.lower()
        
        # Check each intent pattern
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            intent_scores[intent_type] = score
        
        # Return intent with highest score, default to conversational
        if intent_scores and max(intent_scores.values()) > 0:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return IntentType.CONVERSATIONAL
    
    def _classify_karma_state(self, text: str) -> KarmaState:
        """Classify the karma state of the input text"""
        text_lower = text.lower()
        
        # Check each karma pattern
        karma_scores = {}
        
        for karma_type, patterns in self.karma_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            karma_scores[karma_type] = score
        
        # Return karma state with highest score, default to sattvic
        if karma_scores and max(karma_scores.values()) > 0:
            return max(karma_scores.items(), key=lambda x: x[1])[0]
        
        return KarmaState.SATTVIC
    
    def _extract_sanskrit_roots(self, text: str) -> List[str]:
        """Extract Sanskrit roots from the text"""
        found_roots = []
        
        # Check for Sanskrit roots in the text
        for root, metadata in self.ksml_roots.items():
            if root in text:
                found_roots.append(root)
        
        # Also check for common Sanskrit patterns
        sanskrit_patterns = [
            r'[अ-ह]+',  # Devanagari script
            r'[क-ह]+',  # Sanskrit consonants
        ]
        
        for pattern in sanskrit_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match not in found_roots:
                    # Check if it looks like a Sanskrit root
                    if self._is_sanskrit_root(match):
                        found_roots.append(match)
        
        return list(set(found_roots))  # Remove duplicates
    
    def _is_sanskrit_root(self, word: str) -> bool:
        """Check if a word looks like a Sanskrit root"""
        # Basic heuristic: Sanskrit roots are typically 2-4 characters
        # and contain specific consonant patterns
        if len(word) < 2 or len(word) > 6:
            return False
        
        # Check for Sanskrit consonant clusters
        sanskrit_consonants = 'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'
        
        consonant_count = sum(1 for char in word if char in sanskrit_consonants)
        
        # Sanskrit roots typically have 1-3 consonants
        return 1 <= consonant_count <= 3
    
    def _calculate_confidence(self, text: str, intent: IntentType, karma_state: KarmaState, 
                            semantic_roots: List[str]) -> float:
        """Calculate confidence score for the alignment"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on text characteristics
        if len(text) > 10:
            confidence += 0.1
        
        if semantic_roots:
            confidence += min(0.2, len(semantic_roots) * 0.05)
        
        # Boost confidence for clear intent patterns
        if intent != IntentType.CONVERSATIONAL:
            confidence += 0.1
        
        # Boost confidence for clear karma patterns
        if karma_state != KarmaState.SATTVIC:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the KSML aligner"""
        return {
            "initialized": self.initialized,
            "ksml_roots_count": len(self.ksml_roots),
            "intent_patterns_count": sum(len(patterns) for patterns in self.intent_patterns.values()),
            "karma_patterns_count": sum(len(patterns) for patterns in self.karma_patterns.values()),
            "supported_languages": list(self.language_keywords.keys())
        }
