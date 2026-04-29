"""
Knowledge Base Integration Module

This module handles integration with the Black Hole DB and provides
a complete Q&A pipeline: User → Multilingual LM → KB → LM → Response

Features:
- Async HTTP client for KB communication
- Query preprocessing and response post-processing
- Multilingual query handling
- Error handling and fallback responses
- Caching for frequent queries
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries supported by the KB"""
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    CULTURAL = "cultural"
    TRANSLATION = "translation"
    TECHNICAL = "technical"

@dataclass
class KBQuery:
    """Knowledge Base query structure"""
    text: str
    language: str
    query_type: QueryType
    context: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class KBResponse:
    """Knowledge Base response structure"""
    answer: str
    confidence: float
    sources: List[str]
    language: str
    query_type: QueryType
    processing_time: float
    cached: bool = False

class KBIntegrationError(Exception):
    """Custom exception for KB integration errors"""
    pass

class KnowledgeBaseClient:
    """Async client for Knowledge Base operations"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Dict] = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_cache_size = 1000
        
        # KB endpoints from settings
        self.kb_endpoint = settings.KB_ENDPOINT
        self.kb_timeout = settings.KB_TIMEOUT
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "average_response_time": 0.0
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.kb_timeout),
            headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _get_cache_key(self, query: KBQuery) -> str:
        """Generate cache key for query"""
        key_string = f"{query.text}_{query.language}_{query.query_type.value}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry["timestamp"] < self.cache_ttl

    def _clean_cache(self):
        """Remove old cache entries"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if current_time - entry["timestamp"] > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        # Limit cache size
        if len(self.cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["timestamp"])
            excess_count = len(self.cache) - self.max_cache_size
            for key, _ in sorted_items[:excess_count]:
                del self.cache[key]

    async def query_knowledge_base(self, query: KBQuery) -> KBResponse:
        """
        Query the knowledge base with multilingual support
        
        Args:
            query: KBQuery object with query details
            
        Returns:
            KBResponse object with answer and metadata
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            self.stats["cache_hits"] += 1
            cached_response = self.cache[cache_key]["response"]
            cached_response.cached = True
            cached_response.processing_time = time.time() - start_time
            logger.debug(f"Cache hit for query: {query.text[:50]}...")
            return cached_response
        
        self.stats["cache_misses"] += 1
        
        try:
            # If no KB endpoint configured, use mock response
            if not self.kb_endpoint:
                logger.info("No KB endpoint configured, using enhanced mock response")
                return await self._get_mock_response(query, start_time)
            
            # Check if KB endpoint is pointing to self (same host/port as current API)
            # This prevents infinite loops
            import socket
            current_host = socket.gethostbyname(socket.gethostname())
            if (self.kb_endpoint.startswith(f"http://127.0.0.1:") or 
                self.kb_endpoint.startswith(f"http://localhost:") or
                self.kb_endpoint.startswith(f"http://{current_host}:")):
                logger.warning(f"KB endpoint {self.kb_endpoint} appears to point to self, using mock response to prevent loop")
                return await self._get_mock_response(query, start_time)
            
            # Prepare request payload for multilingual-conversation endpoint
            payload = {
                "text": query.text,
                "language": query.language,
                "user_id": query.user_id,
                "session_id": query.session_id,
                "generate_response": True,  # Enable response generation
                "max_response_length": 256
            }
            
            # Make request to multilingual-conversation endpoint
            if not self.session:
                raise KBIntegrationError("HTTP session not initialized")
            
            logger.debug(f"Making KB request to: {self.kb_endpoint}/multilingual-conversation")
            async with self.session.post(
                f"{self.kb_endpoint}/multilingual-conversation",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    kb_response = self._parse_kb_response(data, query, start_time)
                    
                    # Cache the response
                    self.cache[cache_key] = {
                        "response": kb_response,
                        "timestamp": time.time()
                    }
                    self._clean_cache()
                    
                    return kb_response
                    
                else:
                    error_text = await response.text()
                    logger.error(f"KB API error {response.status}: {error_text}")
                    raise KBIntegrationError(f"KB API returned status {response.status}")
        
        except asyncio.TimeoutError:
            logger.error("KB request timed out")
            self.stats["errors"] += 1
            return await self._get_fallback_response(query, start_time, "timeout")
            
        except Exception as e:
            logger.error(f"KB integration error: {e}")
            self.stats["errors"] += 1
            return await self._get_fallback_response(query, start_time, str(e))

    def _parse_kb_response(self, data: Dict, query: KBQuery, start_time: float) -> KBResponse:
        """Parse response from multilingual-conversation endpoint"""
        processing_time = time.time() - start_time
        
        # Extract the KB answer (primary response from multilingual-conversation)
        kb_answer = data.get("kb_answer", data.get("answer", "No answer available"))
        
        # If there's a generated response, prefer it over KB answer
        generated_response = data.get("generated_response")
        if generated_response:
            kb_answer = generated_response
        
        # Extract metadata
        metadata = data.get("metadata", {})
        
        return KBResponse(
            answer=kb_answer,
            confidence=data.get("confidence", metadata.get("kb_confidence", 0.8)),
            sources=data.get("sources", metadata.get("kb_sources", ["Multilingual Conversation API"])),
            language=data.get("language", query.language),
            query_type=query.query_type,
            processing_time=processing_time
        )

    async def _get_mock_response(self, query: KBQuery, start_time: float) -> KBResponse:
        """Generate enhanced mock response for testing when KB is not available"""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Enhanced contextual mock responses based on query content
        mock_responses = {
            "assamese": {
                "greeting": "নমস্কাৰ! মই আপোনাক সহায় কৰিবলৈ ইয়াত আছোঁ।",
                "general": f"আপোনাৰ প্ৰশ্নৰ বিষয়ে: এইটো এটা বহুভাষিক ভাষা প্ৰক্ৰিয়াকৰণ প্ৰণালী। অধিক তথ্যৰ বাবে অনুগ্ৰহ কৰি বিশদভাৱে সোধক।",
                "educational": "এইটো এটা শৈক্ষিক প্ৰশ্ন। মই ভাষা প্ৰক্ৰিয়াকৰণ আৰু টোকেনাইজেশ্যনত সহায় কৰিব পাৰোঁ।",
                "technical": "এইটো এটা কাৰিকৰী প্ৰশ্ন। মই প্ৰগ্ৰামিং আৰু মেশ্বিন লাৰ্নিংত সহায় কৰিব পাৰোঁ।",
                "factual_geography": "ভাৰতৰ ৰাজধানী নতুন দিল্লী। এইটো এটা মক প্ৰতিক্ৰিয়া - সঠিক তথ্যৰ বাবে বিশ্বাসযোগ্য উৎসৰ পৰা পৰীক্ষা কৰক।"
            },
            "bengali": {
                "greeting": "নমস্কার! আমি আপনাকে সাহায্য করতে এখানে আছি।",
                "general": f"আপনার প্রশ্ন সম্পর্কে: এটি একটি বহুভাষিক ভাষা প্রক্রিয়াকরণ সিস্টেম। আরও তথ্যের জন্য অনুগ্রহ করে বিস্তারিত জিজ্ঞাসা করুন।",
                "educational": "এটি একটি শিক্ষামূলক প্রশ্ন। আমি ভাষা প্রক্রিয়াকরণ এবং টোকেনাইজেশনে সাহায্য করতে পারি।",
                "technical": "এটি একটি প্রযুক্তিগত প্রশ্ন। আমি প্রোগ্রামিং এবং মেশিন লার্নিংয়ে সাহায্য করতে পারি।",
                "factual_geography": "ভারতের রাজধানী নতুন দিল্লি। এটি একটি মক রেসপন্স - সঠিক তথ্যের জন্য বিশ্বস্ত উৎস থেকে যাচাই করুন।"
            },
            "bodo": {
                "greeting": "जोहार! आं नोंथाङा मदद खालामनो बेयाव दङ।",
                "general": f"नोंथाङा सोदोबनि बिसोमा: बे गोबां गोजौभाषा भाषा आयदानाय सिसटेम। गोबां बिबुंथिनि थाखाय अननानै बिबुंसिन।",
                "educational": "बे गोबां सिगां सोदोब। आं भाषा आयदानाय आरो टकनाइजेशननि मदद लाबो हायो।",
                "technical": "बे गोबां टेक्निकेल सोदोब। आं फुरगामिं आरो मेसिन लार्निंनि मदद लाबो हायो।",
                "factual_geography": "भारतनि राजधानी नयाँ दिल्ली। बे गोबां मक रेसपन्स - बाहागो बिबुंथिनि थाखाय सिगाङा सोरोखथिनिफ्राय थासिन।"
            },
            "english": {
                "greeting": "Hello! I'm here to help you with your questions.",
                "general": f"Regarding your question: This is a multilingual language processing system. For more specific information, please ask in detail.",
                "educational": "This is an educational question. I can assist with language processing and tokenization topics.",
                "technical": "This is a technical question. I can help with programming and machine learning topics.",
                "factual_geography": "The capital of India is New Delhi. This is a mock response - for accurate information, please consult reliable sources."
            },
            "gujarati": {
                "greeting": "નમસ્તે! હું તમારી સહાય માટે અહીં છું।",
                "general": f"તમારા પ્રશ્ન વિશે: આ એક બહુભાષી ભાષા પ્રોસેસિંગ સિસ્ટમ છે. વધુ માહિતી માટે કૃપા કરીને વિગતવાર પૂછો.",
                "educational": "આ એક શૈક્ષણિક પ્રશ્ન છે. હું ભાષા પ્રક્રિયા અને ટોકનાઇઝેશનમાં મદદ કરી શકું છું.",
                "technical": "આ એક ટેકનિકલ પ્રશ્ન છે. હું પ્રોગ્રામિંગ અને મશીન લર્નિંગમાં મદદ કરી શકું છું.",
                "factual_geography": "ભારતની રાજધાની નવી દિલ્હી છે. આ એક મોક પ્રતિસાદ છે - સચોટ માહિતી માટે વિશ્વસનીય સ્રોતો પાસેથી ચકાસો."
            },
            "hindi": {
                "greeting": "नमस्ते! मैं आपकी सहायता के लिए यहां हूं।",
                "general": f"आपके प्रश्न के बारे में: यह एक बहुभाषी भाषा प्रसंस्करण प्रणाली है। अधिक जानकारी के लिए कृपया विस्तार से पूछें।",
                "educational": "यह शैक्षिक प्रश्न है। मैं भाषा प्रसंस्करण और टोकनाइज़ेशन में सहायता कर सकता हूं।",
                "technical": "यह तकनीकी प्रश्न है। मैं प्रोग्रामिंग और मशीन लर्निंग में सहायता कर सकता हूं।",
                "factual_geography": "भारत की राजधानी नई दिल्ली है। यह एक मॉक प्रतिक्रिया है - सटीक जानकारी के लिए विश्वसनीय स्रोतों से जांच करें।"
            },
            "kannada": {
                "greeting": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ.",
                "general": f"ನಿಮ್ಮ ಪ್ರಶ್ನೆಯ ಬಗ್ಗೆ: ಇದು ಬಹುಭಾಷಾ ಭಾಷಾ ಸಂಸ್ಕರಣಾ ವ್ಯವಸ್ಥೆಯಾಗಿದೆ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ದಯವಿಟ್ಟು ವಿವರವಾಗಿ ಕೇಳಿ.",
                "educational": "ಇದು ಶೈಕ್ಷಣಿಕ ಪ್ರಶ್ನೆಯಾಗಿದೆ. ನಾನು ಭಾಷಾ ಸಂಸ್ಕರಣೆ ಮತ್ತು ಟೋಕನೈಸೇಶನ್‌ನಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ.",
                "technical": "ಇದು ತಾಂತ್ರಿಕ ಪ್ರಶ್ನೆಯಾಗಿದೆ. ನಾನು ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಮತ್ತು ಮೆಷಿನ್ ಲರ್ನಿಂಗ್‌ನಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ.",
                "factual_geography": "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ. ಇದು ಮಾಕ್ ಪ್ರತಿಕ್ರಿಯೆಯಾಗಿದೆ - ನಿಖರವಾದ ಮಾಹಿತಿಗಾಗಿ ವಿಶ್ವಾಸಾರ್ಹ ಮೂಲಗಳಿಂದ ಪರಿಶೀಲಿಸಿ."
            },
            "kashmiri": {
                "greeting": "سلام! می تہ مدد کرنہ خاطرہ ییہ چھس۔",
                "general": f"تہنز سوال کس بارس منز: یہ اکھ کثیر لسانی زبان پروسیسنگ سسٹم چھ۔ زیادہ معلومات خاطرہ مہربانی کرتھ تفصیل سان پرژھیو۔",
                "educational": "یہ اکھ تعلیمی سوال چھ۔ می زبان پروسیسنگ تہ ٹوکنائزیشن منز مدد کرتھ ہیکہ۔",
                "technical": "یہ اکھ تکنیکی سوال چھ۔ می پروگرامنگ تہ مشین لرننگ منز مدد کرتھ ہیکہ۔",
                "factual_geography": "ہندوستان کہ راج گڑھ نئی دلی چھ۔ یہ اکھ ماک رسپانس چھ - صحیح معلومات خاطرہ قابل اعتماد ذرائع پیٹھ پتہ لگایو۔"
            },
            "maithili": {
                "greeting": "नमस्कार! हम अहाँकेँ मदति करबाक लेल एतय छी।",
                "general": f"अहाँक प्रश्नक बारेमे: ई एकटा बहुभाषी भाषा प्रसंस्करण प्रणाली अछि। बेसी जानकारीक लेल कृपया विस्तारसँ पूछू।",
                "educational": "ई एकटा शैक्षिक प्रश्न अछि। हम भाषा प्रसंस्करण आ टोकनाइजेशनमे मदति कऽ सकैत छी।",
                "technical": "ई एकटा तकनीकी प्रश्न अछि। हम प्रोग्रामिंग आ मशीन लर्निंगमे मदति कऽ सकैत छी।",
                "factual_geography": "भारतक राजधानी नवका दिल्ली अछि। ई एकटा मॉक प्रतिक्रिया अछि - सही जानकारीक लेल विश्वसनीय स्रोतसँ जाँच करू।"
            },
            "malayalam": {
                "greeting": "നമസ്കാരം! ഞാൻ നിങ്ങളെ സഹായിക്കാൻ ഇവിടെയുണ്ട്.",
                "general": f"നിങ്ങളുടെ ചോദ്യത്തെക്കുറിച്ച്: ഇത് ഒരു ബഹുഭാഷാ ഭാഷാ പ്രോസസ്സിംഗ് സിസ്റ്റമാണ്. കൂടുതൽ വിവരങ്ങൾക്ക് ദയവായി വിശദമായി ചോദിക്കുക.",
                "educational": "ഇതൊരു വിദ്യാഭ്യാസപരമായ ചോദ്യമാണ്. ഭാഷാ പ്രോസസ്സിംഗിലും ടോക്കണൈസേഷനിലും എനിക്ക് സഹായിക്കാനാകും.",
                "technical": "ഇതൊരു സാങ്കേതിക ചോദ്യമാണ്. പ്രോഗ്രാമിംഗിലും മെഷീൻ ലേണിംഗിലും എനിക്ക് സഹായിക്കാനാകും.",
                "factual_geography": "ഇന്ത്യയുടെ തലസ്ഥാനം നവദില്ലിയാണ്. ഇതൊരു മോക്ക് പ്രതികരണമാണ് - കൃത്യമായ വിവരങ്ങൾക്ക് വിശ്വസനീയമായ സ്രോതസ്സുകളിൽ നിന്ന് പരിശോധിക്കുക."
            },
            "marathi": {
                "greeting": "नमस्कार! मी तुमची मदत करण्यासाठी येथे आहे।",
                "general": f"तुमच्या प्रश्नाबद्दल: ही एक बहुभाषी भाषा प्रसंस्करण प्रणाली आहे। अधिक माहितीसाठी कृपया तपशीलवार विचारा।",
                "educational": "हा शैक्षणिक प्रश्न आहे। मी भाषा प्रक्रिया आणि टोकनाइझेशनमध्ये मदत करू शकतो।",
                "technical": "हा तांत्रिक प्रश्न आहे। मी प्रोग्रामिंग आणि मशीन लर्निंगमध्ये मदत करू शकतो।",
                "factual_geography": "भारताची राजधानी नवी दिल्ली आहे। ही मॉक प्रतिक्रिया आहे - अचूक माहितीसाठी विश्वसनीय स्रोतांकडून तपासा।"
            },
            "meitei": {
                "greeting": "ꯈꯨꯔꯨꯝꯖꯔꯤ! ꯑꯩꯅꯥ ꯅꯍꯥꯀꯄꯨ ꯃꯇꯦꯡ ꯄꯥꯡꯅꯕꯥ ꯃꯐꯝ ꯑꯁꯤꯗꯥ ꯂꯩ।",
                "general": f"ꯅꯍꯥꯛꯀꯤ ꯋꯥꯍꯪꯒꯤ ꯃꯇꯥꯡꯗꯥ: ꯃꯁꯤ ꯃꯔꯤ ꯂꯩꯅꯕꯥ ꯂꯣꯟ ꯄ꯭ꯔꯣꯁꯦꯁꯤꯡ ꯁꯤꯁ꯭ꯇꯦꯝ ꯑꯃꯅꯤ। ꯍꯦꯟꯅꯥ ꯈꯉꯍꯟꯅꯕꯥ ꯃꯇꯝ ꯄꯤꯌꯨ ꯑꯗꯨꯒꯥ ꯍꯪꯖꯤꯟ ꯍꯪꯖꯤꯟ ꯍꯪꯕꯤꯌꯨ।",
                "educational": "ꯃꯁꯤ ꯃꯍꯩꯡ ꯇꯝꯕꯒꯤ ꯋꯥꯍꯪ ꯑꯃꯅꯤ। ꯑꯩꯅꯥ ꯂꯣꯟ ꯄ꯭ꯔꯣꯁꯦꯁꯤꯡ ꯑꯃꯁꯨꯡ ꯇꯣꯀꯦꯅꯥꯏꯖꯦꯁꯟꯗꯥ ꯃꯇꯦꯡ ꯄꯥꯡꯕꯥ ꯉꯃꯒꯅꯤ।",
                "technical": "ꯃꯁꯤ ꯇꯦꯛꯅꯤꯀꯦꯜ ꯋꯥꯍꯪ ꯑꯃꯅꯤ। ꯑꯩꯅꯥ ꯄ꯭ꯔꯣꯒ꯭ꯔꯥꯃꯤꯡ ꯑꯃꯁꯨꯡ ꯃꯦꯁꯤꯟ ꯂꯔꯅꯤꯡꯗꯥ ꯃꯇꯦꯡ ꯄꯥꯡꯕꯥ ꯉꯃꯒꯅꯤ।",
                "factual_geography": "ꯚꯥꯔꯇꯀꯤ ꯔꯥꯖꯓꯥꯅꯤ ꯅꯨꯋꯥ ꯗꯤꯜꯂꯤꯅꯤ। ꯃꯁꯤ ꯃꯣꯛ ꯔꯦꯁ꯭ꯄꯣꯟꯁ ꯑꯃꯅꯤ - ꯑꯆꯨꯝꯕꯥ ꯏꯅꯐꯣꯔꯃꯦꯁꯅꯒꯤꯗꯃꯛ ꯊꯥꯖꯕꯥ ꯌꯥꯕꯥ ꯁꯣꯔꯁꯁꯤꯡꯗꯒꯤ ꯆꯦꯛꯁꯤꯟꯅꯕꯤꯌꯨ।"
            },
            "nepali": {
                "greeting": "नमस्ते! म तपाईंलाई मद्दत गर्न यहाँ छु।",
                "general": f"तपाईंको प्रश्नको बारेमा: यो एक बहुभाषिक भाषा प्रशोधन प्रणाली हो। थप जानकारीको लागि कृपया विस्तृत रूपमा सोध्नुहोस्।",
                "educational": "यो एक शैक्षिक प्रश्न हो। म भाषा प्रशोधन र टोकनाइजेशनमा मद्दत गर्न सक्छु।",
                "technical": "यो एक प्राविधिक प्रश्न हो। म प्रोग्रामिङ र मेशिन लर्निङमा मद्दत गर्न सक्छु।",
                "factual_geography": "भारतको राजधानी नयाँ दिल्ली हो। यो एक मक प्रतिक्रिया हो - सही जानकारीको लागि विश्वसनीय स्रोतहरूबाट जाँच गर्नुहोस्।"
            },
            "odia": {
                "greeting": "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ଏଠାରେ ଅଛି।",
                "general": f"ଆପଣଙ୍କ ପ୍ରଶ୍ନ ବିଷୟରେ: ଏହା ଏକ ବହୁଭାଷୀ ଭାଷା ପ୍ରକ୍ରିୟାକରଣ ସିଷ୍ଟମ୍। ଅଧିକ ସୂଚନା ପାଇଁ ଦୟାକରି ବିସ୍ତୃତ ଭାବରେ ପଚାରନ୍ତୁ।",
                "educational": "ଏହା ଏକ ଶିକ୍ଷାଗତ ପ୍ରଶ୍ନ। ମୁଁ ଭାଷା ପ୍ରକ୍ରିୟାକରଣ ଏବଂ ଟୋକେନାଇଜେସନରେ ସାହାଯ୍ୟ କରିପାରିବି।",
                "technical": "ଏହା ଏକ ବୈଷୟିକ ପ୍ରଶ୍ନ। ମୁଁ ପ୍ରୋଗ୍ରାମିଂ ଏବଂ ମେସିନ୍ ଲର୍ଣ୍ଣିଂରେ ସାହାଯ୍ୟ କରିପାରିବି।",
                "factual_geography": "ଭାରତର ରାଜଧାନୀ ନୂଆଦିଲ୍ଲୀ। ଏହା ଏକ ମକ୍ ପ୍ରତିକ୍ରିୟା - ସଠିକ୍ ସୂଚନା ପାଇଁ ବିଶ୍ୱସ୍ତ ଉତ୍ସରୁ ଯାଞ୍ଚ କରନ୍ତୁ।"
            },
            "punjabi": {
                "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਮਦਦ ਕਰਨ ਲਈ ਇੱਥੇ ਹਾਂ।",
                "general": f"ਤੁਹਾਡੇ ਸਵਾਲ ਬਾਰੇ: ਇਹ ਇੱਕ ਬਹੁਭਾਸ਼ਾਈ ਭਾਸ਼ਾ ਪ੍ਰੋਸੈਸਿੰਗ ਸਿਸਟਮ ਹੈ। ਵਧੇਰੇ ਜਾਣਕਾਰੀ ਲਈ ਕਿਰਪਾ ਕਰਕੇ ਵਿਸਥਾਰ ਨਾਲ ਪੁੱਛੋ।",
                "educational": "ਇਹ ਇੱਕ ਵਿਦਿਅਕ ਸਵਾਲ ਹੈ। ਮੈਂ ਭਾਸ਼ਾ ਪ੍ਰੋਸੈਸਿੰਗ ਅਤੇ ਟੋਕਨਾਈਜ਼ੇਸ਼ਨ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ।",
                "technical": "ਇਹ ਇੱਕ ਤਕਨੀਕੀ ਸਵਾਲ ਹੈ। ਮੈਂ ਪ੍ਰੋਗਰਾਮਿੰਗ ਅਤੇ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ।",
                "factual_geography": "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਨਵੀਂ ਦਿੱਲੀ ਹੈ। ਇਹ ਇੱਕ ਮੌਕ ਜਵਾਬ ਹੈ - ਸਹੀ ਜਾਣਕਾਰੀ ਲਈ ਭਰੋਸੇਯੋਗ ਸਰੋਤਾਂ ਤੋਂ ਜਾਂਚ ਕਰੋ।"
            },
            "sanskrit": {
                "greeting": "नमस्कारः! अहं भवतः सहायतार्थं अत्र अस्मि।",
                "general": f"भवतः प्रश्नस्य विषये: एतत् बहुभाषीय भाषा प्रसंस्करण प्रणाली अस्ति। विशिष्टसूचनायै कृपया विस्तृतं प्रश्नं पृच्छतु।",
                "educational": "एषा शैक्षिका प्रश्ना अस्ति। अहं भाषा प्रसंस्करणे सहायतां कर्तुं शक्नोमि।",
                "technical": "एषा तकनीकी प्रश्ना अस्ति। अहं प्रोग्रामिंगे सहायतां कर्तुं शक्नोमि।",
                "factual_geography": "भारतस्य राजधानी नवदिल्ली अस्ति। एषा मॉक प्रतिक्रिया अस्ति - सटीकजानकार्यर्थं विश्वसनीयस्रोतानां संदर्भः करणीयः।"
            },
            "santali": {
                "greeting": "ᱡᱚᱦᱟᱨ! ᱤᱧ ᱟᱢᱟᱜ ᱜᱚᱲᱚ ᱞᱟᱹᱜᱤᱫ ᱱᱚᱰᱮ ᱢᱮᱱᱟᱹᱧ᱾",
                "general": f"ᱟᱢᱟᱜ ᱠᱩᱠᱢᱩ ᱵᱟᱵᱚᱛ: ᱱᱚᱶᱟ ᱫᱚ ᱢᱤᱫ ᱟᱹᱰᱤ ᱯᱟᱹᱨᱥᱤ ᱯᱟᱹᱨᱥᱤ ᱠᱟᱹᱢᱤ ᱦᱚᱨᱟ ᱠᱟᱱᱟ᱾ ᱰᱷᱮᱨ ᱵᱟᱰᱟᱭ ᱞᱟᱹᱜᱤᱫ ᱫᱚᱭᱟᱠᱟᱛᱮ ᱵᱤᱥᱛᱤ ᱠᱩᱠᱢᱩᱢᱮ᱾",
                "educational": "ᱱᱚᱶᱟ ᱫᱚ ᱢᱤᱫ ᱥᱤᱠᱷᱱᱟᱹᱛ ᱠᱩᱠᱢᱩ ᱠᱟᱱᱟ᱾ ᱤᱧ ᱯᱟᱹᱨᱥᱤ ᱠᱟᱹᱢᱤ ᱟᱨ ᱴᱚᱠᱮᱱᱟᱭᱤᱡᱮᱥᱚᱱ ᱨᱮ ᱜᱚᱲᱚ ᱫᱟᱲᱮᱭᱟᱜ-ᱟ᱾",
                "technical": "ᱱᱚᱶᱟ ᱫᱚ ᱢᱤᱫ ᱴᱮᱠᱱᱤᱠᱟᱞ ᱠᱩᱠᱢᱩ ᱠᱟᱱᱟ᱾ ᱤᱧ ᱯᱨᱚᱜᱨᱟᱢᱤᱝ ᱟᱨ ᱢᱮᱥᱤᱱ ᱞᱟᱨᱱᱤᱝ ᱨᱮ ᱜᱚᱲᱚ ᱫᱟᱲᱮᱭᱟᱜ-ᱟ᱾",
                "factual_geography": "ᱵᱷᱟᱨᱚᱛ ᱨᱮᱭᱟᱜ ᱨᱟᱡᱽᱫᱷᱟᱱᱤ ᱫᱚ ᱱᱟᱣᱟ ᱫᱤᱞᱤ ᱠᱟᱱᱟ᱾ ᱱᱚᱶᱟ ᱫᱚ ᱢᱚᱠ ᱛᱮᱞᱟ ᱠᱟᱱᱟ - ᱴᱷᱤᱠ ᱵᱟᱰᱟᱭ ᱞᱟᱹᱜᱤᱫ ᱵᱷᱚᱨᱚᱥᱟᱭᱚᱜ ᱡᱟᱜᱟ ᱠᱷᱚᱱ ᱧᱮᱞᱢᱮ᱾"
            },
            "sindhi": {
                "greeting": "سلام! مان توهان جي مدد ڪرڻ لاءِ هتي آهيان.",
                "general": f"توهان جي سوال بابت: هي هڪ گهڻ لساني ٻولي پروسيسنگ سسٽم آهي. وڌيڪ معلومات لاءِ مهرباني ڪري تفصيل سان پڇو.",
                "educational": "هي هڪ تعليمي سوال آهي. مان ٻولي پروسيسنگ ۽ ٽوڪنائيزيشن ۾ مدد ڪري سگهان ٿو.",
                "technical": "هي هڪ ٽيڪنيڪل سوال آهي. مان پروگرامنگ ۽ مشين لرننگ ۾ مدد ڪري سگهان ٿو.",
                "factual_geography": "هندستان جو راڄڌاني نئون دهلي آهي. هي هڪ ماڪ جواب آهي - صحيح معلومات لاءِ قابل اعتماد ذريعن کان چيڪ ڪريو."
            },
            "tamil": {
                "greeting": "வணக்கம்! நான் உங்களுக்கு உதவ இங்கே இருக்கிறேன்.",
                "general": f"உங்கள் கேள்வி பற்றி: இது பல மொழி செயலாக்க அமைப்பு. மேலும் தகவலுக்கு தயவுசெய்து விரிவாக கேளுங்கள்.",
                "educational": "இது ஒரு கல்வி சார்ந்த கேள்வி. மொழி செயலாக்கம் மற்றும் டோக்கனைசேஷனில் என்னால் உதவ முடியும்.",
                "technical": "இது ஒரு தொழில்நுட்ப கேள்வி. நிரலாக்கம் மற்றும் இயந்திர கற்றலில் என்னால் உதவ முடியும்.",
                "factual_geography": "இந்தியாவின் தலைநகரம் புதுதில்லி. இது ஒரு பொய் பதில் - சரியான தகவலுக்கு நம்பகமான ஆதாரங்களிலிருந்து சரிபார்க்கவும்."
            },
            "telugu": {
                "greeting": "నమస్కారం! నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.",
                "general": f"మీ ప్రశ్న గురించి: ఇది బహుభాషా భాషా ప్రాసెసింగ్ వ్యవస్థ. మరింత సమాచారం కోసం దయచేసి వివరంగా అడగండి.",
                "educational": "ఇది విద్యా సంబంధిత ప్రశ్న. భాషా ప్రాసెసింగ్ మరియు టోకనైజేషన్‌లో నేను సహాయం చేయగలను.",
                "technical": "ఇది సాంకేతిక ప్రశ్న. ప్రోగ్రామింగ్ మరియు మెషిన్ లర్నింగ్‌లో నేను సహాయం చేయగలను.",
                "factual_geography": "భారతదేశ రాజధాని న్యూఢిల్లీ. ఇది మాక్ ప్రతిస్పందన - ఖచ్చితమైన సమాచారం కోసం విశ్వసనీయ మూలాల నుండి తనిఖీ చేయండి."
            },
            "urdu": {
                "greeting": "السلام علیکم! میں آپ کی مدد کرنے کے لیے یہاں موجود ہوں۔",
                "general": f"آپ کے سوال کے بارے میں: یہ ایک کثیر لسانی زبان کی پروسیسنگ سسٹم ہے۔ مزید معلومات کے لیے براہ کرم تفصیل سے پوچھیں۔",
                "educational": "یہ ایک تعلیمی سوال ہے۔ میں زبان کی پروسیسنگ اور ٹوکنائزیشن میں مدد کر سکتا ہوں۔",
                "technical": "یہ ایک تکنیکی سوال ہے۔ میں پروگرامنگ اور مشین لرننگ میں مدد کر سکتا ہوں۔",
                "factual_geography": "ہندوستان کا دارالحکومت نئی دہلی ہے۔ یہ ایک نمونہ جواب ہے - درست معلومات کے لیے قابل اعتماد ذرائع سے چیک کریں۔"
            }
        }
        
        # Determine response type based on query content and type
        query_lower = query.text.lower()
        
        # Determine response type based on query content and type
        if any(word in query_lower for word in ["hello", "hi", "namaste", "नमस्ते", "नमस्कार"]):
            response_key = "greeting"
        elif query.query_type == QueryType.EDUCATIONAL:
            response_key = "educational"
        elif query.query_type == QueryType.FACTUAL:
            # Handle common factual questions with appropriate responses
            geography_keywords = ["capital", "भारत", "india", "delhi", "mumbai", "country", "city", "राजधानी", "france", "paris", "london", "england"]
            if any(word in query_lower for word in geography_keywords):
                response_key = "factual_geography"
            else:
                response_key = "general"
        elif query.query_type == QueryType.TECHNICAL or any(word in query_lower for word in ["api", "code", "programming", "technical", "model", "token", "python", "javascript", "database"]):
            response_key = "technical"
        else:
            response_key = "general"
        
        lang_responses = mock_responses.get(query.language, mock_responses["english"])
        mock_answer = lang_responses[response_key]
        
        return KBResponse(
            answer=mock_answer,
            confidence=0.8,
            sources=["Multilingual Tokenization API - Mock Response"],
            language=query.language,
            query_type=query.query_type,
            processing_time=time.time() - start_time
        )

    async def _get_fallback_response(self, query: KBQuery, start_time: float, error_reason: str) -> KBResponse:
        """Generate fallback response when KB is unavailable"""
        
        fallback_messages = {
            "assamese": "ক্ষমা কৰিব, মই এতিয়া আপোনাৰ প্ৰশ্নৰ উত্তৰ দিব নোৱাৰোঁ। অনুগ্ৰহ কৰি পাছত চেষ্টা কৰক।",
            "bengali": "ক্ষমা করবেন, আমি এখন আপনার প্রশ্নের উত্তর দিতে পারছি না। দয়া করে পরে চেষ্টা করুন।",
            "bodo": "माफ खालाम, आं दानो नोंथाङा सोदोबनि फैगोन बायदि हानो नाङा। अननानै फिन होनना थासिन।",
            "english": "I apologize, but I'm unable to answer your question right now. Please try again later.",
            "gujarati": "માફ કરશો, હું અત્યારે તમારા પ્રશ્નનો જવાબ આપી શકતો નથી. કૃપા કરીને પછીથી પ્રયાસ કરો.",
            "hindi": "क्षमा करें, मैं अभी आपके प्रश्न का उत्तर नहीं दे सकता। कृपया बाद में कोशिश करें।",
            "kannada": "ಕ್ಷಮಿಸಿ, ನಾನು ಈಗ ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
            "kashmiri": "معذرت، می وننہ تہنز سوال کا جواب دتھ نہ ہیکہ۔ مہربانی کرتھ دوبارہ کوشش کریو۔",
            "maithili": "क्षमा करू, हम अहिना अहाँक प्रश्नक उत्तर नहि दऽ सकैत छी। कृपया बाद मे प्रयास करू।",
            "malayalam": "ക്ഷമിക്കണം, എനിക്ക് ഇപ്പോൾ നിങ്ങളുടെ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ കഴിയുന്നില്ല. ദയവായി പിന്നീട് വീണ്ടും ശ്രമിക്കുക.",
            "marathi": "माफ करा, मी सध्या तुमच्या प्रश्नाचे उत्तर देऊ शकत नाही. कृपया नंतर प्रयत्न करा.",
            "meitei": "ꯀꯥꯡꯂꯨꯞꯀꯅꯨ, ꯑꯩꯅꯥ ꯍꯧꯖꯤꯛ ꯅꯍꯥꯛꯀꯤ ꯋꯥꯍꯪꯒꯤ ꯄꯥꯎꯈꯨꯝ ꯄꯤꯕꯥ ꯉꯃꯗꯦ। ꯃꯇꯝ ꯑꯃꯗꯥ ꯑꯃꯨꯛ ꯍꯟꯅꯥ ꯍꯣꯠꯅꯧ।",
            "nepali": "माफ गर्नुहोस्, म अहिले तपाईंको प्रश्नको उत्तर दिन सक्दिन। कृपया पछि प्रयास गर्नुहोस्।",
            "odia": "କ୍ଷମା କରନ୍ତୁ, ମୁଁ ବର୍ତ୍ତମାନ ଆପଣଙ୍କ ପ୍ରଶ୍ନର ଉତ୍ତର ଦେଇ ପାରୁନାହିଁ। ଦୟାକରି ପରେ ପ୍ରୟାସ କରନ୍ତୁ।",
            "punjabi": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਹੁਣ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਨਹੀਂ ਦੇ ਸਕਦਾ। ਕਿਰਪਾ ਕਰਕੇ ਬਾਅਦ ਵਿੱਚ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
            "sanskrit": "क्षम्यताम्, अहं सम्प्रति भवतः प्रश्नस्य उत्तरं दातुं न शक्नोमि। कृपया पुनः प्रयासं कुर्वन्तु।",
            "santali": "ᱢᱟᱯᱷ ᱢᱮ, ᱤᱧ ᱱᱤᱛᱚᱜ ᱟᱢᱟᱜ ᱠᱩᱠᱢᱩ ᱨᱮᱭᱟᱜ ᱛᱮᱞᱟ ᱵᱟᱝ ᱮᱢ ᱫᱟᱲᱮᱭᱟᱜ-ᱟ᱾ ᱫᱚᱭᱟᱠᱟᱛᱮ ᱛᱟᱭᱚᱢ ᱛᱮ ᱠᱩᱨᱩᱢᱩᱴᱩᱢᱮ᱾",
            "sindhi": "معذرت، مان ھن وقت توهان جي سوال جو جواب نه ٿو ڏئي سگهان. مهرباني ڪري پوءِ ڪوشش ڪريو.",
            "tamil": "மன்னிக்கவும், என்னால் இப்போது உங்கள் கேள்விக்கு பதிலளிக்க முடியவில்லை. தயவுசெய்து பிறகு முயற்சி செய்யுங்கள்.",
            "telugu": "క్షమించండి, నేను ప్రస్తుతం మీ ప్రశ్నకు సమాధానం ఇవ్వలేను. దయచేసి తర్వాత ప్రయత్నించండి.",
            "urdu": "معذرت، میں ابھی آپ کے سوال کا جواب نہیں دے سکتا۔ براہ کرم بعد میں کوشش کریں۔"
        }
        
        fallback_answer = fallback_messages.get(query.language, fallback_messages["english"])
        
        return KBResponse(
            answer=fallback_answer,
            confidence=0.1,
            sources=["Fallback System"],
            language=query.language,
            query_type=query.query_type,
            processing_time=time.time() - start_time
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["total_queries"], 1)) * 100
        
        return {
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "errors": self.stats["errors"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "error_rate": (self.stats["errors"] / max(self.stats["total_queries"], 1)) * 100
        }

class QueryClassifier:
    """Classify queries to determine appropriate KB routing"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.FACTUAL: [
                "what is", "who is", "when", "where", "how many",
                "क्या है", "कौन है", "कब", "कहाँ", "कितने",
                "किम् अस्ति", "कः अस्ति", "कदा", "कुत्र",
                "काय आहे", "कोण आहे", "केव्हा", "कुठे"
            ],
            QueryType.CONVERSATIONAL: [
                "hello", "hi", "how are you", "good morning",
                "नमस्ते", "हैलो", "कैसे हैं", "सुप्रभात",
                "नमस्कार", "कथम्", "सुप्रभातम्",
                "नमस्कार", "कसे आहात", "सुप्रभात"
            ],
            QueryType.EDUCATIONAL: [
                "explain", "teach", "learn", "study", "definition",
                "समझाओ", "सिखाओ", "सीखना", "अध्ययन", "परिभाषा",
                "व्याख्या", "शिक्षा", "अध्ययनम्", "ज्ञानम्",
                "समजावून सांगा", "शिकवा", "शिकणे", "अभ्यास"
            ],
            QueryType.CULTURAL: [
                "tradition", "festival", "culture", "history", "mythology",
                "परंपरा", "त्योहार", "संस्कृति", "इतिहास", "पुराण",
                "संस्कृतिः", "उत्सवः", "इतिहासः", "पुराणम्",
                "परंपरा", "सण", "संस्कृती", "इतिहास"
            ],
            QueryType.TECHNICAL: [
                "api", "code", "programming", "technical", "model", "token", "python", "javascript", "database",
                "प्रोग्रामिंग", "कोड", "तकनीकी", "मॉडल", "टोकन",
                "प्रोग्रामिंग", "कोड", "तांत्रिक", "मॉडेल"
            ]
        }

    def classify_query(self, text: str, language: str) -> QueryType:
        """
        Classify the query type based on content patterns
        
        Args:
            text: Query text
            language: Detected language
            
        Returns:
            QueryType enum value
        """
        text_lower = text.lower()
        
        # Score each query type
        scores = {query_type: 0 for query_type in QueryType}
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    scores[query_type] += 1
        
        # Return the highest scoring type, default to CONVERSATIONAL
        if max(scores.values()) == 0:
            return QueryType.CONVERSATIONAL
        
        return max(scores, key=scores.get)

class MultilingualQAOrchestrator:
    """
    Main orchestrator for the multilingual Q&A pipeline
    Handles: User → Multilingual LM → KB → LM → Response
    """
    
    def __init__(self):
        self.kb_client = None
        self.classifier = QueryClassifier()
        self.conversation_history = {}  # Simple conversation tracking
        
    async def process_multilingual_query(
        self, 
        query_text: str, 
        detected_language: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a multilingual query through the complete pipeline
        
        Args:
            query_text: User's query
            detected_language: Language detected by the API
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional additional context
            
        Returns:
            Tuple of (final_response, metadata)
        """
        
        start_time = time.time()
        
        # Step 1: Classify the query
        query_type = self.classifier.classify_query(query_text, detected_language)
        
        # Step 2: Create KB query object
        kb_query = KBQuery(
            text=query_text,
            language=detected_language,
            query_type=query_type,
            context=self._get_conversation_context(session_id) if session_id else context,
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 3: Query knowledge base
        async with KnowledgeBaseClient() as kb_client:
            kb_response = await kb_client.query_knowledge_base(kb_query)
        
        # Step 4: Post-process response (could involve another LM call here)
        final_response = await self._post_process_response(
            kb_response, query_text, detected_language
        )
        
        # Step 5: Update conversation history
        if session_id:
            self._update_conversation_history(session_id, query_text, final_response, detected_language)
        
        # Prepare metadata
        metadata = {
            "query_type": query_type.value,
            "kb_confidence": kb_response.confidence,
            "kb_sources": kb_response.sources,
            "processing_time": time.time() - start_time,
            "kb_processing_time": kb_response.processing_time,
            "cached_response": kb_response.cached,
            "language": detected_language
        }
        
        return final_response, metadata

    def _get_conversation_context(self, session_id: str) -> Optional[str]:
        """Get recent conversation context for better responses"""
        if session_id in self.conversation_history:
            history = self.conversation_history[session_id]
            # Return last 3 exchanges as context
            recent_exchanges = history[-6:]  # Last 3 Q&A pairs
            context_parts = []
            for i in range(0, len(recent_exchanges), 2):
                if i + 1 < len(recent_exchanges):
                    context_parts.append(f"Q: {recent_exchanges[i]}")
                    context_parts.append(f"A: {recent_exchanges[i+1]}")
            return "\n".join(context_parts)
        return None

    def _update_conversation_history(self, session_id: str, query: str, response: str, language: str):
        """Update conversation history for context"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].extend([query, response])
        
        # Keep only last 20 exchanges (40 items)
        if len(self.conversation_history[session_id]) > 40:
            self.conversation_history[session_id] = self.conversation_history[session_id][-40:]

    async def _post_process_response(
        self, 
        kb_response: KBResponse, 
        original_query: str, 
        target_language: str
    ) -> str:
        """
        Post-process KB response for better user experience
        This could involve:
        - Language consistency checks
        - Response formatting
        - Additional context injection
        """
        
        response = kb_response.answer
        
        # Add source attribution if available and confidence is high
        if kb_response.sources and kb_response.confidence > 0.7:
            source_text = {
                "assamese": "উৎস",
                "bengali": "উৎস",
                "bodo": "सोरोख",
                "english": "Source",
                "gujarati": "સ્રોત",
                "hindi": "स्रोत",
                "kannada": "ಮೂಲ",
                "kashmiri": "ذریعہ",
                "maithili": "स्रोत",
                "malayalam": "ഉറവിടം",
                "marathi": "स्रोत",
                "meitei": "ꯁꯣꯔꯁ",
                "nepali": "स्रोत",
                "odia": "ଉତ୍ସ",
                "punjabi": "ਸਰੋਤ",
                "sanskrit": "स्रोतः",
                "santali": "ᱥᱚᱨᱚᱠᱷ",
                "sindhi": "ذريعو",
                "tamil": "ஆதாரம்",
                "telugu": "మూలం",
                "urdu": "ذریعہ"
            }.get(target_language, "Source")
            
            if len(kb_response.sources) == 1:
                response += f"\n\n({source_text}: {kb_response.sources[0]})"
            elif len(kb_response.sources) > 1:
                response += f"\n\n({source_text}: {', '.join(kb_response.sources[:3])})"
        
        # Add confidence indicator for low-confidence responses
        if kb_response.confidence < 0.5:
            uncertainty_text = {
                "assamese": "মই এই উত্তৰৰ বিষয়ে সম্পূৰ্ণ নিশ্চিত নহয়।",
                "bengali": "আমি এই উত্তর সম্পর্কে সম্পূর্ণ নিশ্চিত নই।",
                "bodo": "आं बे फैगोनबो बिसोमा गासै बिसोराय दङ।",
                "english": "I'm not entirely confident about this answer.",
                "gujarati": "મને આ જવાબ વિશે સંપૂર્ણ વિશ્વાસ નથી.",
                "hindi": "मुझे इस उत्तर के बारे में पूरा भरोसा नहीं है।",
                "kannada": "ಈ ಉತ್ತರದ ಬಗ್ಗೆ ನನಗೆ ಸಂಪೂರ್ಣ ವಿಶ್ವಾಸವಿಲ್ಲ.",
                "kashmiri": "می یتھ جواب کس متعلق مکمل یقین نہ چھ۔",
                "maithili": "हमरा ई उत्तरक बारेमे पूर्ण विश्वास नहि अछि।",
                "malayalam": "ഈ ഉത്തരത്തെക്കുറിച്ച് എനിക്ക് പൂർണ്ണ ആത്മവിശ്വാസമില്ല.",
                "marathi": "या उत्तराबद्दल मला पूर्ण खात्री नाही.",
                "meitei": "ꯄꯥꯎꯈꯨꯝ ꯑꯁꯤꯒꯤ ꯃꯇꯥꯡꯗꯥ ꯑꯩꯒꯤ ꯃꯄꯨꯡ ꯐꯥꯅꯥ ꯊꯥꯖꯕꯥ ꯂꯩꯇꯦ।",
                "nepali": "मलाई यो जवाफको बारेमा पूर्ण विश्वास छैन।",
                "odia": "ମୁଁ ଏହି ଉତ୍ତର ବିଷୟରେ ସମ୍ପୂର୍ଣ୍ଣ ଆତ୍ମବିଶ୍ୱାସୀ ନୁହେଁ।",
                "punjabi": "ਮੈਨੂੰ ਇਸ ਜਵਾਬ ਬਾਰੇ ਪੂਰਾ ਭਰੋਸਾ ਨਹੀਂ ਹੈ।",
                "sanskrit": "अस्य उत्तरस्य विषये मम पूर्णं विश्वासः नास्ति।",
                "santali": "ᱤᱧ ᱱᱚᱶᱟ ᱛᱮᱞᱟ ᱵᱟᱵᱚᱛ ᱯᱩᱨᱟᱹ ᱵᱷᱚᱨᱚᱥᱟ ᱵᱟᱝ ᱢᱮᱱᱟᱜ-ᱟ᱾",
                "sindhi": "مون کي هن جواب بابت مڪمل يقين ناهي.",
                "tamil": "இந்த பதிலைப் பற்றி எனக்கு முழு நம்பிக்கை இல்லை.",
                "telugu": "ఈ సమాధానం గురించి నాకు పూర్తి నమ్మకం లేదు.",
                "urdu": "مجھے اس جواب کے بارے میں پورا یقین نہیں ہے۔"
            }.get(target_language, "I'm not entirely confident about this answer.")
            
            response = f"{response}\n\n({uncertainty_text})"
        
        return response

# Global orchestrator instance
qa_orchestrator = MultilingualQAOrchestrator()

async def process_qa_query(
    query_text: str, 
    detected_language: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to process Q&A queries
    
    Usage in FastAPI:
        response, metadata = await process_qa_query(
            request.text, 
            detected_language,
            user_id="user123",
            session_id="session456"
        )
    """
    return await qa_orchestrator.process_multilingual_query(
        query_text, detected_language, user_id, session_id
    )

def get_kb_stats() -> Dict[str, Any]:
    """Get KB integration statistics"""
    # This would typically be called from the API's /stats endpoint
    return {
        "conversation_sessions": len(qa_orchestrator.conversation_history),
        "total_exchanges": sum(len(history) for history in qa_orchestrator.conversation_history.values())
    }