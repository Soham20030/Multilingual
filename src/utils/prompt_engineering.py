"""
Comprehensive Prompt Engineering Module for 21 Indian Languages
================================================================

This module provides structured prompts for:
1. Text Generation
2. Question Answering
3. Conversation
4. Task-specific instructions

Supports all 21 languages with culturally appropriate templates.
"""

from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PromptTemplates:
    """
    Centralized prompt templates for all 21 supported languages.
    Each language has templates for different use cases.
    """
    
    # ========== INSTRUCTION PROMPTS FOR TEXT GENERATION ==========
    GENERATION_PROMPTS = {
        "hindi": {
            "instruction": "निर्देश: {instruction}\n\nइनपुट: {text}\n\nउत्तर:",
            "simple": "{text}\n\nकृपया केवल हिंदी में उपरोक्त विषय पर विस्तार से लिखें। विषय पर केंद्रित रहें और अन्य भाषाओं का उपयोग न करें:",
            "continue": "{text}",
        },
        "sanskrit": {
            "instruction": "निर्देशः: {instruction}\n\nप्रविष्टिः: {text}\n\nउत्तरम्:",
            "simple": "{text}\n\nकृपया उपरोक्तविषये विस्तृतं लिखतु:",
            "continue": "{text}",
        },
        "tamil": {
            "instruction": "அறிவுறுத்தல்: {instruction}\n\nஉள்ளீடு: {text}\n\nபதில்:",
            "simple": "{text}\n\nதயவுசெய்து தமிழில் மட்டும் மேற்கண்ட தலைப்பில் விரிவாக எழுதுங்கள். தலைப்பில் கவனம் செலுத்துங்கள் மற்றும் பிற மொழிகளைப் பயன்படுத்த வேண்டாம்:",
            "continue": "{text}",
        },
        "telugu": {
            "instruction": "సూచన: {instruction}\n\nఇన్‌పుట్: {text}\n\nసమాధానం:",
            "simple": "{text}\n\nదయచేసి పై అంశంపై విస్తృతంగా రాయండి:",
            "continue": "{text}",
        },
        "bengali": {
            "instruction": "নির্দেশনা: {instruction}\n\nইনপুট: {text}\n\nউত্তর:",
            "simple": "{text}\n\nঅনুগ্রহ করে শুধুমাত্র বাংলায় উপরের বিষয়ে বিস্তারিত লিখুন। বিষয়ে মনোনিবেশ করুন এবং অন্যান্য ভাষা ব্যবহার করবেন না:",
            "continue": "{text}",
        },
        "marathi": {
            "instruction": "सूचना: {instruction}\n\nइनपुट: {text}\n\nउत्तर:",
            "simple": "{text}\n\nकृपया वरील विषयावर तपशीलवार लिहा:",
            "continue": "{text}",
        },
        "gujarati": {
            "instruction": "સૂચના: {instruction}\n\nઇનપુટ: {text}\n\nજવાબ:",
            "simple": "{text}\n\nકૃપા કરીને ઉપરોક્ત વિષય પર વિગતવાર લખો:",
            "continue": "{text}",
        },
        "kannada": {
            "instruction": "ಸೂಚನೆ: {instruction}\n\nಇನ್‌ಪುಟ್: {text}\n\nಉತ್ತರ:",
            "simple": "{text}\n\nದಯವಿಟ್ಟು ಮೇಲಿನ ವಿಷಯದ ಬಗ್ಗೆ ವಿವರವಾಗಿ ಬರೆಯಿರಿ:",
            "continue": "{text}",
        },
        "malayalam": {
            "instruction": "നിർദ്ദേശം: {instruction}\n\nഇൻപുട്ട്: {text}\n\nഉത്തരം:",
            "simple": "{text}\n\nദയവായി മേൽപ്പറഞ്ഞ വിഷയത്തെക്കുറിച്ച് വിശദമായി എഴുതുക:",
            "continue": "{text}",
        },
        "punjabi": {
            "instruction": "ਹਦਾਇਤ: {instruction}\n\nਇਨਪੁੱਟ: {text}\n\nਜਵਾਬ:",
            "simple": "{text}\n\nਕਿਰਪਾ ਕਰਕੇ ਉਪਰੋਕਤ ਵਿਸ਼ੇ 'ਤੇ ਵਿਸਥਾਰ ਨਾਲ ਲਿਖੋ:",
            "continue": "{text}",
        },
        "odia": {
            "instruction": "ନିର୍ଦ୍ଦେଶ: {instruction}\n\nଇନପୁଟ୍: {text}\n\nଉତ୍ତର:",
            "simple": "{text}\n\nଦୟାକରି ଉପରୋକ୍ତ ବିଷୟ ଉପରେ ବିସ୍ତୃତ ଭାବରେ ଲେଖନ୍ତୁ:",
            "continue": "{text}",
        },
        "assamese": {
            "instruction": "নিৰ্দেশনা: {instruction}\n\nইনপুট: {text}\n\nউত্তৰ:",
            "simple": "{text}\n\nঅনুগ্ৰহ কৰি উপৰোক্ত বিষয়ৰ ওপৰত বিস্তাৰিতভাৱে লিখক:",
            "continue": "{text}",
        },
        "urdu": {
            "instruction": "ہدایت: {instruction}\n\nان پٹ: {text}\n\nجواب:",
            "simple": "{text}\n\nبراہ کرم مذکورہ بالا موضوع پر تفصیل سے لکھیں:",
            "continue": "{text}",
        },
        "kashmiri": {
            "instruction": "ہدایت: {instruction}\n\nان پٹ: {text}\n\nجواب:",
            "simple": "{text}\n\nمہربأنؠ کٔرتھ موضوع پؠٹھ تفصیلی لیکھو:",
            "continue": "{text}",
        },
        "nepali": {
            "instruction": "निर्देश: {instruction}\n\nइनपुट: {text}\n\nउत्तर:",
            "simple": "{text}\n\nकृपया माथिको विषयमा विस्तृत रूपमा लेख्नुहोस्:",
            "continue": "{text}",
        },
        "sindhi": {
            "instruction": "هدايت: {instruction}\n\nان پٽ: {text}\n\nجواب:",
            "simple": "{text}\n\nمھرباني ڪري مٿي ڄاڻايل موضوع تي تفصيل سان لکو:",
            "continue": "{text}",
        },
        "bodo": {
            "instruction": "निर्देश: {instruction}\n\nइनपुट: {text}\n\nजाबाब:",
            "simple": "{text}\n\nकृपया उफ्राव बिसायाव थाखाय गोजौनि खन:",
            "continue": "{text}",
        },
        "maithili": {
            "instruction": "निर्देश: {instruction}\n\nइनपुट: {text}\n\nउत्तर:",
            "simple": "{text}\n\nकृपया उपर्युक्त विषय पर विस्तार सं लिखू:",
            "continue": "{text}",
        },
        "meitei": {
            "instruction": "ꯏꯅꯁ꯭ꯠꯔꯛꯁꯟ: {instruction}\n\nꯏꯅꯄꯨꯠ: {text}\n\nꯄꯥꯎꯈꯨꯝ:",
            "simple": "{text}\n\nꯆꯥꯅꯕꯤꯗꯨꯅꯥ ꯃꯊꯛꯇꯥ ꯄꯤꯔꯤꯕꯥ ꯍꯤꯔꯃꯗꯥ ꯌꯥꯝꯅꯥ ꯏꯕꯤꯌꯨ:",
            "continue": "{text}",
        },
        "santali": {
            "instruction": "ᱩᱫᱮᱥ: {instruction}\n\nᱤᱱᱯᱩᱴ: {text}\n\nᱛᱮᱞᱟ:",
            "simple": "{text}\n\nᱫᱚᱭᱟᱠᱟᱛᱮ ᱪᱮᱛᱟᱱ ᱨᱮ ᱵᱤᱥᱚᱭ ᱪᱮᱛᱟᱱ ᱨᱮ ᱚᱞ ᱢᱮ:",
            "continue": "{text}",
        },
        "english": {
            "instruction": "Instruction: {instruction}\n\nInput: {text}\n\nResponse:",
            "simple": "{text}\n\nPlease elaborate on the above topic in English only. Stay focused on the topic and do not use other languages:",
            "continue": "{text}",
        },
    }
    
    # ========== Q&A PROMPTS ==========
    QA_PROMPTS = {
        "hindi": "प्रश्न: {question}\n\nकृपया संक्षिप्त और सटीक उत्तर दें:",
        "sanskrit": "प्रश्नः: {question}\n\nकृपया संक्षिप्तं सटीकं च उत्तरं ददातु:",
        "tamil": "கேள்வி: {question}\n\nதயவுசெய்து சுருக்கமான மற்றும் துல்லியமான பதில் கொடுங்கள்:",
        "telugu": "ప్రశ్న: {question}\n\nదయచేసి సంక్షిప్తమైన మరియు ఖచ్చితమైన సమాధానం ఇవ్వండి:",
        "bengali": "প্রশ্ন: {question}\n\nঅনুগ্রহ করে সংক্ষিপ্ত এবং সঠিক উত্তর দিন:",
        "marathi": "प्रश्न: {question}\n\nकृपया संक्षिप्त आणि अचूक उत्तर द्या:",
        "gujarati": "પ્રશ્ન: {question}\n\nકૃપા કરીને સંક્ષિપ્ત અને ચોક્કસ જવાબ આપો:",
        "kannada": "ಪ್ರಶ್ನೆ: {question}\n\nದಯವಿಟ್ಟು ಸಂಕ್ಷಿಪ್ತ ಮತ್ತು ನಿಖರವಾದ ಉತ್ತರವನ್ನು ನೀಡಿ:",
        "malayalam": "ചോദ്യം: {question}\n\nദയവായി സംക്ഷിപ്തവും കൃത്യവുമായ ഉത്തരം നൽകുക:",
        "punjabi": "ਸਵਾਲ: {question}\n\nਕਿਰਪਾ ਕਰਕੇ ਸੰਖੇਪ ਅਤੇ ਸਟੀਕ ਜਵਾਬ ਦਿਓ:",
        "odia": "ପ୍ରଶ୍ନ: {question}\n\nଦୟାକରି ସଂକ୍ଷିପ୍ତ ଏବଂ ସଠିକ୍ ଉତ୍ତର ଦିଅନ୍ତୁ:",
        "assamese": "প্ৰশ্ন: {question}\n\nঅনুগ্ৰহ কৰি সংক্ষিপ্ত আৰু সঠিক উত্তৰ দিয়ক:",
        "urdu": "سوال: {question}\n\nبراہ کرم مختصر اور درست جواب دیں:",
        "kashmiri": "سوال: {question}\n\nمہربأنؠ کٔرتھ مختصر تہٕ صحیح جواب دِیو:",
        "nepali": "प्रश्न: {question}\n\nकृपया संक्षिप्त र सटीक उत्तर दिनुहोस्:",
        "sindhi": "سوال: {question}\n\nمھرباني ڪري مختصر ۽ صحيح جواب ڏيو:",
        "bodo": "आफाद: {question}\n\nकृपया जायखि आरो गेजेरनि जाबाब बिलाइ:",
        "maithili": "प्रश्न: {question}\n\nकृपया संक्षिप्त आ सही उत्तर दिअ:",
        "meitei": "ꯋꯥꯍꯪ: {question}\n\nꯆꯥꯅꯕꯤꯗꯨꯅꯥ ꯀꯨꯞꯅꯥ ꯑꯃꯁꯨꯡ ꯆꯨꯝꯕꯥ ꯄꯥꯎꯈꯨꯝ ꯄꯤꯌꯨ:",
        "santali": "ᱠᱩᱞᱤ: {question}\n\nᱫᱚᱭᱟᱠᱟᱛᱮ ᱠᱷᱟᱴᱚ ᱟᱨ ᱴᱷᱤᱠ ᱛᱮᱞᱟ ᱮᱢ ᱢᱮ:",
        "english": "Question: {question}\n\nPlease provide a concise and accurate answer:",
    }
    
    # ========== CONVERSATION PROMPTS ==========
    CONVERSATION_PROMPTS = {
        "hindi": {
            "system": "आप एक सहायक AI हैं। कृपया विनम्र और सहायक तरीके से उत्तर दें।",
            "user": "उपयोगकर्ता: {message}",
            "assistant": "सहायक:",
        },
        "sanskrit": {
            "system": "भवान् एकः साहाय्यकारी AI अस्ति। कृपया विनम्रतया साहाय्येन च उत्तरं ददातु।",
            "user": "उपयोक्ता: {message}",
            "assistant": "साहाय्यकारी:",
        },
        "tamil": {
            "system": "நீங்கள் ஒரு உதவி AI ஆவீர்கள். தயவுசெய்து பணிவுடனும் உதவியாகவும் பதிலளியுங்கள்.",
            "user": "பயனர்: {message}",
            "assistant": "உதவியாளர்:",
        },
        "telugu": {
            "system": "మీరు ఒక సహాయక AI. దయచేసి మర్యాదగా మరియు సహాయకరంగా సమాధానం ఇవ్వండి.",
            "user": "వినియోగదారు: {message}",
            "assistant": "సహాయకుడు:",
        },
        "bengali": {
            "system": "আপনি একটি সহায়ক AI। অনুগ্রহ করে বিনীত এবং সহায়ক উপায়ে উত্তর দিন।",
            "user": "ব্যবহারকারী: {message}",
            "assistant": "সহায়ক:",
        },
        "marathi": {
            "system": "तुम्ही एक सहाय्यक AI आहात. कृपया विनम्र आणि उपयुक्त पद्धतीने उत्तर द्या.",
            "user": "वापरकर्ता: {message}",
            "assistant": "सहाय्यक:",
        },
        "gujarati": {
            "system": "તમે એક સહાયક AI છો. કૃપા કરીને નમ્ર અને સહાયક રીતે જવાબ આપો.",
            "user": "વપરાશકર્તા: {message}",
            "assistant": "સહાયક:",
        },
        "kannada": {
            "system": "ನೀವು ಸಹಾಯಕ AI ಆಗಿದ್ದೀರಿ. ದಯವಿಟ್ಟು ವಿನಯಶೀಲ ಮತ್ತು ಸಹಾಯಕವಾದ ರೀತಿಯಲ್ಲಿ ಉತ್ತರಿಸಿ.",
            "user": "ಬಳಕೆದಾರ: {message}",
            "assistant": "ಸಹಾಯಕ:",
        },
        "malayalam": {
            "system": "നിങ്ങൾ ഒരു സഹായക AI ആണ്. ദയവായി മര്യാദയോടെയും സഹായകരമായും മറുപടി നൽകുക.",
            "user": "ഉപയോക്താവ്: {message}",
            "assistant": "സഹായി:",
        },
        "punjabi": {
            "system": "ਤੁਸੀਂ ਇੱਕ ਸਹਾਇਕ AI ਹੋ। ਕਿਰਪਾ ਕਰਕੇ ਨਿਮਰ ਅਤੇ ਸਹਾਇਕ ਤਰੀਕੇ ਨਾਲ ਜਵਾਬ ਦਿਓ।",
            "user": "ਉਪਭੋਗਤਾ: {message}",
            "assistant": "ਸਹਾਇਕ:",
        },
        "odia": {
            "system": "ଆପଣ ଏକ ସହାୟକ AI ଅଟନ୍ତି। ଦୟାକରି ନମ୍ର ଏବଂ ସହାୟକ ଉପାୟରେ ଉତ୍ତର ଦିଅନ୍ତୁ।",
            "user": "ଉପଭୋକ୍ତା: {message}",
            "assistant": "ସହାୟକ:",
        },
        "assamese": {
            "system": "আপুনি এটা সহায়ক AI। অনুগ্ৰহ কৰি বিনয়ী আৰু সহায়ক ধৰণে উত্তৰ দিয়ক।",
            "user": "ব্যৱহাৰকাৰী: {message}",
            "assistant": "সহায়ক:",
        },
        "urdu": {
            "system": "آپ ایک مددگار AI ہیں۔ براہ کرم شائستہ اور مددگار انداز میں جواب دیں۔",
            "user": "صارف: {message}",
            "assistant": "معاون:",
        },
        "kashmiri": {
            "system": "تۄہؠ چھُ اکھ مدد گار AI۔ مہربأنؠ کٔرتھ نرمی سیتؠ تہٕ مددگار انداز مَنٛز جواب دِیو۔",
            "user": "استعمال کران: {message}",
            "assistant": "مدد گار:",
        },
        "nepali": {
            "system": "तपाईं एक सहायक AI हुनुहुन्छ। कृपया विनम्र र सहायक तरिकाले जवाफ दिनुहोस्।",
            "user": "प्रयोगकर्ता: {message}",
            "assistant": "सहायक:",
        },
        "sindhi": {
            "system": "توهان هڪ مددگار AI آهيو. مھرباني ڪري نرمي ۽ مددگار انداز ۾ جواب ڏيو.",
            "user": "استعمال ڪندڙ: {message}",
            "assistant": "مددگار:",
        },
        "bodo": {
            "system": "नों गोसो मदद खालामनाय AI आव। कृपया नोजोरनि आरो मदद खालामनाय होजानाय जाबाब बिलाइ।",
            "user": "महरै खालामग्रा: {message}",
            "assistant": "मदद खालामग्रा:",
        },
        "maithili": {
            "system": "अहाँ एक सहायक AI छी। कृपया विनम्र आ सहायक तरीकासँ उत्तर दिअ।",
            "user": "उपयोगकर्ता: {message}",
            "assistant": "सहायक:",
        },
        "meitei": {
            "system": "ꯅꯍꯥꯛ ꯑꯁꯤ ꯃꯇꯦꯡ ꯄꯥꯡꯕꯥ AI ꯑꯃꯅꯤ꯫ ꯆꯥꯅꯕꯤꯗꯨꯅꯥ ꯏꯀꯥꯏ ꯈꯨꯝꯅꯕꯥ ꯑꯃꯁꯨꯡ ꯃꯇꯦꯡ ꯄꯥꯡꯕꯥ ꯃꯑꯣꯡꯗꯥ ꯄꯥꯎꯈꯨꯝ ꯄꯤꯌꯨ꯫",
            "user": "ꯁꯤꯖꯤꯟꯅꯔꯤꯕꯥ: {message}",
            "assistant": "ꯃꯇꯦꯡ ꯄꯥꯡꯕꯤ:",
        },
        "santali": {
            "system": "ᱟᱢ ᱫᱚ ᱢᱤᱫ ᱜᱚᱲᱚᱣᱤᱡ AI ᱠᱟᱱᱟᱢ᱾ ᱫᱚᱭᱟᱠᱟᱛᱮ ᱱᱟᱯᱟᱭ ᱟᱨ ᱜᱚᱲᱚᱣᱤᱡ ᱛᱮ ᱛᱮᱞᱟ ᱮᱢ ᱢᱮ᱾",
            "user": "ᱵᱮᱵᱷᱟᱨᱤᱭᱟ: {message}",
            "assistant": "ᱜᱚᱲᱚᱣᱤᱡ:",
        },
        "english": {
            "system": "You are a helpful AI assistant. Please respond politely and helpfully.",
            "user": "User: {message}",
            "assistant": "Assistant:",
        },
    }
    
    # ========== FEW-SHOT EXAMPLES ==========
    FEW_SHOT_EXAMPLES = {
        "hindi": [
            {"input": "भारत की राजधानी क्या है?", "output": "भारत की राजधानी नई दिल्ली है।"},
            {"input": "सूर्य किस दिशा में उगता है?", "output": "सूर्य पूर्व दिशा में उगता है।"},
        ],
        "english": [
            {"input": "What is the capital of India?", "output": "The capital of India is New Delhi."},
            {"input": "In which direction does the sun rise?", "output": "The sun rises in the east."},
        ],
        # Add more examples for other languages as needed
    }


class PromptEngineer:
    """
    Main class for generating optimized prompts for different tasks and languages.
    """
    
    def __init__(self):
        self.templates = PromptTemplates()
        logger.info("PromptEngineer initialized with 21 language templates")
    
    def get_generation_prompt(
        self,
        text: str,
        language: str = "english",
        mode: str = "simple",
        instruction: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for text generation.
        
        Args:
            text: Input text to generate from
            language: Target language
            mode: Prompt mode - "simple", "instruction", or "continue"
            instruction: Optional instruction for instruction mode
        
        Returns:
            Formatted prompt string
        """
        language = language.lower()
        
        # Fallback to English if language not supported
        if language not in self.templates.GENERATION_PROMPTS:
            logger.warning(f"Language '{language}' not found in templates, falling back to English")
            language = "english"
        
        lang_prompts = self.templates.GENERATION_PROMPTS[language]
        
        if mode == "instruction" and instruction:
            prompt = lang_prompts["instruction"].format(instruction=instruction, text=text)
        elif mode == "continue":
            prompt = lang_prompts["continue"].format(text=text)
        else:  # simple mode
            prompt = lang_prompts["simple"].format(text=text)
        
        logger.debug(f"Generated {mode} prompt for {language}: {prompt[:100]}...")
        return prompt
    
    def get_qa_prompt(
        self,
        question: str,
        language: str = "english",
        context: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for question answering.
        
        Args:
            question: Question to answer
            language: Target language
            context: Optional context information
        
        Returns:
            Formatted prompt string
        """
        language = language.lower()
        
        # Fallback to English if language not supported
        if language not in self.templates.QA_PROMPTS:
            logger.warning(f"Language '{language}' not found in QA templates, falling back to English")
            language = "english"
        
        prompt = self.templates.QA_PROMPTS[language].format(question=question)
        
        # Add context if provided
        if context:
            if language == "hindi":
                prompt = f"संदर्भ: {context}\n\n{prompt}"
            elif language == "english":
                prompt = f"Context: {context}\n\n{prompt}"
            else:
                # Generic context prefix
                prompt = f"{context}\n\n{prompt}"
        
        logger.debug(f"Generated QA prompt for {language}: {prompt[:100]}...")
        return prompt
    
    def get_conversation_prompt(
        self,
        message: str,
        language: str = "english",
        history: Optional[list] = None
    ) -> str:
        """
        Generate a prompt for conversation.
        
        Args:
            message: Current user message
            language: Target language
            history: Optional conversation history [(user_msg, assistant_msg), ...]
        
        Returns:
            Formatted prompt string
        """
        language = language.lower()
        
        # Fallback to English if language not supported
        if language not in self.templates.CONVERSATION_PROMPTS:
            logger.warning(f"Language '{language}' not found in conversation templates, falling back to English")
            language = "english"
        
        conv_prompts = self.templates.CONVERSATION_PROMPTS[language]
        
        # Start with system message
        prompt_parts = [conv_prompts["system"]]
        
        # Add conversation history if available
        if history:
            for user_msg, assistant_msg in history:
                prompt_parts.append(conv_prompts["user"].format(message=user_msg))
                prompt_parts.append(f"{conv_prompts['assistant']} {assistant_msg}")
        
        # Add current message
        prompt_parts.append(conv_prompts["user"].format(message=message))
        prompt_parts.append(conv_prompts["assistant"])
        
        prompt = "\n".join(prompt_parts)
        
        logger.debug(f"Generated conversation prompt for {language}: {prompt[:100]}...")
        return prompt
    
    def get_few_shot_prompt(
        self,
        text: str,
        language: str = "english",
        num_examples: int = 2
    ) -> str:
        """
        Generate a few-shot learning prompt with examples.
        
        Args:
            text: Input text
            language: Target language
            num_examples: Number of examples to include
        
        Returns:
            Formatted prompt string with examples
        """
        language = language.lower()
        
        if language not in self.templates.FEW_SHOT_EXAMPLES:
            logger.warning(f"No few-shot examples for {language}, using basic prompt")
            return self.get_generation_prompt(text, language, mode="simple")
        
        examples = self.templates.FEW_SHOT_EXAMPLES[language][:num_examples]
        
        # Build prompt with examples
        prompt_parts = []
        for example in examples:
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
            prompt_parts.append("")  # Empty line
        
        # Add current input
        prompt_parts.append(f"Input: {text}")
        prompt_parts.append("Output:")
        
        prompt = "\n".join(prompt_parts)
        
        logger.debug(f"Generated few-shot prompt for {language} with {num_examples} examples")
        return prompt


# Global instance for easy access
prompt_engineer = PromptEngineer()


# Convenience functions
def get_generation_prompt(text: str, language: str = "english", **kwargs) -> str:
    """Convenience function for generation prompts"""
    return prompt_engineer.get_generation_prompt(text, language, **kwargs)


def get_qa_prompt(question: str, language: str = "english", **kwargs) -> str:
    """Convenience function for Q&A prompts"""
    return prompt_engineer.get_qa_prompt(question, language, **kwargs)


def get_conversation_prompt(message: str, language: str = "english", **kwargs) -> str:
    """Convenience function for conversation prompts"""
    return prompt_engineer.get_conversation_prompt(message, language, **kwargs)

