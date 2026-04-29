
"""
Evaluation Metrics for Multilingual Language Model

This module implements comprehensive evaluation metrics as specified in the requirements:
- BLEU/ROUGE scores for text quality
- Perplexity for model performance
- Tokenization accuracy
- Language-specific fluency metrics
- Latency and performance metrics

Based on the requirements:
- Automatic evaluation: BLEU/ROUGE, perplexity, tokenization accuracy
- Manual checks for fluency in 5-10 prompts per language
- Latency checks to ensure API scales for multiple requests concurrently
"""

import os
import sys
import time
import logging
import json
import math
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from pathlib import Path

# Import settings
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualEvaluator:
    """
    Comprehensive evaluation system for multilingual language models
    
    This class implements all evaluation metrics specified in the requirements:
    1. Automatic evaluation: BLEU/ROUGE, perplexity, tokenization accuracy
    2. Manual fluency checks for 5-10 prompts per language
    3. Latency and performance metrics
    4. Language-specific evaluation
    """
    
    def __init__(self):
        self.evaluation_results = {
            "bleu_scores": defaultdict(list),
            "rouge_scores": defaultdict(list),
            "perplexity_scores": defaultdict(list),
            "tokenization_accuracy": defaultdict(list),
            "fluency_scores": defaultdict(list),
            "latency_metrics": defaultdict(list),
            "language_switching_success": defaultdict(list),
            "overall_scores": {}
        }
        
        # Test prompts for each language (5-10 per language as required)
        self.test_prompts = {
            "hindi": [
                "नमस्ते, आप कैसे हैं?",
                "भारत एक सुंदर देश है।",
                "मैं हिंदी सीख रहा हूं।",
                "यह एक परीक्षण वाक्य है।",
                "शिक्षा बहुत महत्वपूर्ण है।",
                "प्रकृति की सुंदरता अद्भुत है।",
                "हमें पर्यावरण की रक्षा करनी चाहिए।",
                "साहित्य और कला मानव सभ्यता की धरोहर हैं।",
                "एकता में शक्ति है।",
                "विज्ञान ने हमारे जीवन को बदल दिया है।"
            ],
            "sanskrit": [
                "नमस्कारः, भवान् कथं वर्तते?",
                "भारतं सुन्दरं देशं अस्ति।",
                "अहं संस्कृतं पठामि।",
                "इयं परीक्षणा वाक्यं अस्ति।",
                "विद्या अत्यन्त महत्वपूर्णा अस्ति।",
                "प्रकृतेः सौन्दर्यं अद्भुतं अस्ति।",
                "वयं पर्यावरणस्य रक्षणं कर्तव्यं।",
                "साहित्यं कला च मानवसभ्यतायाः धरोहरं।",
                "एकतायां शक्तिः अस्ति।",
                "विज्ञानं अस्माकं जीवनं परिवर्तितवत्।"
            ],
            "marathi": [
                "नमस्कार, तुम्ही कसे आहात?",
                "भारत हा सुंदर देश आहे।",
                "मी मराठी शिकत आहे।",
                "हे एक चाचणी वाक्य आहे।",
                "शिक्षण खूप महत्वाचे आहे।",
                "निसर्गाचे सौंदर्य अद्भुत आहे।",
                "आपल्याला पर्यावरणाचे संरक्षण करावे लागेल।",
                "साहित्य आणि कला मानवी संस्कृतीचा वारसा आहे।",
                "एकतेत शक्ती आहे।",
                "विज्ञानाने आपले जीवन बदलले आहे।"
            ],
            "tamil": [
                "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
                "இந்தியா ஒரு அழகான நாடு।",
                "நான் தமிழ் கற்கிறேன்।",
                "இது ஒரு சோதனை வாக்கியம்।",
                "கல்வி மிகவும் முக்கியமானது।",
                "இயற்கையின் அழகு அற்புதமானது।",
                "நாம் சுற்றுச்சூழலை பாதுகாக்க வேண்டும்।",
                "இலக்கியம் மற்றும் கலை மனித நாகரிகத்தின் பாரம்பரியம்।",
                "ஒற்றுமையில் வலிமை உள்ளது।",
                "அறிவியல் நமது வாழ்க்கையை மாற்றியுள்ளது।"
            ],
            "telugu": [
                "నమస్కారం, మీరు ఎలా ఉన్నారు?",
                "భారతదేశం ఒక అందమైన దేశం।",
                "నేను తెలుగు నేర్చుకుంటున్నాను।",
                "ఇది ఒక పరీక్ష వాక్యం।",
                "విద్య చాలా ముఖ్యమైనది।",
                "ప్రకృతి సౌందర్యం అద్భుతమైనది।",
                "మనం పర్యావరణాన్ని రక్షించాలి।",
                "సాహిత్యం మరియు కళలు మానవ నాగరికత యొక్క వారసత్వం।",
                "ఐక్యతలో శక్తి ఉంది।",
                "విజ్ఞానం మన జీవితాన్ని మార్చింది।"
            ],
            "kannada": [
                "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
                "ಭಾರತವು ಸುಂದರ ದೇಶವಾಗಿದೆ।",
                "ನಾನು ಕನ್ನಡ ಕಲಿಯುತ್ತಿದ್ದೇನೆ।",
                "ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ।",
                "ಶಿಕ್ಷಣ ತುಂಬಾ ಮುಖ್ಯವಾಗಿದೆ।",
                "ಪ್ರಕೃತಿಯ ಸೌಂದರ್ಯ ಅದ್ಭುತವಾಗಿದೆ।",
                "ನಾವು ಪರಿಸರವನ್ನು ರಕ್ಷಿಸಬೇಕು।",
                "ಸಾಹಿತ್ಯ ಮತ್ತು ಕಲೆಗಳು ಮಾನವ ನಾಗರಿಕತೆಯ ಸಂಪತ್ತು।",
                "ಏಕತೆಯಲ್ಲಿ ಶಕ್ತಿ ಇದೆ।",
                "ವಿಜ್ಞಾನ ನಮ್ಮ ಜೀವನವನ್ನು ಬದಲಾಯಿಸಿದೆ।"
            ],
            "bengali": [
                "নমস্কার, আপনি কেমন আছেন?",
                "ভারত একটি সুন্দর দেশ।",
                "আমি বাংলা শিখছি।",
                "এটি একটি পরীক্ষার বাক্য।",
                "শিক্ষা খুবই গুরুত্বপূর্ণ।",
                "প্রকৃতির সৌন্দর্য বিস্ময়কর।",
                "আমাদের পরিবেশ রক্ষা করতে হবে।",
                "সাহিত্য এবং শিল্প মানব সভ্যতার সম্পদ।",
                "ঐক্যে শক্তি।",
                "বিজ্ঞান আমাদের জীবনকে পরিবর্তন করেছে।"
            ],
            "gujarati": [
                "નમસ્કાર, તમે કેમ છો?",
                "ભારત એક સુંદર દેશ છે।",
                "હું ગુજરાતી શીખી રહ્યો છું।",
                "આ એક પરીક્ષા વાક્ય છે।",
                "શિક્ષણ ખૂબ મહત્વપૂર્ણ છે।",
                "પ્રકૃતિની સુંદરતા અદ્ભુત છે।",
                "આપણે પર્યાવરણનું રક્ષણ કરવું જોઈએ।",
                "સાહિત્ય અને કલા માનવ સભ્યતાનો વારસો છે।",
                "એકતામાં શક્તિ છે।",
                "વિજ્ઞાને આપણા જીવનને બદલી નાખ્યું છે।"
            ],
            "punjabi": [
                "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
                "ਭਾਰਤ ਇੱਕ ਸੁੰਦਰ ਦੇਸ਼ ਹੈ।",
                "ਮੈਂ ਪੰਜਾਬੀ ਸਿੱਖ ਰਿਹਾ ਹਾਂ।",
                "ਇਹ ਇੱਕ ਪਰੀਖਿਆ ਵਾਕ ਹੈ।",
                "ਸਿੱਖਿਆ ਬਹੁਤ ਮਹੱਤਵਪੂਰਨ ਹੈ।",
                "ਪ੍ਰਕਿਰਤੀ ਦੀ ਸੁੰਦਰਤਾ ਸ਼ਾਨਦਾਰ ਹੈ।",
                "ਸਾਨੂੰ ਵਾਤਾਵਰਣ ਦੀ ਰੱਖਿਆ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ।",
                "ਸਾਹਿਤ ਅਤੇ ਕਲਾ ਮਨੁੱਖੀ ਸਭਿਅਤਾ ਦੀ ਸੰਪਤੀ ਹਨ।",
                "ਏਕਤਾ ਵਿੱਚ ਸ਼ਕਤੀ ਹੈ।",
                "ਵਿਗਿਆਨ ਨੇ ਸਾਡੇ ਜੀਵਨ ਨੂੰ ਬਦਲ ਦਿੱਤਾ ਹੈ।"
            ],
            "odia": [
                "ନମସ୍କାର, ଆପଣ କିପରି ଅଛନ୍ତି?",
                "ଭାରତ ଏକ ସୁନ୍ଦର ଦେଶ।",
                "ମୁଁ ଓଡ଼ିଆ ଶିଖୁଛି।",
                "ଏହା ଏକ ପରୀକ୍ଷା ବାକ୍ୟ।",
                "ଶିକ୍ଷା ବହୁତ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ।",
                "ପ୍ରକୃତିର ସୌନ୍ଦର୍ଯ୍ୟ ଅଦ୍ଭୁତ।",
                "ଆମେ ପରିବେଶ ରକ୍ଷା କରିବା ଉଚିତ।",
                "ସାହିତ୍ୟ ଏବଂ କଳା ମାନବ ସଭ୍ୟତାର ସମ୍ପଦ।",
                "ଏକତାରେ ଶକ୍ତି।",
                "ବିଜ୍ଞାନ ଆମ ଜୀବନକୁ ବଦଳାଇଛି।"
            ],
            "malayalam": [
                "നമസ്കാരം, നിങ്ങൾ എങ്ങനെയുണ്ട്?",
                "ഇന്ത്യ ഒരു മനോഹരമായ രാജ്യമാണ്।",
                "ഞാൻ മലയാളം പഠിക്കുന്നു।",
                "ഇത് ഒരു പരീക്ഷാ വാക്യമാണ്।",
                "വിദ്യാഭ്യാസം വളരെ പ്രധാനമാണ്।",
                "പ്രകൃതിയുടെ സൗന്ദര്യം അതിശയകരമാണ്।",
                "നമ്മൾ പരിസ്ഥിതി സംരക്ഷിക്കണം।",
                "സാഹിത്യവും കലയും മനുഷ്യ സംസ്കാരത്തിന്റെ സമ്പത്താണ്।",
                "ഐക്യത്തിൽ ശക്തി।",
                "ശാസ്ത്രം നമ്മുടെ ജീവിതത്തെ മാറ്റി।"
            ],
            "assamese": [
                "নমস্কাৰ, আপুনি কেনেকৈ আছা?",
                "ভাৰত এখন সুন্দৰ দেশ।",
                "মই অসমীয়া শিকি আছোঁ।",
                "এইটো এটা পৰীক্ষাৰ বাক্য।",
                "শিক্ষা বৰ গুৰুত্বপূৰ্ণ।",
                "প্ৰকৃতিৰ সৌন্দৰ্য বিস্ময়কৰ।",
                "আমি পৰিৱেশ ৰক্ষা কৰিব লাগে।",
                "সাহিত্য আৰু কলা মানৱ সভ্যতাৰ সম্পদ।",
                "ঐক্যত শক্তি।",
                "বিজ্ঞানে আমাৰ জীৱনক সলনি কৰিছে।"
            ],
            "urdu": [
                "السلام علیکم، آپ کیسے ہیں؟",
                "بھارت ایک خوبصورت ملک ہے۔",
                "میں اردو سیکھ رہا ہوں۔",
                "یہ ایک امتحانی جملہ ہے۔",
                "تعلیم بہت اہم ہے۔",
                "فطرت کا حسن حیرت انگیز ہے۔",
                "ہمیں ماحول کا تحفظ کرنا چاہیے۔",
                "ادب اور فن انسانی تہذیب کا خزانہ ہے۔",
                "اتحاد میں طاقت۔",
                "سائنس نے ہماری زندگی بدل دی ہے۔"
            ],
            "nepali": [
                "नमस्कार, तपाईं कसरी हुनुहुन्छ?",
                "भारत एक सुन्दर देश हो।",
                "म नेपाली सिक्दै छु।",
                "यो एक परीक्षा वाक्य हो।",
                "शिक्षा धेरै महत्वपूर्ण छ।",
                "प्रकृतिको सुन्दरता अद्भुत छ।",
                "हामीले वातावरणको रक्षा गर्नुपर्छ।",
                "साहित्य र कला मानव सभ्यताको सम्पदा हो।",
                "एकतामा शक्ति छ।",
                "विज्ञानले हाम्रो जीवन परिवर्तन गरेको छ।"
            ],
            "english": [
                "Hello, how are you?",
                "India is a beautiful country.",
                "I am learning English.",
                "This is a test sentence.",
                "Education is very important.",
                "The beauty of nature is amazing.",
                "We must protect the environment.",
                "Literature and art are treasures of human civilization.",
                "Unity is strength.",
                "Science has changed our lives."
            ]
        }
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate BLEU score for text quality evaluation
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            BLEU score (0-1)
        """
        # Simple BLEU implementation - in production, use nltk.translate.bleu_score
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if len(cand_words) == 0:
            return 0.0
        
        # Calculate 1-gram precision
        ref_word_counts = {}
        for word in ref_words:
            ref_word_counts[word] = ref_word_counts.get(word, 0) + 1
        
        cand_word_counts = {}
        for word in cand_words:
            cand_word_counts[word] = cand_word_counts.get(word, 0) + 1
        
        # Count matches
        matches = 0
        for word, count in cand_word_counts.items():
            matches += min(count, ref_word_counts.get(word, 0))
        
        precision = matches / len(cand_words)
        
        # Brevity penalty
        if len(cand_words) < len(ref_words):
            bp = math.exp(1 - len(ref_words) / len(cand_words))
        else:
            bp = 1.0
        
        bleu_score = bp * precision
        return min(bleu_score, 1.0)
    
    def calculate_rouge_score(self, reference: str, candidate: str) -> float:
        """
        Calculate ROUGE score for text quality evaluation
        
        Args:
            reference: Reference text
            candidate: Generated text
            
        Returns:
            ROUGE score (0-1)
        """
        # Simple ROUGE-L implementation
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if len(ref_words) == 0 or len(cand_words) == 0:
            return 0.0
        
        # Calculate LCS (Longest Common Subsequence)
        def lcs_length(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(ref_words, cand_words)
        
        # Calculate ROUGE-L
        recall = lcs_len / len(ref_words)
        precision = lcs_len / len(cand_words)
        
        if recall + precision == 0:
            return 0.0
        
        rouge_l = 2 * recall * precision / (recall + precision)
        return rouge_l
    
    def calculate_perplexity(self, text: str, model, tokenizer) -> float:
        """
        Calculate perplexity for model performance evaluation
        
        Args:
            text: Input text
            model: Language model
            tokenizer: Tokenizer
            
        Returns:
            Perplexity score
        """
        try:
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to same device as model
            if hasattr(model, 'device'):
                inputs = inputs.to(model.device)
            
            # Calculate loss
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            # Calculate perplexity
            perplexity = math.exp(loss.item())
            return perplexity
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def evaluate_tokenization_accuracy(self, original_text: str, tokenized_text: str, 
                                     detokenized_text: str) -> float:
        """
        Evaluate tokenization accuracy
        
        Args:
            original_text: Original input text
            tokenized_text: Tokenized text
            detokenized_text: Detokenized text
            
        Returns:
            Accuracy score (0-1)
        """
        # Simple accuracy based on character-level similarity
        if len(original_text) == 0:
            return 0.0
        
        # Normalize texts for comparison
        orig_norm = original_text.strip().lower()
        detok_norm = detokenized_text.strip().lower()
        
        # Calculate character-level accuracy
        matches = sum(1 for a, b in zip(orig_norm, detok_norm) if a == b)
        max_len = max(len(orig_norm), len(detok_norm))
        
        if max_len == 0:
            return 1.0
        
        accuracy = matches / max_len
        return accuracy
    
    def evaluate_fluency(self, text: str, language: str) -> float:
        """
        Evaluate fluency of generated text
        
        Args:
            text: Generated text
            language: Language code
            
        Returns:
            Fluency score (0-1)
        """
        # Simple fluency evaluation based on:
        # 1. Sentence length appropriateness
        # 2. Word repetition
        # 3. Character distribution
        
        if len(text.strip()) == 0:
            return 0.0
        
        score = 1.0
        
        # Check sentence length (not too short, not too long)
        words = text.split()
        if len(words) < 3:
            score -= 0.3  # Too short
        elif len(words) > 50:
            score -= 0.2  # Too long
        
        # Check for excessive repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition > len(words) * 0.3:  # More than 30% repetition
            score -= 0.3
        
        # Check character distribution (for non-Latin scripts)
        if language != "english":
            # Ensure good character distribution
            unique_chars = len(set(text))
            if unique_chars < len(text) * 0.3:  # Less than 30% unique characters
                score -= 0.2
        
        return max(score, 0.0)
    
    def evaluate_language_switching(self, text: str, expected_language: str) -> float:
        """
        Evaluate language switching capability
        
        Args:
            text: Generated text
            expected_language: Expected language
            
        Returns:
            Language switching success score (0-1)
        """
        # Import the language detection function
        from src.api.main import detect_language
        
        detected_lang, confidence = detect_language(text)
        
        if detected_lang == expected_language:
            return confidence
        else:
            return 0.0
    
    def measure_latency(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure function execution latency
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, latency_in_seconds)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency = end_time - start_time
        return result, latency
    
    def evaluate_language(self, language: str, model, tokenizer) -> Dict[str, Any]:
        """
        Comprehensive evaluation for a specific language
        
        Args:
            language: Language code
            model: Language model
            tokenizer: Tokenizer
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating language: {language}")
        
        if language not in self.test_prompts:
            logger.warning(f"No test prompts available for {language}")
            return {}
        
        results = {
            "language": language,
            "bleu_scores": [],
            "rouge_scores": [],
            "perplexity_scores": [],
            "tokenization_accuracy": [],
            "fluency_scores": [],
            "latency_metrics": [],
            "language_switching_success": []
        }
        
        for i, prompt in enumerate(self.test_prompts[language]):
            logger.info(f"Evaluating prompt {i+1}/{len(self.test_prompts[language])} for {language}")
            
            try:
                # Generate response
                def generate_response():
                    inputs = tokenizer(prompt, return_tensors="pt")
                    if hasattr(model, 'device'):
                        inputs = inputs.to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return generated_text[len(prompt):].strip()
                
                # Measure latency
                generated_text, latency = self.measure_latency(generate_response)
                
                # Calculate metrics
                bleu_score = self.calculate_bleu_score(prompt, generated_text)
                rouge_score = self.calculate_rouge_score(prompt, generated_text)
                perplexity = self.calculate_perplexity(generated_text, model, tokenizer)
                fluency = self.evaluate_fluency(generated_text, language)
                lang_switching = self.evaluate_language_switching(generated_text, language)
                
                # Tokenization accuracy (simplified)
                tokenized = tokenizer.encode(generated_text)
                detokenized = tokenizer.decode(tokenized)
                token_acc = self.evaluate_tokenization_accuracy(generated_text, str(tokenized), detokenized)
                
                # Store results
                results["bleu_scores"].append(bleu_score)
                results["rouge_scores"].append(rouge_score)
                results["perplexity_scores"].append(perplexity)
                results["tokenization_accuracy"].append(token_acc)
                results["fluency_scores"].append(fluency)
                results["latency_metrics"].append(latency)
                results["language_switching_success"].append(lang_switching)
                
            except Exception as e:
                logger.error(f"Error evaluating prompt {i+1} for {language}: {e}")
                continue
        
        # Calculate averages
        for metric in ["bleu_scores", "rouge_scores", "perplexity_scores", 
                      "tokenization_accuracy", "fluency_scores", "latency_metrics", 
                      "language_switching_success"]:
            if results[metric]:
                results[f"avg_{metric}"] = np.mean(results[metric])
                results[f"std_{metric}"] = np.std(results[metric])
            else:
                results[f"avg_{metric}"] = 0.0
                results[f"std_{metric}"] = 0.0
        
        return results
    
    def evaluate_all_languages(self, model, tokenizer) -> Dict[str, Any]:
        """
        Evaluate all supported languages
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive evaluation of all languages")
        
        all_results = {}
        overall_scores = {}
        
        for language in settings.SUPPORTED_LANGUAGES:
            try:
                results = self.evaluate_language(language, model, tokenizer)
                all_results[language] = results
                
                # Calculate overall score for this language
                if results:
                    overall_score = (
                        results.get("avg_bleu_scores", 0) * 0.2 +
                        results.get("avg_rouge_scores", 0) * 0.2 +
                        (1.0 / (1.0 + results.get("avg_perplexity_scores", 1))) * 0.2 +
                        results.get("avg_tokenization_accuracy", 0) * 0.2 +
                        results.get("avg_fluency_scores", 0) * 0.2
                    )
                    overall_scores[language] = overall_score
                
            except Exception as e:
                logger.error(f"Error evaluating {language}: {e}")
                continue
        
        # Calculate overall system score
        if overall_scores:
            system_score = np.mean(list(overall_scores.values()))
        else:
            system_score = 0.0
        
        return {
            "language_results": all_results,
            "overall_scores": overall_scores,
            "system_score": system_score,
            "evaluation_timestamp": time.time()
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            filepath: Output file path
        """
        # Only create directory if filepath contains a directory
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {filepath}")
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print evaluation summary
        
        Args:
            results: Evaluation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("MULTILINGUAL LANGUAGE MODEL EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        if "system_score" in results:
            logger.info(f"Overall System Score: {results['system_score']:.3f}")
        
        if "language_results" in results:
            logger.info("\nLanguage-wise Results:")
            logger.info("-" * 50)
            
            for lang, lang_results in results["language_results"].items():
                if lang_results:
                    logger.info(f"\n{lang.upper()}:")
                    logger.info(f"  BLEU Score: {lang_results.get('avg_bleu_scores', 0):.3f} ± {lang_results.get('std_bleu_scores', 0):.3f}")
                    logger.info(f"  ROUGE Score: {lang_results.get('avg_rouge_scores', 0):.3f} ± {lang_results.get('std_rouge_scores', 0):.3f}")
                    logger.info(f"  Perplexity: {lang_results.get('avg_perplexity_scores', 0):.2f} ± {lang_results.get('std_perplexity_scores', 0):.2f}")
                    logger.info(f"  Tokenization Accuracy: {lang_results.get('avg_tokenization_accuracy', 0):.3f} ± {lang_results.get('std_tokenization_accuracy', 0):.3f}")
                    logger.info(f"  Fluency Score: {lang_results.get('avg_fluency_scores', 0):.3f} ± {lang_results.get('std_fluency_scores', 0):.3f}")
                    logger.info(f"  Avg Latency: {lang_results.get('avg_latency_metrics', 0):.3f}s ± {lang_results.get('std_latency_metrics', 0):.3f}s")
                    logger.info(f"  Language Switching: {lang_results.get('avg_language_switching_success', 0):.3f} ± {lang_results.get('std_language_switching_success', 0):.3f}")
        
        logger.info("=" * 80)


def main():
    """Main function to run evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multilingual language model")
    parser.add_argument("--languages", nargs="+", default=settings.SUPPORTED_LANGUAGES,
                       help="Languages to evaluate")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MultilingualEvaluator()
    
    # Note: In a real implementation, you would load the model and tokenizer here
    # For now, we'll create a mock evaluation
    logger.info("Note: This is a mock evaluation. In production, load your model and tokenizer.")
    
    # Mock results for demonstration
    mock_results = {
        "language_results": {},
        "overall_scores": {},
        "system_score": 0.75,
        "evaluation_timestamp": time.time()
    }
    
    for lang in args.languages:
        mock_results["language_results"][lang] = {
            "language": lang,
            "avg_bleu_scores": 0.65,
            "avg_rouge_scores": 0.70,
            "avg_perplexity_scores": 15.5,
            "avg_tokenization_accuracy": 0.95,
            "avg_fluency_scores": 0.80,
            "avg_latency_metrics": 0.5,
            "avg_language_switching_success": 0.85
        }
        mock_results["overall_scores"][lang] = 0.75
    
    # Print summary
    evaluator.print_summary(mock_results)
    
    # Save results
    evaluator.save_results(mock_results, args.output)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
