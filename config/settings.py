"""
Unified Configuration Settings for BHIV Sovereign AI Platform
Supports:
1. Multilingual Tokenization & Inference API (Sovereign LM Bridge)
2. BHIV Core Multi-Modal AI Processing
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ============================================================================
# SYSTEM 1: MULTILINGUAL TOKENIZATION & INFERENCE API
# ============================================================================

DEFAULT_MAX_LENGTH = 1024  # Optimal for most generation tasks
TRAINING_MAX_LENGTH = 1024  # For fine-tuning
INFERENCE_MAX_LENGTH = 1536  # For inference (can be higher since no gradient computation)

# API Configuration (Sovereign LM Bridge)
API_HOST = "127.0.0.1"
API_PORT = 8000
API_TITLE = "Multilingual Tokenization & Inference API"
API_DESCRIPTION = "API for Hindi, Sanskrit, Marathi, and English tokenization and text generation"
API_VERSION = "1.0.1"
DEBUG_MODE = True

# Model and Tokenizer Paths
TOKENIZER_MODEL_PATH = "model/multilingual_tokenizer.model"   # SentencePiece .model (optional)
TOKENIZER_VOCAB_PATH = "model/multilingual_tokenizer.vocab"
TOKENIZER_MERGE_PATH = "model/tokenizer_merge.txt"

# Model Configuration
# 🔄 TESTING: Comparing 560m + Gurukul Lite vs 1b7
MODEL_NAME = os.getenv("MODEL_NAME", "bigscience/bloomz-560m")  # HF model name fallback (decoder-only)
MODEL_PATH = os.getenv("MODEL_PATH", "adapters/gurukul_lite")  # local checkpoint folder (if used). Empty -> use MODEL_NAME

# Generation params - Optimized for accuracy
MAX_GENERATION_LENGTH = 256
NUM_RETURN_SEQUENCES = 1
TEMPERATURE = 0.6  # 🎯 Lowered from 0.7 → More focused, less random
TOP_P = 0.85  # 🎯 Lowered from 0.9 → More conservative choices
TOP_K = 40  # 🎯 NEW: Limit vocabulary for consistency
DO_SAMPLE = True
REPETITION_PENALTY = 1.3  # 🎯 Increased from default → Reduce repetition
NO_REPEAT_NGRAM_SIZE = 3  # 🎯 NEW: Prevent 3-gram repetition
MIN_LENGTH = 50  # 🎯 NEW: Ensure minimum output length

# Supported Languages - 21 Indian languages
SUPPORTED_LANGUAGES = [
    "assamese", "bengali", "bodo", "english", "gujurati", "hindi",
    "kannada", "kashmiri", "maithili", "malyalam", "marathi", "meitei",
    "nepali", "odia", "punjabi", "sanskrit", "santali", "sindhi",
    "tamil", "telugu", "urdu"
]
DEFAULT_LANGUAGE = "english"

# Language Detection Configuration
# Unicode ranges for different scripts
UNICODE_RANGES = {
    "devanagari": (0x0900, 0x097F),  # Hindi, Sanskrit, Marathi, Nepali
    "tamil": (0x0B80, 0x0BFF),       # Tamil
    "telugu": (0x0C00, 0x0C7F),      # Telugu
    "kannada": (0x0C80, 0x0CFF),     # Kannada
    "bengali": (0x0980, 0x09FF),     # Bengali, Assamese
    "gujarati": (0x0A80, 0x0AFF),    # Gujarati
    "punjabi": (0x0A00, 0x0A7F),     # Punjabi (Gurmukhi)
    "odia": (0x0B00, 0x0B7F),        # Odia
    "malayalam": (0x0D00, 0x0D7F),   # Malayalam
    "urdu": (0x0600, 0x06FF),        # Urdu (Arabic script)
    "latin": (0x0000, 0x007F),       # English (Latin script)
    "meetei_mayek": (0xABC0, 0xABFF) # Meitei (Meetei Mayek script)
}

# Legacy support
DEVANAGARI_UNICODE_RANGE = UNICODE_RANGES["devanagari"]
LANGUAGE_CONFIDENCE_THRESHOLD = 0.5
ENGLISH_RATIO_THRESHOLD = 0.8
DEVANAGARI_RATIO_THRESHOLD = 0.5

# Language-specific keywords for detection
LANGUAGE_KEYWORDS = {
    "sanskrit": ["संस्कृत", "श्लोक", "मन्त्र", "वेद", "उपनिषद्", "गीता", "धर्मो", "अस्ति", "भवति", "भवान्", "कथं", "वर्तते", "नमस्कारः", "सः", "सा", "तत्", "यत्", "किम्", "कुत्र", "कदा", "कथम्", "केन", "कस्य", "कस्मै", "कस्मात्", "कस्याम्", "कस्मिन्"],
    "marathi": ["महाराष्ट्र", "आहे", "आहेत", "होते", "होता", "म्हणजे", "काय", "कसे", "एकदा", "मी", "तू", "तुम्ही", "आहात", "म्हणतात", "करतात", "जातात", "येतात", "घेतात", "देतात"],
    "hindi": ["है", "हैं", "था", "थी", "होगा", "होगी", "क्या", "कैसे", "यह", "वह", "एक", "मैं", "तुम", "आप", "कैसे", "कैसा", "कैसी", "कहाँ", "कब", "क्यों", "कौन", "किसने", "किसको", "किससे"],
    "tamil": ["தமிழ்", "ஆகும்", "இருக்கிறது", "செய்கிறது", "என்ன", "எப்படி", "நான்", "நீ"],
    "telugu": ["తెలుగు", "అవుతుంది", "ఉంది", "చేస్తుంది", "ఏమి", "ఎలా", "నేను", "నువ్వు"],
    "kannada": ["ಕನ್ನಡ", "ಆಗುತ್ತದೆ", "ಇದೆ", "ಮಾಡುತ್ತದೆ", "ಏನು", "ಹೇಗೆ", "ನಾನು", "ನೀನು"],
    "bengali": ["বাংলা", "হয়", "আছে", "করে", "কী", "কীভাবে", "আমি", "তুমি"],
    "gujurati": ["ગુજરાતી", "છે", "છે", "કરે", "શું", "કેવી", "હું", "તું"],
    "punjabi": ["ਪੰਜਾਬੀ", "ਹੈ", "ਹੈ", "ਕਰਦਾ", "ਕੀ", "ਕਿਵੇਂ", "ਮੈਂ", "ਤੂੰ"],
    "odia": ["ଓଡ଼ିଆ", "ହୁଏ", "ଅଛି", "କରେ", "କଣ", "କିପରି", "ମୁଁ", "ତୁମେ"],
    "malyalam": ["മലയാളം", "ആണ്", "ഉണ്ട്", "ചെയ്യുന്നു", "എന്ത്", "എങ്ങനെ", "ഞാൻ", "നീ"],
    "assamese": ["অসমীয়া", "হয়", "আছে", "কৰে", "কি", "কেনেকৈ", "মই", "তুমি"],
    "urdu": ["اردو", "ہے", "ہے", "کرتا", "کیا", "کیسے", "میں", "تم"],
    "nepali": ["नेपाली", "छ", "छ", "गर्छ", "के", "कसरी", "म", "तिमी", "हो", "हुन्छ", "गर्नुहुन्छ", "कसरी", "कहाँ", "कब", "किन", "कसो", "मैले", "तिमीले", "हामीले", "उहाँले"],
    "kashmiri": ["کٲشُر", "چھ", "چھ", "کران", "کیا", "کیوی", "می", "تہ"],
    "meitei": ["মৈতৈলোন", "দৈ", "দৈ", "নরবা", "কী", "কীদা", "ঈ", "নুং"],
    "sindhi": ["سنڌي", "آهي", "آهي", "ڪري", "ڇا", "ڪيئن", "مان", "تون"],
    "bodo": ["बड़ो", "जायो", "जायो", "खालाम", "मा", "माब्ला", "आं", "नों"],
    "maithili": ["मैथिली", "अछि", "अछि", "करैत", "कि", "कहाँ", "हम", "तोहर"],
    "santali": ["ᱥᱟᱱᱛᱟᱲᱤ", "ᱦᱩᱭ", "ᱦᱩᱭ", "ᱠᱚᱨ", "ᱢᱮ", "ᱠᱮᱢᱚᱱ", "ᱟᱢ", "ᱟᱢᱮ"]
}

# Training / Fine-tuning Configuration
TRAINING_DATA_PATH = "data/training"
VALIDATION_DATA_PATH = "data/validation"
CORPUS_FILES = {
    "assamese": "as_train.txt",
    "bengali": "bn_train.txt",
    "bodo": "bd_train.txt",
    "english": "en_train.txt",
    "gujurati": "gu_train.txt",
    "hindi": "hi_train.txt",
    "kannada": "kn_train.txt",
    "kashmiri": "ks_train.txt",
    "maithili": "mai_train.txt",
    "malyalam": "ml_train.txt",
    "marathi": "mr_train.txt",
    "meitei": "mni_train.txt",
    "nepali": "ne_train.txt",
    "odia": "or_train.txt",
    "punjabi": "pa_train.txt",
    "sanskrit": "sa_train.txt",
    "santali": "sat_train.txt",
    "sindhi": "sd_train.txt",
    "tamil": "ta_train.txt",
    "telugu": "te_train.txt",
    "urdu": "ur_train.txt"
}
FINE_TUNED_MODEL_PATH = "model"

# SentencePiece Training Parameters
SP_VOCAB_SIZE = 32000
SP_MODEL_TYPE = "bpe"  
SP_CHARACTER_COVERAGE = 0.9995
SP_INPUT_SENTENCE_SIZE = 10000000
SP_SHUFFLE_INPUT_SENTENCE = True

# FastText Configuration
FASTTEXT_MODEL_PATH = "models/lid.176.bin"
FASTTEXT_DETECTION_THRESHOLD = 0.7

# KB and TTS (Vaani) integration endpoints
KB_ENDPOINT = os.getenv("KB_ENDPOINT", "http://127.0.0.1:8001")  # Custom KB service
KB_TIMEOUT = float(os.getenv("KB_TIMEOUT", 120.0))

VAANI_ENDPOINT = os.getenv("VAANI_ENDPOINT", "")
VAANI_TIMEOUT = float(os.getenv("VAANI_TIMEOUT", 120.0))

# Device and performance options
USE_FP16_IF_GPU = True

# Quantization Configuration for Inference
# 🔄 DISABLED for BLOOMZ-560m: Not needed for smaller model
USE_4BIT_QUANTIZATION = False  # Disable for 560m (only needed for 1b7+)
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",  # NormalFloat4 quantization
    "bnb_4bit_compute_dtype": "float16",  # Compute dtype for 4-bit base models
    "bnb_4bit_use_double_quant": True,  # Use double quantization for better accuracy
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/api.log"

# Enhanced logging format
DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"

# ============================================================================
# SYSTEM 2: BHIV CORE MULTI-MODAL AI PROCESSING
# ============================================================================

# Model Configuration (BHIV Core)
MODEL_CONFIG = {
    "llama": {
        "api_url": "http://localhost:1234/v1/chat/completions",
        "model_name": "llama-3.1-8b-instruct"
    },
    "vedas_agent": {
        "endpoint": "http://localhost:8001/ask-vedas",
        "headers": {"Content-Type": "application/json"},
        "api_key": os.getenv("GEMINI_API_KEY"),
        "backup_api_key": os.getenv("GEMINI_API_KEY_BACKUP")
    },
    "edumentor_agent": {
        "endpoint": "http://localhost:8001/edumentor",
        "headers": {"Content-Type": "application/json"},
        "api_key": os.getenv("GEMINI_API_KEY"),
        "backup_api_key": os.getenv("GEMINI_API_KEY_BACKUP")
    },
    "wellness_agent": {
        "endpoint": "http://localhost:8001/wellness",
        "headers": {"Content-Type": "application/json"},
        "api_key": os.getenv("GEMINI_API_KEY"),
        "backup_api_key": os.getenv("GEMINI_API_KEY_BACKUP")
    }
}

# MongoDB Configuration (BHIV Core)
MONGO_CONFIG = {
    "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    "database": "bhiv_core",
    "collection": "task_logs"
}

# Timeout Configuration (BHIV Core)
TIMEOUT_CONFIG = {
    "default_timeout": int(os.getenv("DEFAULT_TIMEOUT", 120)),
    "image_processing_timeout": int(os.getenv("IMAGE_PROCESSING_TIMEOUT", 180)),
    "audio_processing_timeout": int(os.getenv("AUDIO_PROCESSING_TIMEOUT", 240)),
    "pdf_processing_timeout": int(os.getenv("PDF_PROCESSING_TIMEOUT", 150)),
    "llm_timeout": int(os.getenv("LLM_TIMEOUT", 120)),
    "file_upload_timeout": int(os.getenv("FILE_UPLOAD_TIMEOUT", 300))
}

# RL Configuration (BHIV Core)
RL_CONFIG = {
    "use_rl": os.getenv("USE_RL", "true").lower() == "true",
    "exploration_rate": float(os.getenv("RL_EXPLORATION_RATE", 0.2)),
    "buffer_file": "logs/learning_log.json",
    "model_log_file": "logs/model_logs.json",
    "agent_log_file": "logs/agent_logs.json",
    "memory_size": int(os.getenv("RL_MEMORY_SIZE", 1000)),
    "min_exploration_rate": float(os.getenv("RL_MIN_EXPLORATION", 0.05)),
    "exploration_decay": float(os.getenv("RL_EXPLORATION_DECAY", 0.995)),
    "confidence_threshold": float(os.getenv("RL_CONFIDENCE_THRESHOLD", 0.7)),
    "enable_ucb": os.getenv("RL_ENABLE_UCB", "true").lower() == "true",
    "enable_fallback_learning": os.getenv("RL_ENABLE_FALLBACK_LEARNING", "true").lower() == "true",
    "log_to_mongo": os.getenv("RL_LOG_TO_MONGO", "true").lower() == "true"
}

# ============================================================================
# SHARED UTILITIES
# ============================================================================

# Create necessary directories
DIRECTORIES_TO_CREATE = [
    "data/training",
    "data/validation",
    "logs",
    "model"
]

def create_directories():
    """Create necessary directories if they don't exist"""
    created_dirs = []
    for directory in DIRECTORIES_TO_CREATE:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created_dirs.append(directory)
            print(f"[OK] Created directory: {directory}")
        else:
            print(f"[EXISTS] Directory exists: {directory}")
    return created_dirs

def get_model_config():
    """Get model configuration dictionary"""
    return {
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "max_new_tokens": MAX_GENERATION_LENGTH,
        "num_return_sequences": NUM_RETURN_SEQUENCES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": DO_SAMPLE,
        "use_4bit_quantization": USE_4BIT_QUANTIZATION,
        "quantization_config": QUANTIZATION_CONFIG
    }

def get_tokenizer_config():
    """Get tokenizer configuration dictionary"""
    return {
        "model_path": TOKENIZER_MODEL_PATH,
        "vocab_path": TOKENIZER_VOCAB_PATH,
        "vocab_size": SP_VOCAB_SIZE,
        "model_type": SP_MODEL_TYPE,
        "character_coverage": SP_CHARACTER_COVERAGE
    }

def get_api_config():
    """Get API configuration dictionary"""
    return {
        "host": API_HOST,
        "port": API_PORT,
        "title": API_TITLE,
        "description": API_DESCRIPTION,
        "version": API_VERSION,
        "debug": DEBUG_MODE
    }

def get_logging_config():
    """Get logging configuration dictionary"""
    return {
        "level": LOG_LEVEL,
        "format": LOG_FORMAT,
        "detailed_format": DETAILED_LOG_FORMAT,
        "log_file": LOG_FILE,
        "log_file_exists": os.path.exists(LOG_FILE) if LOG_FILE else False
    }

def print_startup_info():
    """Print startup information to console"""
    print("\n" + "=" * 80)
    print(f"[START] {API_TITLE} v{API_VERSION}")
    print("=" * 80)
    print(f"[CONFIG] Host: {API_HOST}:{API_PORT}")
    print(f"[CONFIG] Debug Mode: {DEBUG_MODE}")
    print(f"[CONFIG] Log Level: {LOG_LEVEL}")
    print(f"[CONFIG] Log File: {LOG_FILE}")
    print(f"[CONFIG] Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    print(f"[CONFIG] Model name (fallback): {MODEL_NAME}")
    print(f"[CONFIG] Tokenizer (SentencePiece): {TOKENIZER_MODEL_PATH}")
    if MODEL_PATH:
        print(f"[CONFIG] Local model path: {MODEL_PATH}")
    print(f"[CONFIG] 4-bit Quantization: {'Enabled' if USE_4BIT_QUANTIZATION else 'Disabled'}")
    print("=" * 80)

# Environment-specific overrides
ENV = os.getenv("ENVIRONMENT", "development").lower()

if ENV == "production":
    DEBUG_MODE = False
    API_HOST = "0.0.0.0"
    LOG_LEVEL = "WARNING"
    print("[ENV] Production environment detected")
elif ENV == "development":
    DEBUG_MODE = True
    LOG_LEVEL = LOG_LEVEL or "DEBUG"
    print("[ENV] Development environment detected")
elif ENV == "testing":
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    API_PORT = 8001
    print("[ENV] Testing environment detected")

# Override settings from environment variables if present
API_HOST = os.getenv("API_HOST", API_HOST)
API_PORT = int(os.getenv("API_PORT", API_PORT))
LOG_LEVEL = os.getenv("LOG_LEVEL", LOG_LEVEL).upper()
DEBUG_MODE = os.getenv("DEBUG_MODE", str(DEBUG_MODE)).lower() == "true"

# Validate LOG_LEVEL
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in VALID_LOG_LEVELS:
    print(f"⚠️  Invalid LOG_LEVEL '{LOG_LEVEL}', defaulting to 'INFO'")
    LOG_LEVEL = "INFO"

# Print configuration on import
if os.getenv("SILENT_START", "false").lower() != "true":
    print_startup_info()
