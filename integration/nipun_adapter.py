from typing import Dict, Any, List, Union
import uuid
import asyncio
import re
from datetime import datetime
from utils.logger import get_logger
from reinforcement.reward_functions import get_reward_from_output
from reinforcement.replay_buffer import replay_buffer
from config.settings import MONGO_CONFIG
import motor.motor_asyncio
import pymongo
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

logger = get_logger(__name__)

# Initialize MongoDB client
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
mongo_db = mongo_client[MONGO_CONFIG['database']]
nlo_collection = mongo_db['nlo_collection']

# Initialize NLP models
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    # Try to load spaCy model, fallback if not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        nlp = None
except Exception as e:
    logger.error(f"Error initializing NLP models: {str(e)}")
    sentence_model = None
    stop_words = set()
    nlp = None

def extract_bloom_taxonomy_level(text: str) -> str:
    """Extract Bloom's taxonomy level from text using NLP analysis."""
    if not text:
        return "remember"

    text_lower = text.lower()

    # Define Bloom's taxonomy keywords
    bloom_keywords = {
        "remember": ["recall", "recognize", "list", "describe", "identify", "retrieve", "name", "locate", "find"],
        "understand": ["explain", "interpret", "summarize", "infer", "paraphrase", "classify", "compare", "exemplify"],
        "apply": ["execute", "implement", "solve", "use", "demonstrate", "operate", "schedule", "sketch"],
        "analyze": ["differentiate", "organize", "relate", "compare", "contrast", "distinguish", "examine", "experiment", "question", "test"],
        "evaluate": ["appraise", "argue", "defend", "judge", "select", "support", "value", "critique", "weigh"],
        "create": ["design", "assemble", "construct", "conjecture", "develop", "formulate", "author", "investigate", "compose", "plan"]
    }

    # Score each level based on keyword presence
    level_scores = {}
    for level, keywords in bloom_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        level_scores[level] = score

    # Return the level with highest score, default to "understand"
    if max(level_scores.values()) > 0:
        return max(level_scores, key=level_scores.get)
    return "understand"

def extract_subject_tags(text: str) -> List[str]:
    """Extract subject tags from text using NLP."""
    if not text or not nlp:
        return ["general"]

    try:
        doc = nlp(text)

        # Extract named entities and noun phrases
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "EVENT", "WORK_OF_ART"]]
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]

        # Combine and filter
        all_tags = entities + noun_phrases

        # Filter out common words and short tags
        filtered_tags = []
        for tag in all_tags:
            if len(tag) > 2 and tag not in stop_words and not tag.isdigit():
                filtered_tags.append(tag)

        # Return top 5 most relevant tags
        return filtered_tags[:5] if filtered_tags else ["general"]

    except Exception as e:
        logger.error(f"Error extracting subject tags: {str(e)}")
        return ["general"]

def aggregate_multi_input_metadata(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metadata from multiple input outputs."""
    if not outputs:
        return {}

    # Aggregate keywords
    all_keywords = []
    for output in outputs:
        keywords = output.get('keywords', [])
        if isinstance(keywords, list):
            all_keywords.extend(keywords)

    # Remove duplicates while preserving order
    unique_keywords = list(dict.fromkeys(all_keywords))

    # Calculate average confidence
    confidences = [output.get('confidence', 0.8) for output in outputs if 'confidence' in output]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.8

    # Combine results
    combined_result = " ".join([output.get('result', '') for output in outputs if output.get('result')])

    # Determine content types
    content_types = [output.get('content_type', 'text') for output in outputs]
    unique_content_types = list(set(content_types))

    return {
        'combined_keywords': unique_keywords,
        'average_confidence': round(avg_confidence, 2),
        'combined_result': combined_result,
        'content_types': unique_content_types,
        'input_count': len(outputs)
    }

async def ensure_nlo_indexes():
    """Ensure MongoDB indexes exist for NLO collection."""
    try:
        # Create indexes for efficient querying
        await nlo_collection.create_index("task_id")
        await nlo_collection.create_index("subject_tag")
        await nlo_collection.create_index("timestamp")
        await nlo_collection.create_index([("task_id", 1), ("subject_tag", 1)])
        logger.info("NLO collection indexes ensured")
    except Exception as e:
        logger.error(f"Error creating NLO indexes: {str(e)}")

async def store_nlo_in_mongodb(nlo: Dict[str, Any], task_id: str) -> bool:
    """Store NLO in MongoDB with proper indexing."""
    try:
        # Add metadata
        nlo_document = {
            **nlo,
            "task_id": task_id,
            "timestamp": datetime.now(),
            "version": "2.0"  # Version for schema tracking
        }

        # Ensure indexes exist
        await ensure_nlo_indexes()

        # Insert document
        result = await nlo_collection.insert_one(nlo_document)
        logger.info(f"NLO stored in MongoDB with ID: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing NLO in MongoDB: {str(e)}")
        return False

def map_output_to_learning(output: Dict[str, Any], task_id: str = None) -> Dict[str, Any]:
    """Map agent output to a Named Learning Object (NLO) with enhanced features."""
    task_id = task_id or str(uuid.uuid4())

    try:
        # Handle both single output and multi-input scenarios
        if isinstance(output, list):
            return map_multi_input_to_learning(output, task_id)

        # Extract basic information with defaults
        result = output.get('result', '')
        model = output.get('model', 'unknown')
        keywords = output.get('keywords', [])

        logger.info(f"Mapping output to NLO for model: {model}, task_id: {task_id}")

        # Dynamic Bloom's taxonomy extraction
        bloom_level = extract_bloom_taxonomy_level(result)

        # Dynamic subject tag extraction
        subject_tags = extract_subject_tags(result)
        primary_subject_tag = subject_tags[0] if subject_tags else 'general'

        # Enhanced NLO structure
        nlo = {
            "task_id": task_id,
            "summary": result,
            "tags": keywords if keywords else subject_tags,
            "bloom_level": bloom_level,
            "model": model,
            "content_type": output.get('content_type', 'text'),
            "confidence": output.get('confidence', 0.8),
            "difficulty": output.get('difficulty', 'medium'),
            "subject_tag": primary_subject_tag,
            "subject_tags": subject_tags,
            "processing_time": output.get('processing_time', 0),
            "tokens_used": output.get('tokens_used', 0),
            "cost_estimate": output.get('cost_estimate', 0.0),
            "agent": output.get('agent', 'unknown'),
            "input_length": len(str(output.get('input', ''))),
            "output_length": len(result),
            "timestamp": datetime.now().isoformat()
        }

        # Calculate reward
        reward = get_reward_from_output(output, task_id)
        nlo['reward'] = reward

        # Store in MongoDB asynchronously
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(store_nlo_in_mongodb(nlo, task_id))
        except RuntimeError:
            # If no event loop is running, create a new one
            asyncio.run(store_nlo_in_mongodb(nlo, task_id))

        # Add to replay buffer
        replay_buffer.add_run(task_id, output.get('input', ''), nlo, output.get('agent', 'unknown'), model, reward)

        logger.info(f"Enhanced NLO created: task_id={task_id}, bloom_level={bloom_level}, subject_tag={primary_subject_tag}")
        return nlo

    except Exception as e:
        logger.error(f"Error mapping to NLO: {str(e)}")
        # Create error NLO with defaults
        error_nlo = {
            "task_id": task_id,
            "error": f"NLO mapping failed: {str(e)}",
            "status": 500,
            "summary": "",
            "tags": [],
            "bloom_level": "remember",
            "model": output.get('model', 'unknown'),
            "content_type": output.get('content_type', 'text'),
            "confidence": 0.0,
            "difficulty": "medium",
            "subject_tag": "error",
            "subject_tags": ["error"],
            "agent": output.get('agent', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }

        reward = get_reward_from_output(error_nlo, task_id)
        error_nlo['reward'] = reward
        replay_buffer.add_run(task_id, output.get('input', ''), error_nlo, output.get('agent', 'unknown'), output.get('model', 'unknown'), reward)
        return error_nlo

def map_multi_input_to_learning(outputs: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    """Map multiple agent outputs to a single aggregated NLO."""
    try:
        if not outputs:
            raise ValueError("No outputs provided for multi-input mapping")

        logger.info(f"Mapping {len(outputs)} outputs to aggregated NLO for task_id: {task_id}")

        # Aggregate metadata from all outputs
        aggregated = aggregate_multi_input_metadata(outputs)

        # Combine all results for analysis
        combined_text = aggregated['combined_result']

        # Extract enhanced features from combined text
        bloom_level = extract_bloom_taxonomy_level(combined_text)
        subject_tags = extract_subject_tags(combined_text)
        primary_subject_tag = subject_tags[0] if subject_tags else 'general'

        # Get models and agents from all outputs
        models = [output.get('model', 'unknown') for output in outputs]
        agents = [output.get('agent', 'unknown') for output in outputs]

        # Create aggregated NLO
        nlo = {
            "task_id": task_id,
            "summary": combined_text,
            "tags": aggregated['combined_keywords'],
            "bloom_level": bloom_level,
            "models": models,
            "primary_model": models[0] if models else 'unknown',
            "content_types": aggregated['content_types'],
            "primary_content_type": "multi" if len(aggregated['content_types']) > 1 else aggregated['content_types'][0],
            "confidence": aggregated['average_confidence'],
            "difficulty": "medium",  # Could be enhanced with difficulty analysis
            "subject_tag": primary_subject_tag,
            "subject_tags": subject_tags,
            "agents": agents,
            "primary_agent": agents[0] if agents else 'unknown',
            "input_count": aggregated['input_count'],
            "total_tokens": sum(output.get('tokens_used', 0) for output in outputs),
            "total_cost": sum(output.get('cost_estimate', 0.0) for output in outputs),
            "processing_times": [output.get('processing_time', 0) for output in outputs],
            "total_processing_time": sum(output.get('processing_time', 0) for output in outputs),
            "timestamp": datetime.now().isoformat(),
            "is_multi_input": True
        }

        # Calculate aggregated reward
        total_reward = sum(get_reward_from_output(output, task_id) for output in outputs)
        avg_reward = total_reward / len(outputs)
        nlo['reward'] = avg_reward
        nlo['total_reward'] = total_reward

        # Store in MongoDB
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(store_nlo_in_mongodb(nlo, task_id))
        except RuntimeError:
            asyncio.run(store_nlo_in_mongodb(nlo, task_id))

        # Add to replay buffer
        replay_buffer.add_run(task_id, f"multi_input_{len(outputs)}_files", nlo, nlo['primary_agent'], nlo['primary_model'], avg_reward)

        logger.info(f"Multi-input NLO created: task_id={task_id}, inputs={len(outputs)}, bloom_level={bloom_level}")
        return nlo

    except Exception as e:
        logger.error(f"Error mapping multi-input to NLO: {str(e)}")
        error_nlo = {
            "task_id": task_id,
            "error": f"Multi-input NLO mapping failed: {str(e)}",
            "status": 500,
            "summary": "",
            "tags": [],
            "bloom_level": "remember",
            "models": ["unknown"],
            "primary_model": "unknown",
            "content_types": ["error"],
            "primary_content_type": "error",
            "confidence": 0.0,
            "difficulty": "medium",
            "subject_tag": "error",
            "subject_tags": ["error"],
            "agents": ["unknown"],
            "primary_agent": "unknown",
            "is_multi_input": True,
            "timestamp": datetime.now().isoformat()
        }
        reward = get_reward_from_output(error_nlo, task_id)
        error_nlo['reward'] = reward
        replay_buffer.add_run(task_id, "multi_input_error", error_nlo, "unknown", "unknown", reward)
        return error_nlo

async def get_nlos_by_subject(subject_tag: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve NLOs by subject tag from MongoDB."""
    try:
        cursor = nlo_collection.find({"subject_tag": subject_tag}).limit(limit)
        nlos = await cursor.to_list(length=limit)
        return nlos
    except Exception as e:
        logger.error(f"Error retrieving NLOs by subject: {str(e)}")
        return []

async def get_nlos_by_task_id(task_id: str) -> Dict[str, Any]:
    """Retrieve NLO by task ID from MongoDB."""
    try:
        nlo = await nlo_collection.find_one({"task_id": task_id})
        return nlo if nlo else {}
    except Exception as e:
        logger.error(f"Error retrieving NLO by task ID: {str(e)}")
        return {}

if __name__ == "__main__":
    # Test single input
    test_output = {
        "result": "This text analyzes the impact of artificial intelligence on education systems.",
        "model": "edumentor_agent",
        "keywords": ["AI", "education"],
        "input": "Sample input about AI in education",
        "confidence": 0.9,
        "processing_time": 2.5
    }
    result = map_output_to_learning(test_output)
    print("Single input result:", result)

    # Test multi-input
    multi_outputs = [
        {"result": "PDF analysis shows AI trends", "model": "archive_agent", "content_type": "pdf", "keywords": ["AI", "trends"]},
        {"result": "Image shows AI diagram", "model": "image_agent", "content_type": "image", "keywords": ["AI", "diagram"]},
        {"result": "Audio discusses AI applications", "model": "audio_agent", "content_type": "audio", "keywords": ["AI", "applications"]}
    ]
    multi_result = map_output_to_learning(multi_outputs, "test_multi_task")
    print("Multi-input result:", multi_result)
