#!/usr/bin/env python3
"""
Sovereign LM Bridge + Multilingual KSML Core API

This is the main FastAPI application that provides the sovereign multilingual 
reasoning bridge connecting the LM Core, Vaani TTS, and Gurukul/Uniguru front-end.

Core Endpoints:
- /align.ksml - KSML semantic alignment engine
- /rl.feedback - RL self-improvement loop
- /compose.speech_ready - Vaani compatibility layer
- /bridge.reason - Multilingual reasoning bridge

Author: Soham Kotkar
Duration: Oct 28 – Nov 2
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root and this package directory to sys.path so local modules resolve
project_root = Path(__file__).parent.parent
package_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(package_root))

# Load environment variables from project-root .env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / ".env")
except Exception:
    # dotenv not installed or .env missing; ignore and continue
    pass

# Import existing services
from src.services.knowledge_base import process_qa_query
from src.integration.tts_integration import VaaniTTSIntegration
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Sovereign LM Bridge + Multilingual KSML",
    description="Sovereign multilingual reasoning bridge with KSML alignment, RL feedback, and Vaani integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state
_ksml_aligner = None
_mcp_feedback = None
_rl_policy = None
_vaani_composer = None
_bridge_reasoner = None

# =============================================================================
# Request/Response Models
# =============================================================================

class KSMLAlignmentRequest(BaseModel):
    """Request model for KSML semantic alignment"""
    text: str = Field(..., description="Raw LM text from the LM Core system")
    source_lang: Optional[str] = Field(None, description="Source language (auto-detect if not provided)")
    target_lang: Optional[str] = Field("en", description="Target language")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class KSMLAlignmentResponse(BaseModel):
    """Response model for KSML semantic alignment"""
    intent: str
    source_lang: str
    target_lang: str
    karma_state: str  # sattva/rajas/tamas
    semantic_roots: List[str]  # Sanskrit roots
    confidence: float
    processing_time: float

class RLFeedbackRequest(BaseModel):
    """Request model for RL feedback"""
    prompt: str = Field(..., description="User prompt")
    output: str = Field(..., description="Model output")
    reward: float = Field(..., ge=0.0, le=1.0, description="Reward score (0-1)")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

class RLFeedbackResponse(BaseModel):
    """Response model for RL feedback"""
    status: str
    policy_updated: bool
    reward_logged: bool
    s3_synced: bool
    processing_time: float

class SpeechReadyRequest(BaseModel):
    """Request model for speech-ready text composition"""
    text: str = Field(..., description="Aligned text to convert")
    language: str = Field(..., description="Target language")
    tone: Optional[str] = Field("calm", description="Speech tone")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class SpeechReadyResponse(BaseModel):
    """Response model for speech-ready text"""
    text: str
    tone: str
    lang: str
    prosody_hint: str
    audio_metadata: Optional[Dict[str, Any]] = None
    processing_time: float

class BridgeReasonRequest(BaseModel):
    """Request model for multilingual reasoning bridge"""
    text: str = Field(..., description="Input text")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    include_audio: bool = Field(True, description="Include TTS audio")

class BridgeReasonResponse(BaseModel):
    """Response model for multilingual reasoning bridge"""
    aligned_text: str
    ksml_metadata: Dict[str, Any]
    speech_ready: Optional[Dict[str, Any]] = None
    processing_time: float
    trace_id: str

# =============================================================================
# Initialization Functions
# =============================================================================

async def initialize_components():
    """Initialize all sovereign core components"""
    global _ksml_aligner, _mcp_feedback, _rl_policy, _vaani_composer, _bridge_reasoner
    
    try:
        logger.info("Initializing Sovereign Core components...")
        
        # Initialize KSML aligner
        from ksml.aligner import KSMLAligner
        _ksml_aligner = KSMLAligner()
        await _ksml_aligner.initialize()
        
        # Initialize MCP feedback collector
        from mcp.feedback_stream import MCPFeedbackCollector
        _mcp_feedback = MCPFeedbackCollector()
        await _mcp_feedback.initialize()
        
        # Initialize RL policy updater
        from rl.policy_updater import RLPolicyUpdater
        _rl_policy = RLPolicyUpdater()
        await _rl_policy.initialize()
        
        # Initialize Vaani composer
        from vaani.speech_composer import VaaniSpeechComposer
        _vaani_composer = VaaniSpeechComposer()
        await _vaani_composer.initialize()
        
        # Initialize bridge reasoner
        from bridge.reasoner import MultilingualReasoner
        _bridge_reasoner = MultilingualReasoner()
        await _bridge_reasoner.initialize()
        
        logger.info("✅ All Sovereign Core components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise

# =============================================================================
# Core API Endpoints
# =============================================================================

@app.post("/align.ksml", response_model=KSMLAlignmentResponse)
async def align_ksml(request: KSMLAlignmentRequest):
    """
    KSML Semantic Alignment Engine
    
    Accepts raw LM text from the LM Core system and adds:
    - Intent classification
    - Language detection (source/target)
    - Karma state classification (sattva/rajas/tamas)
    - Sanskrit root tagging
    """
    start_time = time.time()
    
    try:
        if not _ksml_aligner:
            raise HTTPException(status_code=503, detail="KSML aligner not initialized")
        
        # Perform KSML alignment
        result = await _ksml_aligner.align_text(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            context=request.context
        )
        
        processing_time = time.time() - start_time
        
        return KSMLAlignmentResponse(
            intent=result["intent"],
            source_lang=result["source_lang"],
            target_lang=result["target_lang"],
            karma_state=result["karma_state"],
            semantic_roots=result["semantic_roots"],
            confidence=result["confidence"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"KSML alignment failed: {e}")
        raise HTTPException(status_code=500, detail=f"KSML alignment failed: {str(e)}")

@app.post("/rl.feedback", response_model=RLFeedbackResponse)
async def rl_feedback(request: RLFeedbackRequest):
    """
    RL Self-Improvement Loop
    
    Accepts { prompt, output, reward } and updates local adapter delta or policy table.
    Runs periodic reward-based adjustments and syncs logs to S3.
    """
    start_time = time.time()
    
    try:
        if not _rl_policy:
            raise HTTPException(status_code=503, detail="RL policy updater not initialized")
        
        # Process RL feedback
        result = await _rl_policy.process_feedback(
            prompt=request.prompt,
            output=request.output,
            reward=request.reward,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        processing_time = time.time() - start_time
        
        return RLFeedbackResponse(
            status="success",
            policy_updated=result["policy_updated"],
            reward_logged=result["reward_logged"],
            s3_synced=result["s3_synced"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"RL feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"RL feedback failed: {str(e)}")

@app.post("/compose.speech_ready", response_model=SpeechReadyResponse)
async def compose_speech_ready(request: SpeechReadyRequest):
    """
    Vaani Compatibility Layer
    
    Converts aligned text to prosody-optimized JSON for Karthikeya's TTS engine.
    """
    start_time = time.time()
    
    try:
        if not _vaani_composer:
            raise HTTPException(status_code=503, detail="Vaani composer not initialized")
        
        # Compose speech-ready text
        result = await _vaani_composer.compose_speech(
            text=request.text,
            language=request.language,
            tone=request.tone,
            context=request.context
        )
        
        processing_time = time.time() - start_time
        
        return SpeechReadyResponse(
            text=result["text"],
            tone=result["tone"],
            lang=result["lang"],
            prosody_hint=result["prosody_hint"],
            audio_metadata=result.get("audio_metadata"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Speech composition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech composition failed: {str(e)}")

@app.post("/bridge.reason", response_model=BridgeReasonResponse)
async def bridge_reason(request: BridgeReasonRequest):
    """
    =============================================================================
    MAIN ENDPOINT: MULTILINGUAL REASONING BRIDGE
    =============================================================================
    TODO: This is the main endpoint that Bhavesh's system will call
    TODO: Confirm request/response format with Bhavesh
    TODO: Test integration with Bhavesh's system
    TODO: Handle error cases and timeouts
    TODO: Monitor performance metrics
    """
    start_time = time.time()
    trace_id = f"bridge_{int(time.time() * 1000)}"
    
    try:
        if not _bridge_reasoner:
            raise HTTPException(status_code=503, detail="Bridge reasoner not initialized")
        
        # Process through complete bridge pipeline
        result = await _bridge_reasoner.process_reasoning(
            text=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            include_audio=request.include_audio,
            trace_id=trace_id
        )
        
        processing_time = time.time() - start_time
        
        return BridgeReasonResponse(
            aligned_text=result["aligned_text"],
            ksml_metadata=result["ksml_metadata"],
            speech_ready=result.get("speech_ready"),
            processing_time=processing_time,
            trace_id=trace_id
        )
        
    except Exception as e:
        logger.error(f"Bridge reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bridge reasoning failed: {str(e)}")

# =============================================================================
# System Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        components_status = {
            "ksml_aligner": _ksml_aligner is not None,
            "mcp_feedback": _mcp_feedback is not None,
            "rl_policy": _rl_policy is not None,
            "vaani_composer": _vaani_composer is not None,
            "bridge_reasoner": _bridge_reasoner is not None
        }
        
        all_healthy = all(components_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "components": components_status,
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - start_time if 'start_time' in globals() else 0,
            "components": {}
        }
        
        # Get stats from each component
        if _ksml_aligner:
            stats["components"]["ksml"] = await _ksml_aligner.get_stats()
        if _mcp_feedback:
            stats["components"]["mcp"] = await _mcp_feedback.get_stats()
        if _rl_policy:
            stats["components"]["rl"] = await _rl_policy.get_stats()
        if _vaani_composer:
            stats["components"]["vaani"] = await _vaani_composer.get_stats()
        if _bridge_reasoner:
            stats["components"]["bridge"] = await _bridge_reasoner.get_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

# =============================================================================
# Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global start_time
    start_time = time.time()
    
    logger.info("🚀 Starting Sovereign LM Bridge + Multilingual KSML Core")
    await initialize_components()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔌 Shutting down Sovereign Core")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("SOVEREIGN_HOST", "127.0.0.1")
    port = int(os.getenv("SOVEREIGN_PORT", "8116"))
    
    logger.info(f"Starting Sovereign Core API on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,  # Disable reload for production stability
        log_level="info"
    )
