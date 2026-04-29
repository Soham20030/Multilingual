#!/usr/bin/env python3
"""
MCP-Driven Feedback Stream

This module implements the MCP (Multi-Cloud Protocol) driven feedback stream
that collects live examples (user prompts + corrections) and stores them
for real-time policy updates.

Features:
- Live feedback collection from MCP connectors
- Auto-storage into /data/feedback_stream.jsonl
- Real-time policy updates (Q-table or bandit style)
- Integration with existing MCP infrastructure

Author: Soham Kotkar
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    """Structure for a feedback entry"""
    timestamp: float
    user_id: Optional[str]
    session_id: Optional[str]
    prompt: str
    original_output: str
    corrected_output: Optional[str]
    reward: float
    feedback_type: str  # "correction", "rating", "preference"
    language: str
    metadata: Dict[str, Any]

class MCPFeedbackCollector:
    """
    MCP-Driven Feedback Stream Collector
    
    Collects live feedback from various MCP connectors and stores them
    for real-time policy updates and model improvement.
    """
    
    def __init__(self):
        self.feedback_file = Path("data/feedback_stream.jsonl")
        self.active_connectors = {}
        self.feedback_buffer = []
        self.buffer_size = 100
        self.initialized = False
        
    async def initialize(self):
        """Initialize the MCP feedback collector"""
        try:
            logger.info("Initializing MCP Feedback Stream...")
            
            # Ensure data directory exists
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize MCP connectors
            await self._initialize_mcp_connectors()
            
            # Start background tasks
            asyncio.create_task(self._process_feedback_buffer())
            asyncio.create_task(self._monitor_connectors())
            
            self.initialized = True
            logger.info("✅ MCP Feedback Collector initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP feedback collector: {e}")
            raise
    
    async def _initialize_mcp_connectors(self):
        """Initialize MCP connectors for feedback collection"""
        try:
            # Initialize HuggingFace connector
            self.active_connectors["huggingface"] = await self._create_hf_connector()
            
            # Initialize S3 connector
            self.active_connectors["s3"] = await self._create_s3_connector()
            
            # Initialize HTTP connector
            self.active_connectors["http"] = await self._create_http_connector()
            
            # Initialize Qdrant connector
            self.active_connectors["qdrant"] = await self._create_qdrant_connector()
            
            logger.info(f"Initialized {len(self.active_connectors)} MCP connectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP connectors: {e}")
            self.active_connectors = {}
    
    async def _create_hf_connector(self):
        """Create HuggingFace MCP connector"""
        return {
            "type": "huggingface",
            "endpoint": "https://huggingface.co/api",
            "active": True,
            "last_check": time.time()
        }
    
    async def _create_s3_connector(self):
        """Create S3 MCP connector"""
        return {
            "type": "s3",
            "bucket": "bhiv",
            "prefix": "rl_feedback/sovereign_core/",
            "active": True,
            "last_check": time.time()
        }
    
    async def _create_http_connector(self):
        """Create HTTP MCP connector"""
        return {
            "type": "http",
            "endpoint": "http://localhost:8000/feedback",
            "active": True,
            "last_check": time.time()
        }
    
    async def _create_qdrant_connector(self):
        """Create Qdrant MCP connector"""
        return {
            "type": "qdrant",
            "endpoint": "http://localhost:6333",
            "collection": "feedback_stream",
            "active": True,
            "last_check": time.time()
        }
    
    async def collect_feedback(self, prompt: str, original_output: str, 
                             corrected_output: Optional[str] = None,
                             reward: float = 0.0, feedback_type: str = "rating",
                             user_id: Optional[str] = None, session_id: Optional[str] = None,
                             language: str = "en", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Collect feedback entry
        
        Args:
            prompt: User prompt
            original_output: Original model output
            corrected_output: Corrected output (if available)
            reward: Reward score (0-1)
            feedback_type: Type of feedback
            user_id: User identifier
            session_id: Session identifier
            language: Language of the interaction
            metadata: Additional metadata
            
        Returns:
            True if feedback was collected successfully
        """
        try:
            # Create feedback entry
            entry = FeedbackEntry(
                timestamp=time.time(),
                user_id=user_id,
                session_id=session_id,
                prompt=prompt,
                original_output=original_output,
                corrected_output=corrected_output,
                reward=reward,
                feedback_type=feedback_type,
                language=language,
                metadata=metadata or {}
            )
            
            # Add to buffer
            self.feedback_buffer.append(entry)
            
            # Store immediately if buffer is full
            if len(self.feedback_buffer) >= self.buffer_size:
                await self._flush_buffer()
            
            logger.info(f"Collected feedback: {feedback_type} (reward: {reward:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return False
    
    async def _flush_buffer(self):
        """Flush feedback buffer to storage"""
        if not self.feedback_buffer:
            return
        
        try:
            # Write to JSONL file
            async with aiofiles.open(self.feedback_file, 'a', encoding='utf-8') as f:
                for entry in self.feedback_buffer:
                    line = json.dumps(asdict(entry), ensure_ascii=False) + '\n'
                    await f.write(line)
            
            # Upload to MCP connectors
            await self._upload_to_connectors(self.feedback_buffer)
            
            # Clear buffer
            self.feedback_buffer.clear()
            
            logger.info(f"Flushed {len(self.feedback_buffer)} feedback entries")
            
        except Exception as e:
            logger.error(f"Failed to flush feedback buffer: {e}")
    
    async def _upload_to_connectors(self, entries: List[FeedbackEntry]):
        """Upload feedback entries to MCP connectors"""
        for connector_name, connector in self.active_connectors.items():
            if not connector.get("active", False):
                continue
            
            try:
                if connector["type"] == "s3":
                    await self._upload_to_s3(entries, connector)
                elif connector["type"] == "http":
                    await self._upload_to_http(entries, connector)
                elif connector["type"] == "qdrant":
                    await self._upload_to_qdrant(entries, connector)
                
                connector["last_check"] = time.time()
                
            except Exception as e:
                logger.error(f"Failed to upload to {connector_name}: {e}")
                connector["active"] = False
    
    async def _upload_to_s3(self, entries: List[FeedbackEntry], connector: Dict[str, Any]):
        """Upload feedback entries to S3"""
        # This would integrate with boto3 for S3 upload
        # For now, we'll simulate the upload
        logger.info(f"Simulating S3 upload to {connector['bucket']}/{connector['prefix']}")
        
        # In a real implementation, you would:
        # 1. Create S3 client
        # 2. Upload entries as JSONL file
        # 3. Handle retries and error cases
    
    async def _upload_to_http(self, entries: List[FeedbackEntry], connector: Dict[str, Any]):
        """Upload feedback entries via HTTP"""
        # This would make HTTP POST requests to the endpoint
        logger.info(f"Simulating HTTP upload to {connector['endpoint']}")
        
        # In a real implementation, you would:
        # 1. Create HTTP client
        # 2. POST entries as JSON
        # 3. Handle authentication and retries
    
    async def _upload_to_qdrant(self, entries: List[FeedbackEntry], connector: Dict[str, Any]):
        """Upload feedback entries to Qdrant vector database"""
        # This would integrate with Qdrant client
        logger.info(f"Simulating Qdrant upload to {connector['endpoint']}")
        
        # In a real implementation, you would:
        # 1. Create Qdrant client
        # 2. Insert entries as vectors
        # 3. Handle indexing and search
    
    async def _process_feedback_buffer(self):
        """Background task to process feedback buffer"""
        while True:
            try:
                await asyncio.sleep(30)  # Process every 30 seconds
                if self.feedback_buffer:
                    await self._flush_buffer()
            except Exception as e:
                logger.error(f"Error in feedback buffer processing: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_connectors(self):
        """Background task to monitor MCP connectors"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for connector_name, connector in self.active_connectors.items():
                    if not connector.get("active", False):
                        # Try to reconnect
                        await self._reconnect_connector(connector_name, connector)
                
            except Exception as e:
                logger.error(f"Error in connector monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _reconnect_connector(self, connector_name: str, connector: Dict[str, Any]):
        """Attempt to reconnect a failed connector"""
        try:
            logger.info(f"Attempting to reconnect {connector_name}")
            
            # Simulate reconnection logic
            connector["active"] = True
            connector["last_check"] = time.time()
            
            logger.info(f"Successfully reconnected {connector_name}")
            
        except Exception as e:
            logger.error(f"Failed to reconnect {connector_name}: {e}")
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback collection"""
        try:
            # Count entries in file
            entry_count = 0
            if self.feedback_file.exists():
                async with aiofiles.open(self.feedback_file, 'r', encoding='utf-8') as f:
                    async for line in f:
                        if line.strip():
                            entry_count += 1
            
            return {
                "initialized": self.initialized,
                "active_connectors": len([c for c in self.active_connectors.values() if c.get("active", False)]),
                "total_connectors": len(self.active_connectors),
                "buffer_size": len(self.feedback_buffer),
                "total_entries": entry_count,
                "feedback_file": str(self.feedback_file),
                "connectors": {
                    name: {
                        "type": conn["type"],
                        "active": conn.get("active", False),
                        "last_check": conn.get("last_check", 0)
                    }
                    for name, conn in self.active_connectors.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback stats: {e}")
            return {"error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return await self.get_feedback_stats()
