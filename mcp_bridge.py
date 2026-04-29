#!/usr/bin/env python3
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from agents.agent_registry import agent_registry
from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from reinforcement.reward_functions import get_reward_from_output
from agents.agent_memory_handler import agent_memory_handler
from utils.mongo_logger import mongo_logger
import uuid
import importlib
import requests
import motor.motor_asyncio
from config.settings import MONGO_CONFIG, TIMEOUT_CONFIG
import shutil
import os
from tempfile import NamedTemporaryFile

logger = get_logger(__name__)
app = FastAPI(title="BHIV Core MCP Bridge", version="2.0.0")

# Async MongoDB client
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
mongo_db = mongo_client[MONGO_CONFIG['database']]
mongo_collection = mongo_db[MONGO_CONFIG['collection']]

# Health check data
health_status = {
    "startup_time": datetime.now(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "agent_status": {}
}

class TaskPayload(BaseModel):
    agent: str
    input: str
    pdf_path: str = ""
    input_type: str = "text"
    retries: int = 3
    fallback_model: str = "edumentor_agent"

async def handle_task_request(payload: TaskPayload) -> dict:
    """Handle task request with agent routing."""
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    logger.info(f"[MCP_BRIDGE] Task ID: {task_id} | Agent: {payload.agent} | Input: {payload.input[:50]}... | File: {payload.pdf_path} | Type: {payload.input_type}")
    
    try:
        # Find appropriate agent based on input type
        task_context = {
            "task": "process",
            "keywords": [payload.agent, payload.input_type],
            "model": payload.agent,
            "input_type": payload.input_type
        }
        
        # Route to specific agents based on input type
        if payload.input_type == "image":
            agent_id = "image_agent"
        elif payload.input_type == "audio":
            agent_id = "audio_agent"
        elif payload.input_type == "pdf":
            agent_id = "archive_agent"
        else:
            agent_id = agent_registry.find_agent(task_context)
        
        agent_config = agent_registry.get_agent_config(agent_id)
        
        if not agent_config:
            logger.warning(f"[MCP_BRIDGE] Agent config not found for {agent_id}, trying direct import")
        
        # Route to appropriate handler based on connection type or direct import
        if agent_config and agent_config['connection_type'] == 'python_module':
            module_path = agent_config['module_path']
            class_name = agent_config['class_name']
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            agent = agent_class()
            # Pass the file path as input_path for file-based agents
            input_path = payload.pdf_path if payload.pdf_path else payload.input
            result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
        elif payload.input_type == "image":
            # Direct import for image agent
            from agents.image_agent import ImageAgent
            agent = ImageAgent()
            input_path = payload.pdf_path if payload.pdf_path else payload.input
            result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
        elif payload.input_type == "audio":
            # Direct import for audio agent
            from agents.audio_agent import AudioAgent
            agent = AudioAgent()
            input_path = payload.pdf_path if payload.pdf_path else payload.input
            result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
        elif payload.input_type == "pdf":
            # Direct import for archive agent
            from agents.archive_agent import ArchiveAgent
            agent = ArchiveAgent()
            input_path = payload.pdf_path if payload.pdf_path else payload.input
            result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
        elif agent_config and agent_config['connection_type'] == 'http_api':
            # Make HTTP request to external agent
            endpoint = agent_config['endpoint']
            headers = agent_config['headers']
            request_payload = {
                'query': payload.input,
                'user_id': 'bhiv_core',
                'task_id': task_id,
                'input_type': payload.input_type
            }
            if payload.pdf_path:
                request_payload['file_path'] = payload.pdf_path
            
            # Use appropriate timeout based on input type
            timeout = TIMEOUT_CONFIG.get('default_timeout', 120)
            if payload.input_type == 'image':
                timeout = TIMEOUT_CONFIG.get('image_processing_timeout', 180)
            elif payload.input_type == 'audio':
                timeout = TIMEOUT_CONFIG.get('audio_processing_timeout', 240)
            elif payload.input_type == 'pdf':
                timeout = TIMEOUT_CONFIG.get('pdf_processing_timeout', 150)

            response = requests.post(endpoint, json=request_payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()
        else:
            # Fallback to stream transformer agent
            from agents.stream_transformer_agent import StreamTransformerAgent
            agent = StreamTransformerAgent()
            input_path = payload.pdf_path if payload.pdf_path else payload.input
            result = agent.run(input_path, "", payload.agent, payload.input_type, task_id)
        
        # Enhanced logging with MongoDB and memory handler
        processing_time = (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0.0

        # Log to MongoDB with enhanced data
        task_log_data = {
            "task_id": task_id,
            "agent": payload.agent,
            "input": payload.input,
            "file_path": payload.pdf_path,
            "output": result,
            "timestamp": datetime.now(),
            "input_type": payload.input_type,
            "processing_time": processing_time,
            "success": result.get('status', 500) == 200
        }
        await mongo_collection.insert_one(task_log_data)

        # Log to enhanced MongoDB logger
        await mongo_logger.log_task_execution(task_log_data)

        # Log token/cost data if available
        if 'tokens_used' in result or 'cost_estimate' in result:
            await mongo_logger.log_token_cost({
                'task_id': task_id,
                'model': result.get('model', payload.agent),
                'agent': agent_id,
                'tokens_used': result.get('tokens_used', 0),
                'cost_estimate': result.get('cost_estimate', 0.0),
                'processing_time': processing_time
            })

        # Add to agent memory
        memory_entry = {
            'task_id': task_id,
            'input': payload.input,
            'output': result,
            'input_type': payload.input_type,
            'model': result.get('model', payload.agent),
            'status': result.get('status', 200),
            'response_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        agent_memory_handler.add_memory(agent_id, memory_entry)

        # Calculate reward and log to replay buffer
        reward = get_reward_from_output(result, task_id)
        replay_buffer.add_run(task_id, payload.input, result, agent_id, payload.agent, reward)

        return {"task_id": task_id, "agent_output": result, "status": 200}
    
    except Exception as e:
        logger.error(f"[MCP_BRIDGE] Error processing task {task_id}: {str(e)}")
        error_output = {"error": f"Task processing failed: {str(e)}", "status": 500}
        reward = get_reward_from_output(error_output, task_id)
        replay_buffer.add_run(task_id, payload.input, error_output, payload.agent, payload.agent, reward)
        return {"task_id": task_id, "agent_output": error_output, "status": 500}

@app.post("/handle_task")
async def handle_task(payload: TaskPayload):
    """Handle task via JSON payload."""
    return await handle_task_request(payload)

@app.post("/handle_task_with_file")
async def handle_task_with_file(
    agent: str = Form(...),
    input: str = Form(...),
    file: UploadFile = File(None),
    input_type: str = Form("text"),
    retries: int = Form(3),
    fallback_model: str = Form("edumentor_agent")
):
    """Handle task with file upload."""
    task_id = str(uuid.uuid4())
    temp_file_path = ""
    
    try:
        if file:
            # Save uploaded file to temp location
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file_path = temp_file.name
                shutil.copyfileobj(file.file, temp_file)
            
            logger.info(f"[MCP_BRIDGE] Task ID: {task_id} | File uploaded: {file.filename} -> {temp_file_path}")
        
        # Create payload with file path
        payload = TaskPayload(
            agent=agent,
            input=input,
            pdf_path=temp_file_path,
            input_type=input_type,
            retries=retries,
            fallback_model=fallback_model
        )
        
        # Process the task
        result = await handle_task_request(payload)
        
        return result
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        # Check MongoDB connection
        await mongo_client.admin.command('ping')
        mongodb_status = "healthy"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        mongodb_status = f"unhealthy: {str(e)}"

    # Check agent registry
    try:
        available_agents = len(agent_registry.agent_configs)
        agent_registry_status = "healthy"
    except Exception as e:
        logger.error(f"Agent registry health check failed: {str(e)}")
        available_agents = 0
        agent_registry_status = f"unhealthy: {str(e)}"

    # Calculate uptime
    uptime_seconds = (datetime.now() - health_status["startup_time"]).total_seconds()

    # Calculate success rate
    total_requests = health_status["total_requests"]
    success_rate = (health_status["successful_requests"] / total_requests * 100) if total_requests > 0 else 0

    health_data = {
        "status": "healthy" if mongodb_status == "healthy" and agent_registry_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "services": {
            "mongodb": mongodb_status,
            "agent_registry": agent_registry_status,
            "available_agents": available_agents
        },
        "metrics": {
            "total_requests": total_requests,
            "successful_requests": health_status["successful_requests"],
            "failed_requests": health_status["failed_requests"],
            "success_rate_percent": round(success_rate, 2)
        },
        "agent_status": health_status["agent_status"]
    }

    return health_data

@app.get("/config")
async def get_config():
    """Get current configuration."""
    try:
        return {
            "agents": agent_registry.agent_configs,
            "mongodb": {
                "database": MONGO_CONFIG["database"],
                "collection": MONGO_CONFIG["collection"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/reload")
async def reload_config():
    """Reload agent configuration dynamically."""
    try:
        # Reload agent registry
        importlib.reload(importlib.import_module('agents.agent_registry'))
        from agents.agent_registry import agent_registry as new_registry

        # Update global registry
        global agent_registry
        agent_registry = new_registry

        logger.info("Agent configuration reloaded successfully")
        return {
            "status": "success",
            "message": "Configuration reloaded",
            "agents": agent_registry.agent_configs,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get detailed system metrics."""
    try:
        # Get MongoDB stats
        db_stats = await mongo_db.command("dbStats")
        collection_stats = await mongo_collection.estimated_document_count()

        # Get recent task performance
        recent_tasks = await mongo_collection.find().sort("timestamp", -1).limit(100).to_list(100)

        # Calculate performance metrics
        processing_times = []
        agent_usage = {}

        for task in recent_tasks:
            if 'output' in task and isinstance(task['output'], dict):
                if 'processing_time' in task['output']:
                    processing_times.append(task['output']['processing_time'])

                agent = task.get('agent', 'unknown')
                agent_usage[agent] = agent_usage.get(agent, 0) + 1

        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "database": {
                "total_documents": collection_stats,
                "database_size_mb": round(db_stats.get("dataSize", 0) / 1024 / 1024, 2),
                "storage_size_mb": round(db_stats.get("storageSize", 0) / 1024 / 1024, 2)
            },
            "performance": {
                "avg_processing_time_seconds": round(avg_processing_time, 3),
                "total_tasks_processed": len(recent_tasks),
                "agent_usage": agent_usage
            },
            "system": health_status
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/handle_multi_task")
async def handle_multi_task(request: dict):
    """Handle multiple files/inputs asynchronously for improved performance."""
    try:
        files = request.get('files', [])
        agent = request.get('agent', 'edumentor_agent')
        task_type = request.get('task_type', 'summarize')

        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        logger.info(f"[MCP_BRIDGE] Multi-task request: {len(files)} files with agent {agent}")

        # Create tasks for async processing
        tasks = []
        for file_info in files:
            payload = TaskPayload(
                agent=agent,
                input=file_info.get('data', ''),
                pdf_path=file_info.get('path', ''),
                input_type=file_info.get('type', 'text'),
                retries=3,
                fallback_model='edumentor_agent'
            )
            tasks.append(handle_task_request(payload))

        # Process all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Process results and handle exceptions
        processed_results = []
        successful_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Multi-task file {i} failed: {str(result)}")
                processed_results.append({
                    "file_index": i,
                    "filename": files[i].get('path', f'file_{i}'),
                    "error": str(result),
                    "status": 500
                })
            else:
                processed_results.append({
                    "file_index": i,
                    "filename": files[i].get('path', f'file_{i}'),
                    "result": result,
                    "status": 200
                })
                successful_count += 1

        # Create aggregated response
        response = {
            "multi_task_id": str(uuid.uuid4()),
            "total_files": len(files),
            "successful_files": successful_count,
            "failed_files": len(files) - successful_count,
            "total_processing_time": total_time,
            "average_time_per_file": total_time / len(files),
            "results": processed_results,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[MCP_BRIDGE] Multi-task completed: {successful_count}/{len(files)} successful in {total_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"[MCP_BRIDGE] Multi-task failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)












