#!/usr/bin/env python3
"""
Web Interface for BHIV Core - Bootstrap UI with real-time results and authentication.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
import requests

# Simple logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from integration.nipun_adapter import get_nlos_by_subject, get_nlos_by_task_id
except ImportError:
    logger.warning("Nipun adapter not available")
    async def get_nlos_by_subject(*args, **kwargs): return []
    async def get_nlos_by_task_id(*args, **kwargs): return {}

import motor.motor_asyncio
try:
    from config.settings import MONGO_CONFIG, TIMEOUT_CONFIG
except ImportError:
    logger.warning("Config settings not available, using defaults")
    MONGO_CONFIG = {
        "uri": "mongodb://localhost:27017",
        "database": "bhiv_core",
        "collection": "task_logs"
    }
    TIMEOUT_CONFIG = {
        "default_timeout": 120,
        "image_processing_timeout": 180,
        "file_upload_timeout": 300
    }

# Initialize FastAPI app
app = FastAPI(title="BHIV Core Web Interface", version="1.0.0")

# Security setup
security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple user database (in production, use proper database)
USERS_DB = {
    "admin": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
    "user": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"   # secret
}

# Templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# MongoDB client
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_CONFIG['uri'])
mongo_db = mongo_client[MONGO_CONFIG['database']]

# Task storage for real-time updates
active_tasks = {}

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=10)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate user."""
    username = credentials.username
    if username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if not verify_password(credentials.password, USERS_DB[username]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return username

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: str = Depends(get_current_user)):
    """Home page with file upload interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": current_user,
        "title": "BHIV Core - AI Processing Pipeline"
    })

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: str = Depends(get_current_user)):
    """Dashboard with recent tasks and analytics."""
    # Get recent tasks from MongoDB
    recent_tasks = await mongo_db.task_logs.find().sort("timestamp", -1).limit(10).to_list(10)
    
    # Get NLO statistics
    nlo_stats = await mongo_db.nlo_collection.aggregate([
        {"$group": {
            "_id": "$subject_tag",
            "count": {"$sum": 1},
            "avg_confidence": {"$avg": "$confidence"}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]).to_list(10)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": current_user,
        "recent_tasks": recent_tasks,
        "nlo_stats": nlo_stats,
        "title": "Dashboard - BHIV Core"
    })

@app.post("/upload")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    agent: str = Form("edumentor_agent"),
    task_description: str = Form(""),
    current_user: str = Depends(get_current_user)
):
    """Handle file uploads and process them asynchronously."""
    task_id = str(uuid.uuid4())
    
    try:
        # Store task info for real-time updates
        active_tasks[task_id] = {
            "status": "processing",
            "files": [f.filename for f in files],
            "agent": agent,
            "user": current_user,
            "start_time": datetime.now(),
            "description": task_description
        }
        
        # Save uploaded files temporarily using thread pool
        temp_files = await asyncio.get_event_loop().run_in_executor(
            executor, 
            _save_uploaded_files, 
            files, 
            task_id
        )
        
        # Process files based on type asynchronously
        results = await _process_files_async(temp_files, agent, task_description)
        
        # Update task status
        active_tasks[task_id].update({
            "status": "completed",
            "results": results,
            "end_time": datetime.now(),
            "processing_time": (datetime.now() - active_tasks[task_id]["start_time"]).total_seconds()
        })
        
        # Clean up temp files using thread pool
        await asyncio.get_event_loop().run_in_executor(
            executor,
            _cleanup_temp_files,
            temp_files
        )
        
        return JSONResponse({
            "task_id": task_id,
            "status": "completed",
            "results": results,
            "processing_time": active_tasks[task_id]["processing_time"]
        })
        
    except Exception as e:
        logger.error(f"Error in upload processing: {str(e)}")
        active_tasks[task_id] = {
            "status": "error",
            "error": str(e),
            "end_time": datetime.now()
        }
        return JSONResponse({
            "task_id": task_id,
            "status": "error",
            "error": str(e)
        }, status_code=500)

def _save_uploaded_files(files, task_id):
    """Save uploaded files in a separate thread."""
    temp_files = []
    for file in files:
        temp_path = f"temp/{task_id}_{file.filename}"
        os.makedirs("temp", exist_ok=True)
        
        # Read file content synchronously in thread
        content = file.file.read()
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        temp_files.append({
            "path": temp_path,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content)
        })
    return temp_files

def _cleanup_temp_files(temp_files):
    """Clean up temporary files in a separate thread."""
    for temp_file in temp_files:
        try:
            os.remove(temp_file["path"])
        except:
            pass

async def _process_files_async(temp_files, agent, task_description):
    """Process files asynchronously with connection pooling."""
    global http_session
    
    results = []
    tasks = []
    
    for temp_file in temp_files:
        # Determine input type from content type
        content_type = temp_file["content_type"] or ""
        if "pdf" in content_type:
            input_type = "pdf"
        elif "image" in content_type:
            input_type = "image"
        elif "audio" in content_type:
            input_type = "audio"
        else:
            input_type = "text"
        
        # Create async task for processing
        task = _process_single_file(http_session, temp_file, agent, task_description, input_type)
        tasks.append(task)
    
    # Process all files concurrently (with limits)
    for i in range(0, len(tasks), 5):  # Process max 5 files at a time
        batch = tasks[i:i+5]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for result in batch_results:
            if not isinstance(result, Exception):
                results.append(result)
    
    return results

async def _process_single_file(session, temp_file, agent, task_description, input_type):
    """Process a single file using connection pooling."""
    try:
        # Use connection pooling for HTTP requests
        async with session.post(
            "http://localhost:8002/handle_task",
            json={
                "agent": agent,
                "input": task_description,
                "pdf_path": temp_file["path"],
                "input_type": input_type
            }
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return {
                "filename": temp_file["filename"],
                "result": result,
                "input_type": input_type
            }
    except Exception as e:
        logger.error(f"Error processing {temp_file['filename']}: {str(e)}")
        return {
            "filename": temp_file["filename"],
            "error": str(e),
            "input_type": input_type
        }

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str, current_user: str = Depends(get_current_user)):
    """Get real-time task status."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id].copy()
    
    # Convert datetime objects to strings for JSON serialization
    if "start_time" in task_info:
        task_info["start_time"] = task_info["start_time"].isoformat()
    if "end_time" in task_info:
        task_info["end_time"] = task_info["end_time"].isoformat()
    
    return JSONResponse(task_info)

@app.get("/download_nlo/{task_id}")
async def download_nlo(
    task_id: str,
    format: str = "json",
    current_user: str = Depends(get_current_user)
):
    """Download NLO data in specified format."""
    try:
        # Get NLO from MongoDB
        nlo = await get_nlos_by_task_id(task_id)
        
        if not nlo:
            raise HTTPException(status_code=404, detail="NLO not found")
        
        # Remove MongoDB ObjectId for JSON serialization
        if "_id" in nlo:
            del nlo["_id"]
        
        if format.lower() == "json":
            # Create temporary JSON file
            filename = f"nlo_{task_id}.json"
            filepath = f"temp/{filename}"
            os.makedirs("temp", exist_ok=True)
            
            with open(filepath, "w") as f:
                json.dump(nlo, f, indent=2, default=str)
            
            return FileResponse(
                filepath,
                media_type="application/json",
                filename=filename,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        elif format.lower() == "pdf":
            # For PDF generation, you would need a PDF library like reportlab
            # For now, return JSON with PDF content-type
            return JSONResponse(nlo, headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": f"attachment; filename=nlo_{task_id}.pdf"
            })
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        logger.error(f"Error downloading NLO: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nlos")
async def get_nlos(
    subject: Optional[str] = None,
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """API endpoint to get NLOs by subject."""
    try:
        if subject:
            nlos = await get_nlos_by_subject(subject, limit)
        else:
            # Get recent NLOs
            cursor = mongo_db.nlo_collection.find().sort("timestamp", -1).limit(limit)
            nlos = await cursor.to_list(limit)
        
        # Clean up ObjectIds for JSON serialization
        for nlo in nlos:
            if "_id" in nlo:
                nlo["_id"] = str(nlo["_id"])
        
        return JSONResponse(nlos)
        
    except Exception as e:
        logger.error(f"Error getting NLOs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check MongoDB connection
        await mongo_client.admin.command('ping')
        
        # Check MCP bridge
        response = requests.get("http://localhost:8002/health", timeout=5)
        mcp_status = response.status_code == 200
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "mongodb": True,
                "mcp_bridge": mcp_status
            }
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status_code=503)

@app.on_event("startup")
async def startup_event():
    """Initialize HTTP session with connection pooling."""
    global http_session
    # Create aiohttp client session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=100,  # Max number of connections
        limit_per_host=30,  # Max connections per host
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
    )
    http_session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=300)
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Close HTTP session."""
    global http_session
    if http_session:
        await http_session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
