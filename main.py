#!/usr/bin/env python3
"""
Main Entry Point for Multilingual Tokenization Model Integration

This script serves as the main entry point for the application.
It imports and runs the FastAPI application from the src/api module.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the FastAPI application
from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    from config import settings
    
    # Get configuration
    cfg = settings.get_api_config()
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=cfg["host"],
        port=cfg["port"],
        reload=cfg["debug"],
        log_level=settings.LOG_LEVEL.lower()
    )
