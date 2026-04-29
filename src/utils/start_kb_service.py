#!/usr/bin/env python3
"""
Start KB Service Script

This script starts the custom Knowledge Base service on port 8001
"""

import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def start_kb_service():
    """Start the KB service"""
    logger.info("🚀 Starting Knowledge Base Service...")
    logger.info("📍 KB Service will run on: http://127.0.0.1:8001")
    logger.info("📚 KB Service includes:")
    logger.info("   • Geography knowledge (capitals, countries)")
    logger.info("   • Language and linguistics information")
    logger.info("   • Cultural information (Indian culture, traditions)")
    logger.info("   • Technical NLP knowledge")
    logger.info("   • Multilingual support (Hindi, Sanskrit, Marathi, English)")
    logger.info("")
    
    try:
        # Start the KB service
        subprocess.run([sys.executable, "kb_service.py"], check=True)
    except KeyboardInterrupt:
        logger.info("🛑 KB Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start KB service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_kb_service()
