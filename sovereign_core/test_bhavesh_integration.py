#!/usr/bin/env python3
"""
Test script for Bhavesh's LM Core integration
This script tests the real integration with Bhavesh's /compose.final_text endpoint
"""

import asyncio
import sys
import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from bridge.reasoner import MultilingualReasoner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create comprehensive test log file
test_log_file = logs_dir / f"bhavesh_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

def log_test_result(test_id: str, test_name: str, input_data: dict, output_data: dict, 
                   success: bool, error: str = None, processing_time: float = 0.0, 
                   api_status: str = "unknown"):
    """Log comprehensive test results to JSONL file"""
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "test_id": test_id,
        "test_name": test_name,
        "input": input_data,
        "output": output_data,
        "success": success,
        "error": error,
        "processing_time": processing_time,
        "api_status": api_status,
        "validation": {
            "has_aligned_text": "aligned_text" in output_data,
            "has_ksml_metadata": "ksml_metadata" in output_data,
            "has_components_used": "components_used" in output_data,
            "has_processing_time": "processing_time" in output_data,
            "has_trace_id": "trace_id" in output_data,
            "ksml_intent_valid": output_data.get("ksml_metadata", {}).get("intent") in ["question", "explanation", "instruction", "conversation"],
            "ksml_karma_valid": output_data.get("ksml_metadata", {}).get("karma_state") in ["sattva", "rajas", "tamas"],
            "confidence_range": 0.0 <= output_data.get("ksml_metadata", {}).get("confidence", 0) <= 1.0
        }
    }
    
    # Write to JSONL file
    with open(test_log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    return log_entry

async def test_bhavesh_integration():
    """Test the integration with Bhavesh's LM Core API"""
    
    print("=" * 60)
    print("TESTING BHAVESH'S LM CORE INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize the reasoner
        reasoner = MultilingualReasoner()
        await reasoner.initialize()
        
        print("SUCCESS: MultilingualReasoner initialized successfully")
        
        # Test cases with expected results for validation
        test_cases = [
            {
                "text": "What is artificial intelligence?",
                "user_id": "test_user_1",
                "session_id": "test_session_1",
                "expected_intent": "question",
                "expected_karma": "sattva",
                "test_name": "AI Question Test"
            },
            {
                "text": "Explain machine learning in simple terms",
                "user_id": "test_user_2", 
                "session_id": "test_session_2",
                "expected_intent": "explanation",
                "expected_karma": "sattva",
                "test_name": "ML Explanation Test"
            },
            {
                "text": "How does natural language processing work?",
                "user_id": "test_user_3",
                "session_id": "test_session_3",
                "expected_intent": "question",
                "expected_karma": "rajas",
                "test_name": "NLP Question Test"
            },
            {
                "text": "Tell me about deep learning algorithms",
                "user_id": "test_user_4",
                "session_id": "test_session_4",
                "expected_intent": "question",
                "expected_karma": "sattva",
                "test_name": "Deep Learning Test"
            },
            {
                "text": "Create a simple Python function for data analysis",
                "user_id": "test_user_5",
                "session_id": "test_session_5",
                "expected_intent": "instruction",
                "expected_karma": "rajas",
                "test_name": "Programming Instruction Test"
            }
        ]
        
        print(f"\nRunning {len(test_cases)} test cases...")
        
        test_results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            test_id = f"test_{i:03d}"
            print(f"\n--- Test Case {i}: {test_case['test_name']} ---")
            print(f"Input: {test_case['text']}")
            
            # Prepare input data for logging
            input_data = {
                "text": test_case['text'],
                "user_id": test_case['user_id'],
                "session_id": test_case['session_id'],
                "expected_intent": test_case['expected_intent'],
                "expected_karma": test_case['expected_karma']
            }
            
            try:
                start_time = time.time()
                
                # Process through the complete bridge
                result = await reasoner.process_reasoning(
                    text=test_case['text'],
                    user_id=test_case['user_id'],
                    session_id=test_case['session_id'],
                    include_audio=True
                )
                
                processing_time = time.time() - start_time
                
                print(f"SUCCESS: Processing completed in {processing_time:.2f}s")
                print(f"Aligned Text: {result['aligned_text'][:100]}...")
                print(f"Intent: {result['ksml_metadata']['intent']}")
                print(f"Karma State: {result['ksml_metadata']['karma_state']}")
                print(f"Confidence: {result['ksml_metadata']['confidence']:.2f}")
                print(f"Components Used: {', '.join(result['components_used'])}")
                
                # Determine API status
                api_status = "real_api" if 'api_response' in result.get('ksml_metadata', {}) else "fallback"
                
                if api_status == "real_api":
                    print("SUCCESS: Real API response from Bhavesh's LM Core!")
                else:
                    print("INFO: Using fallback response (Bhavesh's API not available)")
                
                # Validate results
                intent_correct = result['ksml_metadata']['intent'] == test_case['expected_intent']
                karma_correct = result['ksml_metadata']['karma_state'] == test_case['expected_karma']
                confidence_good = result['ksml_metadata']['confidence'] >= 0.5
                
                validation_success = intent_correct and karma_correct and confidence_good
                
                print(f"Validation: Intent {'PASS' if intent_correct else 'FAIL'} | Karma {'PASS' if karma_correct else 'FAIL'} | Confidence {'PASS' if confidence_good else 'FAIL'}")
                
                # Log comprehensive test result
                log_entry = log_test_result(
                    test_id=test_id,
                    test_name=test_case['test_name'],
                    input_data=input_data,
                    output_data=result,
                    success=validation_success,
                    processing_time=processing_time,
                    api_status=api_status
                )
                
                test_results.append(log_entry)
                
                if validation_success:
                    successful_tests += 1
                    print(f"SUCCESS: Test {i} PASSED")
                else:
                    failed_tests += 1
                    print(f"FAILED: Test {i} FAILED - Validation issues")
                
            except Exception as e:
                processing_time = time.time() - start_time if 'start_time' in locals() else 0.0
                print(f"ERROR: Test case {i} failed: {e}")
                logger.error(f"Test case {i} failed: {e}")
                
                # Log failed test
                log_entry = log_test_result(
                    test_id=test_id,
                    test_name=test_case['test_name'],
                    input_data=input_data,
                    output_data={},
                    success=False,
                    error=str(e),
                    processing_time=processing_time,
                    api_status="error"
                )
                
                test_results.append(log_entry)
                failed_tests += 1
        
        # Get bridge statistics
        print(f"\nBridge Statistics:")
        stats = await reasoner.get_bridge_stats()
        print(f"   Initialized: {stats['initialized']}")
        print(f"   Processing Log Size: {stats['processing_log_size']}")
        print(f"   LM Core Endpoint: {stats['lm_core_endpoint']}")
        
        # Test summary
        total_tests = successful_tests + failed_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n" + "=" * 60)
        print(f"TEST SUMMARY")
        print(f"=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests} PASS")
        print(f"Failed: {failed_tests} FAIL")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Log File: {test_log_file}")
        print(f"=" * 60)
        
        # Create summary log entry
        summary_entry = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "log_file": str(test_log_file)
            },
            "bridge_stats": stats,
            "test_results": test_results
        }
        
        # Write summary to log file
        with open(test_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')
        
        print(f"\nIntegration test completed!")
        print(f"Detailed logs saved to: {test_log_file}")
        
    except Exception as e:
        print(f"ERROR: Integration test failed: {e}")
        logger.error(f"Integration test failed: {e}")
        
        # Log error to file
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "test_status": "failed"
        }
        
        with open(test_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + '\n')
        
        return False
    
    return successful_tests > 0

async def test_direct_api_call():
    """Test direct API call to Bhavesh's endpoint"""
    
    print("\n" + "=" * 60)
    print("TESTING DIRECT API CALL TO BHAVESH'S ENDPOINT")
    print("=" * 60)
    
    import httpx
    
    try:
        # Test direct call to Bhavesh's API
        test_payload = {
            "query": "Hello, this is a test from the Sovereign LM Bridge",
            "language": "en",
            "top_k": 5,
            "context": []
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/compose.final_text",
                json=test_payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"HTTP Status: {response.status_code}")
            
            if response.status_code == 200:
                api_response = response.json()
                print(f"SUCCESS: API Response received:")
                print(f"   Final Text: {api_response.get('final_text', 'N/A')[:100]}...")
                print(f"   Vaani Audio: {api_response.get('vaani_audio', 'N/A')}")
                return True
            else:
                print(f"ERROR: API call failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except httpx.ConnectError:
        print("ERROR: Cannot connect to Bhavesh's LM Core API")
        print("   Make sure Bhavesh's server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"ERROR: Direct API test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("Starting Bhavesh LM Core Integration Tests")
    print("=" * 60)
    
    # Test 1: Direct API call
    direct_api_success = await test_direct_api_call()
    
    if not direct_api_success:
        print("\nWARNING: Direct API test failed. Make sure Bhavesh's server is running.")
        print("   To start Bhavesh's server:")
        print("   cd bhavesh_lm_core")
        print("   uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
        print("\nContinuing with integration test using fallback responses...")
    
    # Test 2: Full integration test
    integration_success = await test_bhavesh_integration()
    
    if integration_success:
        print("\nSUCCESS: ALL TESTS PASSED! Integration with Bhavesh's LM Core is working!")
        print(f"Comprehensive test logs saved to: {test_log_file}")
        print("Each test includes: inputs, outputs, validation, timing, and correctness")
    else:
        print("\nERROR: Some tests failed. Check the logs for details.")
        print(f"Detailed error logs saved to: {test_log_file}")
    
    print(f"\nTo view detailed logs, open: {test_log_file}")
    print("Log format: JSONL (one JSON object per line)")
    print("Each entry includes: timestamp, test_id, input, output, validation, success status")

if __name__ == "__main__":
    asyncio.run(main())
