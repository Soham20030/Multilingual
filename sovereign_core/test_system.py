#!/usr/bin/env python3
"""
Test Script for Sovereign LM Bridge + Multilingual KSML Core

This script tests the complete system to ensure all components
are working correctly and can communicate with each other.

Author: Soham Kotkar
"""

import asyncio
import sys
import time
from pathlib import Path

# Add sovereign_core to path
sys.path.append(str(Path(__file__).parent / "sovereign_core"))

async def test_ksml_aligner():
    """Test KSML aligner component"""
    print("Testing KSML Aligner...")
    
    try:
        from ksml.aligner import KSMLAligner
        
        aligner = KSMLAligner()
        await aligner.initialize()
        
        # Test alignment
        result = await aligner.align_text(
            text="What is the meaning of dharma?",
            source_lang="en",
            target_lang="hi"
        )
        
        print(f"SUCCESS: KSML Aligner: {result['intent']} ({result['karma_state']}) - {result['confidence']:.2f} confidence")
        return True
        
    except Exception as e:
        print(f"ERROR: KSML Aligner failed: {e}")
        return False

async def test_mcp_feedback():
    """Test MCP feedback collector"""
    print("Testing MCP Feedback Collector...")
    
    try:
        from mcp.feedback_stream import MCPFeedbackCollector
        
        collector = MCPFeedbackCollector()
        await collector.initialize()
        
        # Test feedback collection
        success = await collector.collect_feedback(
            prompt="What is yoga?",
            original_output="Yoga is a physical exercise",
            corrected_output="Yoga is a spiritual practice",
            reward=0.8,
            user_id="test_user"
        )
        
        if success:
            print("SUCCESS: MCP Feedback Collector: Successfully collected feedback")
            return True
        else:
            print("ERROR: MCP Feedback Collector: Failed to collect feedback")
            return False
        
    except Exception as e:
        print(f"ERROR: MCP Feedback Collector failed: {e}")
        return False

async def test_rl_policy():
    """Test RL policy updater"""
    print("Testing RL Policy Updater...")
    
    try:
        from rl.policy_updater import RLPolicyUpdater
        
        policy = RLPolicyUpdater()
        await policy.initialize()
        
        # Test policy update
        result = await policy.process_feedback(
            prompt="Explain meditation",
            output="Meditation is a mental practice",
            reward=0.9,
            user_id="test_user"
        )
        
        if result["policy_updated"]:
            print("SUCCESS: RL Policy Updater: Successfully updated policy")
            return True
        else:
            print("ERROR: RL Policy Updater: Failed to update policy")
            return False
        
    except Exception as e:
        print(f"ERROR: RL Policy Updater failed: {e}")
        return False

async def test_vaani_composer():
    """Test Vaani speech composer"""
    print("Testing Vaani Speech Composer...")
    
    try:
        from vaani.speech_composer import VaaniSpeechComposer
        
        composer = VaaniSpeechComposer()
        await composer.initialize()
        
        # Test speech composition
        result = await composer.compose_speech(
            text="Yoga is a spiritual practice",
            language="hindi",
            tone="calm"
        )
        
        print(f"SUCCESS: Vaani Composer: {result['tone']} tone, {result['prosody_hint']} prosody")
        return True
        
    except Exception as e:
        print(f"ERROR: Vaani Speech Composer failed: {e}")
        return False

async def test_bridge_reasoner():
    """
    =============================================================================
    TEST: MULTILINGUAL REASONING BRIDGE
    =============================================================================
    This test demonstrates the COMPLETE PIPELINE integration with LM Core
    
    Test Flow:
    1. Initialize the bridge reasoner
    2. Process a sample text through the complete pipeline
    3. Verify all components are working (LM Core, KSML, RL, Vaani, MCP)
    4. Check processing time and performance metrics
    
    This test shows how LM Core responses will be processed through our system
    """
    print("Testing Multilingual Reasoning Bridge...")
    
    try:
        from bridge.reasoner import MultilingualReasoner
        
        # =============================================================================
        # INITIALIZE BRIDGE REASONER
        # =============================================================================
        reasoner = MultilingualReasoner()
        await reasoner.initialize()
        
        # =============================================================================
        # TEST COMPLETE PIPELINE PROCESSING
        # =============================================================================
        # This simulates the complete flow: User Input → LM Core → KSML → RL → Vaani
        result = await reasoner.process_reasoning(
            text="What is the meaning of dharma?",  # Sample input
            user_id="test_user",                    # User context
            include_audio=True                      # Include TTS processing
        )
        
        # =============================================================================
        # VERIFY RESULTS
        # =============================================================================
        print(f"SUCCESS: Bridge Reasoner: Processed in {result['processing_time']:.2f}s")
        print(f"   Components used: {', '.join(result['components_used'])}")
        print(f"   KSML Intent: {result['ksml_metadata']['intent']}")
        print(f"   Karma State: {result['ksml_metadata']['karma_state']}")
        print(f"   Sanskrit Roots: {result['ksml_metadata']['semantic_roots']}")
        return True
        
    except Exception as e:
        print(f"ERROR: Multilingual Reasoning Bridge failed: {e}")
        return False

async def test_complete_system():
    """Test the complete system integration"""
    print("Testing Complete System Integration...")
    
    try:
        # Import the main API
        from api import app
        
        print("SUCCESS: Main API imported successfully")
        
        # Test health endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.get("/health")
        if response.status_code == 200:
            print("SUCCESS: Health endpoint working")
            return True
        else:
            print(f"ERROR: Health endpoint failed: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"ERROR: Complete system test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Starting Sovereign LM Bridge + Multilingual KSML Core Tests")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run individual component tests
    tests = [
        test_ksml_aligner,
        test_mcp_feedback,
        test_rl_policy,
        test_vaani_composer,
        test_bridge_reasoner,
        test_complete_system
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results)
    total = len(results)
    
    print("=" * 70)
    print(f"Test Summary: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("SUCCESS: All tests passed! System is ready for integration.")
        return True
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
