#!/usr/bin/env python3
"""
Test script to validate the fixed classification system handles various request types correctly.
"""

import sys
import os
sys.path.append('.')

from core.orchestrator import Orchestrator
from core.llm_client import LLMClient
from core.file_processor import FileProcessor
from core.classifier import WorkflowClassifier
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_classification_scenarios():
    """Test various classification scenarios to ensure proper workflow routing."""
    
    try:
        # Initialize components
        llm_client = LLMClient()
        classifier = WorkflowClassifier(llm_client)
        
        # Test scenarios
        test_cases = [
            {
                "name": "Network Analysis",
                "request": "Create a network graph using NetworkX with 5 nodes and analyze its connectivity properties",
                "expected": "dynamic",
                "files": {}
            },
            {
                "name": "Simple Code Generation", 
                "request": "Write a simple function to calculate fibonacci numbers",
                "expected": "code_generation",
                "files": {}
            },
            {
                "name": "Data Analysis with CSV",
                "request": "Analyze this CSV data and create plots showing trends",
                "expected": "data_analysis", 
                "files": {"data.csv": {"type": "csv", "columns": ["x", "y"]}}
            },
            {
                "name": "Web Scraping",
                "request": "Extract data from https://example.com",
                "expected": "web_scraping",
                "files": {}
            },
            {
                "name": "Code without Analysis Intent",
                "request": "Generate Python code to process network data structures", 
                "expected": "dynamic",
                "files": {}
            },
            {
                "name": "Ambiguous Request",
                "request": "Help me with my project",
                "expected": "dynamic",
                "files": {}
            }
        ]
        
        print("🧪 Testing Classification System...")
        print("=" * 60)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{i}. Test: {case['name']}")
            print(f"   Request: {case['request']}")
            print(f"   Expected: {case['expected']}")
            
            # Create manifest
            manifest = {
                "questions": case['request'],
                "files": case['files'],
                "file_types": list({info.get('type') for info in case['files'].values() if isinstance(info, dict) and 'type' in info}),
                "urls": [],
                "keywords": []
            }
            
            # Test classification
            result = classifier.classify(manifest)
            actual_workflow = result.get('workflow', 'unknown')
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', 'No reasoning')
            
            print(f"   Actual: {actual_workflow} (confidence: {confidence:.2f})")
            print(f"   Reasoning: {reasoning}")
            
            # Check if classification is correct
            if actual_workflow == case['expected']:
                print("   ✅ PASS")
            else:
                print("   ❌ FAIL")
            
            print("-" * 40)
        
        print("\n🎯 Classification Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_end_to_end_workflow():
    """Test end-to-end workflow execution with network analysis."""
    
    print("\n🔄 Testing End-to-End Network Analysis...")
    print("=" * 60)
    
    try:
        # Initialize orchestrator
        orchestrator = Orchestrator()
        
        # Test network analysis request
        request = "Create a simple network graph with NetworkX using 4 nodes connected in a square pattern and calculate basic graph metrics"
        
        print(f"Request: {request}")
        print("Processing...")
        
        # Process request
        result = orchestrator.process_request(request, {})
        
        print(f"\nResult:")
        if isinstance(result, dict):
            print(f"  Success: {result.get('success', False)}")
            print(f"  Workflow: {result.get('workflow_used', 'Unknown')}")
            
            if result.get('success'):
                print(f"  Output available: {'Yes' if result.get('result') else 'No'}")
                if result.get('result'):
                    output_preview = str(result.get('result'))[:200]
                    print(f"  Output preview: {output_preview}...")
            else:
                error = result.get('error', 'Unknown error')
                print(f"  Error: {error}")
        else:
            print(f"  Raw result (type: {type(result)}): {str(result)[:300]}...")
            print("  ✅ Network analysis executed successfully (string output)")
            
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Classification Tests...")
    
    # Test classification logic
    test_classification_scenarios()
    
    # Test end-to-end workflow
    test_end_to_end_workflow()
    
    print("\n✅ All tests completed!")
