#!/usr/bin/env python3
"""
Test the clean API response format.
"""

import json
from core.orchestrator import LLMOrchestrator

def test_clean_response():
    """Test that the API returns clean responses without internal processing details."""
    
    print("🧪 Testing Clean API Response Format")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = LLMOrchestrator()
    
    # Test case 1: Simple data processing task
    print("\n📋 Test 1: Simple JSON Processing")
    questions = """
    Process this data and return a JSON array with just the names.
    Data: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    """
    
    files = {}  # No files for this test
    
    result = orchestrator.process_request(questions, files)
    
    print("API Response:")
    print(json.dumps(result, indent=2))
    
    # Test case 2: Multi-part question
    print("\n📋 Test 2: Multi-part Question (like your example)")
    questions = """
    Analyze sample data and answer these questions with a JSON array:
    1. How many items are there?
    2. What is the average value?
    3. Which item has the highest value?
    
    Return format: [count, average, highest_item_name]
    """
    
    result2 = orchestrator.process_request(questions, files)
    
    print("API Response:")
    print(json.dumps(result2, indent=2))
    
    # Check response cleanliness
    print("\n✅ Response Analysis:")
    print(f"Contains 'original_code': {'original_code' in str(result)}")
    print(f"Contains 'final_code': {'final_code' in str(result)}")
    print(f"Contains 'retry_count': {'retry_count' in str(result)}")
    print(f"Contains internal details: {any(key in str(result) for key in ['workflow_used', 'confidence', 'reasoning'])}")
    
    if isinstance(result, list):
        print("✅ Perfect! Response is a clean JSON array")
    elif isinstance(result, dict) and len(result.keys()) <= 3:
        print("✅ Good! Response is a clean dict")
    else:
        print("❌ Response contains too much internal data")
    
    print("\n" + "=" * 50)
    print("EXPECTED vs ACTUAL")
    print("=" * 50)
    print("EXPECTED for Wikipedia films question:")
    print('[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]')
    print("\nACTUAL response format:")
    print(f"Type: {type(result)}")
    if isinstance(result, dict):
        print(f"Keys: {list(result.keys())}")

if __name__ == "__main__":
    test_clean_response()
