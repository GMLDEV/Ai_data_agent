import asyncio
import requests
import json

def test_llm_classification():
    """Test the LLM classifier"""
    print("=== Testing LLM Classification ===")
    
    # Test data
    test_cases = [
        {
            "questions": "What is the average age in my dataset?",
            "file_types": ["csv"],
            "expected": "data_analysis"
        },
        {
            "questions": "Extract data from https://example.com",
            "file_types": [],
            "expected": "web_scraping"
        },
        {
            "questions": "Process this image and extract text",
            "file_types": ["image"],
            "expected": "image_analysis"
        }
    ]
    
    try:
        from core.classifier import WorkflowClassifier
        from config import settings
        
        if not settings.openai_api_key:
            print("❌ OpenAI API key not found. Please add it to .env file")
            return False
            
        classifier = WorkflowClassifier()
        
        for i, test_case in enumerate(test_cases):
            print(f"\nTest {i+1}: {test_case['questions'][:50]}...")
            
            manifest = {
                "questions": test_case["questions"],
                "file_types": test_case["file_types"],
                "urls": [],
                "keywords": [],
                "files": {}
            }
            
            result = classifier.classify(manifest)
            print(f"Result: {result['workflow']} (confidence: {result.get('confidence', 0):.2f})")
            print(f"Reasoning: {result.get('reasoning', 'No reasoning')}")
            
        print("\n✅ LLM Classification tests completed")
        return True
        
    except Exception as e:
        print(f"❌ LLM Classification test failed: {e}")
        return False

def test_api_endpoint():
    """Test the new API endpoint"""
    print("\n=== Testing API Endpoint ===")