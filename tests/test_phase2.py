import asyncio
import requests
import json
import pytest

def test_llm_classification():
    """Test the LLM classifier"""
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
            pytest.skip("OpenAI API key not found. Please add it to .env file")
            
        classifier = WorkflowClassifier()
        
        for i, test_case in enumerate(test_cases):
            manifest = {
                "questions": test_case["questions"],
                "file_types": test_case["file_types"],
                "urls": [],
                "keywords": [],
                "files": {}
            }
            
            result = classifier.classify(manifest)
            assert result['workflow'] == test_case['expected'], f"Expected {test_case['expected']} but got {result['workflow']}"
            assert 'confidence' in result, "Confidence score missing in result"
            assert 'reasoning' in result, "Reasoning missing in result"
            
    except Exception as e:
        pytest.fail(f"LLM Classification test failed: {e}")

def test_api_endpoint():
    """Test the new API endpoint"""
    print("\n=== Testing API Endpoint ===")