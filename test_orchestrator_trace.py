#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import traceback
import json
from core.orchestrator import LLMOrchestrator

def test_orchestrator_execution():
    """Test the orchestrator with the exact parameters that caused the error"""
    
    # Manifest from logs
    manifest = {
        'files': {},
        'urls': ['https://news.ycombinator.com/'],
        'keywords': ['json', 'Scrape', 'JSON'],
        'questions': 'Scrape the main headlines from https://news.ycombinator.com/ and return them as a JSON array of strings.',
        'file_types': [],
        'total_files': 0,
        'processing_errors': []
    }
    
    questions = "Scrape the main headlines from https://news.ycombinator.com/ and return them as a JSON array of strings."
    
    print(f"Testing orchestrator with manifest: {manifest}")
    print(f"Questions: {questions}")
    
    try:
        orchestrator = LLMOrchestrator()
        print("✅ Orchestrator created successfully")
        
        # Process the request
        result = orchestrator.process_request(questions, manifest)
        print(f"✅ Request processed: {result.get('success', False)}")
        
        if not result.get('success', False):
            print(f"❌ Error in result: {result.get('error', 'No error message')}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_orchestrator_execution()
