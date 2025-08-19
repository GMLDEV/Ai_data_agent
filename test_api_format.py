#!/usr/bin/env python3
"""
Test script to verify API endpoint matches evaluation team format
"""

import requests
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoint():
    """Test the /api/ endpoint with the evaluation team format"""
    
    # Test data
    questions_content = """Analyze the network data and provide:
1. Total number of edges
2. Node with the highest degree
3. Average degree of all nodes
4. Network density
5. Shortest path length between Alice and Eve
6. Network graph visualization
7. Degree distribution histogram"""
    
    csv_content = """source,target
alice,bob
bob,charlie
charlie,dave
dave,eve
eve,alice
alice,charlie
bob,dave"""
    
    # Prepare files as the evaluation team would send them
    files = {
        'questions.txt': ('questions.txt', io.StringIO(questions_content), 'text/plain'),
        'edges.csv': ('edges.csv', io.StringIO(csv_content), 'text/csv')
    }
    
    try:
        # Test the main API endpoint
        url = "http://localhost:8000/api/"
        logger.info(f"Testing {url} with evaluation team format...")
        
        response = requests.post(url, files=files)
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ API endpoint test successful!")
            logger.info(f"Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Check if it has the expected structure
            if isinstance(result, dict) and any(key in result for key in ['edge_count', 'success', 'result']):
                logger.info("✅ Response has expected structure")
            else:
                logger.warning("⚠️ Response structure may not match expectations")
                
        else:
            logger.error(f"❌ API test failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        logger.error("❌ Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_api_endpoint()
