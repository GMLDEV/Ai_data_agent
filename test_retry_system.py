#!/usr/bin/env python3
"""
Test script to demonstrate the intelligent retry system with LLM error fixing.
"""

import logging
import asyncio
from core.sandbox_executor import SandboxExecutor
from core.llm_client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dependency_error():
    """Test automatic dependency installation and error fixing."""
    print("=== Testing Dependency Error Handling ===")
    
    # Code that will fail with missing dependency
    failing_code = """
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import seaborn as sns
import sklearn.linear_model

# This should work after auto-installation
data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print("Data created successfully:", data.shape)

# Use multiple libraries to test comprehensive installation
response = requests.get('https://httpbin.org/json')
print("HTTP request successful:", response.status_code)

fig = go.Figure(data=go.Scatter(x=data['x'], y=data['y']))
print("Plotly figure created successfully")

print("All dependencies working correctly!")
"""
    
    executor = SandboxExecutor()
    result = executor.execute_code(
        code=failing_code,
        files={},
        timeout=180
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Retry count: {result.get('retry_count', 0)}")
    print(f"Output: {result.get('stdout', '')[:500]}...")
    if result.get('retry_count', 0) > 0:
        print("✅ Dependencies were automatically installed and code executed successfully!")
    return result

async def test_syntax_error_fixing():
    """Test LLM-powered syntax error fixing."""
    print("\n=== Testing Syntax Error Fixing ===")
    
    # Code with deliberate syntax errors
    broken_code = """
import pandas as pd

# Intentional syntax errors for LLM to fix
def analyze_data(data)
    print("Starting analysis"
    
    # Missing colon and parentheses
    if len(data) > 0
        result = data.describe(
        print(f"Analysis complete: {result}")
    else:
        print("No data to analyze")
    
    return result

# Test the function
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
analyze_data(df)
"""
    
    executor = SandboxExecutor()
    result = executor.execute_code(
        code=broken_code,
        files={},
        timeout=180
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Retry count: {result.get('retry_count', 0)}")
    if result.get('success'):
        print("✅ Syntax errors were automatically fixed by LLM!")
        print(f"Output: {result.get('stdout', '')[:500]}...")
    else:
        print("❌ Could not fix syntax errors")
        print(f"Error: {result.get('error', '')}")
    
    return result

async def test_logic_error_fixing():
    """Test LLM-powered logic error fixing."""
    print("\n=== Testing Logic Error Fixing ===")
    
    # Code with logic errors
    buggy_code = """
import pandas as pd

def calculate_statistics(data):
    # This will cause a division by zero error
    mean_val = sum(data) / 0  # Intentional bug
    
    # This will cause a type error
    result = mean_val + "statistics"  # Intentional bug
    
    return result

# Test data
test_data = [1, 2, 3, 4, 5]
result = calculate_statistics(test_data)
print(f"Statistics: {result}")
"""
    
    executor = SandboxExecutor()
    result = executor.execute_code(
        code=buggy_code,
        files={},
        timeout=180
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Retry count: {result.get('retry_count', 0)}")
    if result.get('success'):
        print("✅ Logic errors were automatically fixed by LLM!")
        print(f"Output: {result.get('stdout', '')[:500]}...")
    else:
        print("❌ Could not fix logic errors")
        print(f"Final error: {result.get('error', '')}")
    
    return result

async def test_complex_web_scraping():
    """Test complex web scraping with potential errors."""
    print("\n=== Testing Complex Web Scraping with Error Recovery ===")
    
    # Complex web scraping code that might have various issues
    scraping_code = """
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json

def scrape_data():
    # This URL might not work or structure might change
    url = "https://httpbin.org/html"
    
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract some data
        title = soup.find('title')
        if title:
            print(f"Page title: {title.text}")
        
        # Try to find links
        links = soup.find_all('a')
        print(f"Found {len(links)} links")
        
        # Create a simple dataset
        data = {
            'title': title.text if title else 'No title',
            'link_count': len(links),
            'status_code': response.status_code
        }
        
        df = pd.DataFrame([data])
        print("Data extracted successfully:")
        print(df.to_string())
        
        return df
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        return None

result = scrape_data()
if result is not None:
    print("\\nWeb scraping completed successfully!")
else:
    print("\\nWeb scraping failed!")
"""
    
    executor = SandboxExecutor()
    result = executor.execute_code(
        code=scraping_code,
        files={},
        timeout=180
    )
    
    print(f"Success: {result.get('success')}")
    print(f"Retry count: {result.get('retry_count', 0)}")
    if result.get('success'):
        print("✅ Web scraping code executed successfully!")
        print(f"Output: {result.get('stdout', '')[:800]}...")
    else:
        print("❌ Web scraping failed")
        print(f"Error: {result.get('error', '')}")
    
    return result

async def main():
    """Run all retry system tests."""
    print("🚀 Testing Intelligent Retry System with LLM Error Fixing")
    print("=" * 70)
    
    results = []
    
    try:
        # Test 1: Dependency errors
        result1 = await test_dependency_error()
        results.append(("Dependency Error", result1.get('success', False)))
        
        # Test 2: Syntax errors
        result2 = await test_syntax_error_fixing()
        results.append(("Syntax Error", result2.get('success', False)))
        
        # Test 3: Logic errors
        result3 = await test_logic_error_fixing()
        results.append(("Logic Error", result3.get('success', False)))
        
        # Test 4: Complex web scraping
        result4 = await test_complex_web_scraping()
        results.append(("Web Scraping", result4.get('success', False)))
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        results.append(("Test Execution", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("🔍 TEST SUMMARY")
    print("=" * 70)
    
    success_count = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if success:
            success_count += 1
    
    print(f"\nOverall: {success_count}/{len(results)} tests passed")
    
    if success_count == len(results):
        print("🎉 All tests passed! The intelligent retry system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
