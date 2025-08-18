#!/usr/bin/env python3
"""
Test the enhanced sandbox executor with comprehensive logging.
"""

import logging
from core.sandbox_executor import SandboxExecutor

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sandbox_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def test_with_detailed_logging():
    """Test the web scraping code with detailed logging."""
    
    logger.info("=" * 60)
    logger.info("STARTING ENHANCED SANDBOX DIAGNOSTICS TEST")
    logger.info("=" * 60)
    
    # The failing web scraping code
    test_code = '''
# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import json

print("Starting web scraping script...")

# Define URL to scrape
url = "https://news.ycombinator.com/"
print(f"Target URL: {url}")

try:
    # Use requests library to get the HTML content of the webpage
    print("Sending HTTP request...")
    response = requests.get(url, timeout=10)
    print(f"Response status code: {response.status_code}")
    print(f"Response content length: {len(response.content)} bytes")

    # Use BeautifulSoup to parse the HTML content
    print("Parsing HTML with BeautifulSoup...")
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML parsing completed")

    # Find all the headlines using the 'a' tag and class 'storylink'
    print("Looking for headlines...")
    headlines = soup.find_all('a', class_='storylink')
    print(f"Found {len(headlines)} headlines with class 'storylink'")

    if not headlines:
        # Try alternative selectors
        print("No storylink headlines found, trying alternatives...")
        alternatives = [
            ('a.titlelink', 'titlelink class'),
            ('.titleline > a', 'titleline child links'),
            ('a[href*="item?id="]', 'item links')
        ]
        
        for selector, description in alternatives:
            elements = soup.select(selector)
            print(f"  {description}: {len(elements)} elements")
            if elements:
                headlines = elements
                break

    # Create an empty list to store the headlines
    headlines_list = []

    # Loop through the headlines and extract the text
    for i, headline in enumerate(headlines):
        text = headline.get_text().strip()
        if text:
            headlines_list.append(text)
            if i < 3:  # Log first 3 for debugging
                print(f"  Headline {i+1}: {text}")

    print(f"Extracted {len(headlines_list)} headline texts")

    # Convert the list to a JSON array
    final_result = json.dumps(headlines_list)
    print(f"Final result length: {len(final_result)} characters")

    # Print the final result
    print("=== FINAL RESULT ===")
    print(final_result)
    print("=== END RESULT ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("Script execution completed.")
'''
    
    # Create executor and run with diagnostics
    executor = SandboxExecutor()
    
    logger.info("Creating file manifest for test...")
    file_manifest = {
        "urls": ["https://news.ycombinator.com/"],
        "keywords": ["scraping", "headlines"],
        "questions": "Extract headlines from Hacker News",
        "file_types": [".html"],
        "total_files": 0,
        "processing_errors": []
    }
    
    logger.info("Executing code with enhanced diagnostics...")
    result = executor.execute_code(
        code=test_code,
        files=file_manifest,
        timeout=120,
        allowed_libraries=['requests', 'beautifulsoup4', 'lxml', 'html5lib']
    )
    
    logger.info("=" * 60)
    logger.info("SANDBOX EXECUTION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Success: {result.get('success')}")
    logger.info(f"Error: {repr(result.get('error'))}")
    logger.info(f"Return code: {result.get('return_code', 'N/A')}")
    logger.info(f"Retry count: {result.get('retry_count', 0)}")
    logger.info(f"Was fixed by LLM: {result.get('was_fixed_by_llm', False)}")
    
    stdout_content = result.get('stdout', '')
    stderr_content = result.get('stderr', '')
    
    logger.info(f"Stdout length: {len(stdout_content)} characters")
    logger.info(f"Stderr length: {len(stderr_content)} characters")
    
    if stdout_content:
        logger.info("Stdout preview:")
        logger.info("-" * 40)
        logger.info(stdout_content[:500] + "..." if len(stdout_content) > 500 else stdout_content)
        logger.info("-" * 40)
    
    if stderr_content:
        logger.warning("Stderr content:")
        logger.warning("-" * 40)
        logger.warning(stderr_content[:500] + "..." if len(stderr_content) > 500 else stderr_content)
        logger.warning("-" * 40)
    
    output = result.get('output')
    if output:
        logger.info(f"Parsed output type: {type(output)}")
        if isinstance(output, list):
            logger.info(f"Output list length: {len(output)}")
            if output:
                logger.info(f"First few items: {output[:3]}")
        else:
            logger.info(f"Output preview: {str(output)[:200]}")
    else:
        logger.warning("No output captured")
    
    # Check error history if available
    error_history = result.get('error_history', [])
    if error_history:
        logger.warning(f"Error history ({len(error_history)} attempts):")
        for i, attempt in enumerate(error_history):
            logger.warning(f"  Attempt {attempt.get('attempt', i+1)}: {attempt.get('error', 'Unknown error')}")
    
    logger.info("Test completed. Check sandbox_debug.log for full details.")
    return result

if __name__ == "__main__":
    result = test_with_detailed_logging()
    
    print("\\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Success: {result.get('success')}")
    print(f"Output available: {bool(result.get('output'))}")
    print("Check sandbox_debug.log for detailed diagnostics.")
