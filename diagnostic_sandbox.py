#!/usr/bin/env python3
"""
Comprehensive diagnostic script to debug sandbox execution issues.
"""

import tempfile
import os
import sys
import json
import logging
from pathlib import Path
from core.sandbox_executor import SandboxExecutor

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_1_package_installation():
    """Test if packages are available in current environment."""
    print("=" * 60)
    print("TEST 1: Package Installation Check")
    print("=" * 60)
    
    packages = ['requests', 'beautifulsoup4', 'pandas', 'lxml', 'html5lib']
    
    for package in packages:
        try:
            __import__(package if package != 'beautifulsoup4' else 'bs4')
            print(f"✅ {package}: AVAILABLE")
        except ImportError as e:
            print(f"❌ {package}: MISSING - {e}")
    
    # Test basic functionality
    try:
        import requests
        import pandas as pd
        from bs4 import BeautifulSoup
        
        # Test basic operations
        df = pd.DataFrame({'test': [1, 2, 3]})
        soup = BeautifulSoup('<html><body>test</body></html>', 'html.parser')
        
        print(f"✅ Basic operations: pandas DataFrame shape {df.shape}, BeautifulSoup title: {soup.body.text}")
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")

def test_2_internet_access():
    """Test internet access from current environment."""
    print("\\n" + "=" * 60)
    print("TEST 2: Internet Access Check")
    print("=" * 60)
    
    test_urls = [
        'https://httpbin.org/get',
        'https://news.ycombinator.com/',
        'https://www.google.com'
    ]
    
    for url in test_urls:
        try:
            import requests
            response = requests.get(url, timeout=10)
            print(f"✅ {url}: Status {response.status_code}, Content-Length: {len(response.text)}")
        except Exception as e:
            print(f"❌ {url}: FAILED - {e}")

def test_3_sandbox_environment_setup():
    """Test if sandbox environment is set up correctly."""
    print("\\n" + "=" * 60)
    print("TEST 3: Sandbox Environment Setup")
    print("=" * 60)
    
    executor = SandboxExecutor()
    
    # Test with simple code
    simple_code = '''
print("Environment test successful!")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
'''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Temp directory: {temp_dir}")
        
        try:
            # Setup sandbox
            executor._setup_sandbox_environment(temp_dir, {}, simple_code)
            
            # Check files created
            files = list(Path(temp_dir).iterdir())
            print(f"📋 Files created: {[f.name for f in files]}")
            
            # Check main.py content
            main_py_path = Path(temp_dir) / "main.py"
            if main_py_path.exists():
                content = main_py_path.read_text()
                print(f"📄 main.py created ({len(content)} chars)")
                print("📄 main.py preview:")
                print("-" * 40)
                print(content[:500] + "..." if len(content) > 500 else content)
                print("-" * 40)
            else:
                print("❌ main.py was not created!")
                
        except Exception as e:
            print(f"❌ Sandbox setup failed: {e}")
            import traceback
            traceback.print_exc()

def test_4_subprocess_execution():
    """Test subprocess execution within sandbox."""
    print("\\n" + "=" * 60)
    print("TEST 4: Subprocess Execution")
    print("=" * 60)
    
    executor = SandboxExecutor()
    
    # Test with a simple working code
    test_code = '''
print("Subprocess execution test!")
print("Python is working correctly")

# Test basic imports
import json
import os
result = {"status": "success", "message": "Subprocess execution works!"}
print(json.dumps(result))
'''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Setup and run
            executor._setup_sandbox_environment(temp_dir, {}, test_code)
            result = executor._run_code_in_sandbox(temp_dir, 30)
            
            print(f"🔄 Execution result:")
            print(f"  ✅ Success: {result.get('success')}")
            print(f"  📤 Return code: {result.get('return_code')}")
            print(f"  📝 Error: {repr(result.get('error'))}")
            print(f"  📤 Stdout: {repr(result.get('stdout'))}")
            print(f"  📤 Stderr: {repr(result.get('stderr'))}")
            
            # Check output files
            output_files = ['stdout.txt', 'stderr.txt', 'error.json', 'success.txt']
            for filename in output_files:
                filepath = Path(temp_dir) / filename
                if filepath.exists():
                    content = filepath.read_text().strip()
                    print(f"  📄 {filename}: {repr(content[:100])}")
                else:
                    print(f"  ❌ {filename}: NOT FOUND")
                    
        except Exception as e:
            print(f"❌ Subprocess execution failed: {e}")
            import traceback
            traceback.print_exc()

def test_5_network_in_sandbox():
    """Test network access from within sandbox."""
    print("\\n" + "=" * 60)
    print("TEST 5: Network Access in Sandbox")
    print("=" * 60)
    
    executor = SandboxExecutor()
    
    # Test network code
    network_code = '''
import requests
import json

try:
    print("Testing network access from sandbox...")
    
    # Test simple GET request
    response = requests.get('https://httpbin.org/get', timeout=10)
    print(f"HTTP GET Status: {response.status_code}")
    
    # Test the actual target site
    hn_response = requests.get('https://news.ycombinator.com/', timeout=10)
    print(f"Hacker News Status: {hn_response.status_code}")
    print(f"Content length: {len(hn_response.text)}")
    
    # Test BeautifulSoup parsing
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(hn_response.content, 'html.parser')
    title = soup.find('title')
    print(f"Page title: {title.text if title else 'No title found'}")
    
    # Look for headlines
    headlines = soup.find_all('a', class_='storylink')
    print(f"Headlines found: {len(headlines)}")
    
    if headlines:
        print("First few headlines:")
        for i, headline in enumerate(headlines[:3]):
            print(f"  {i+1}. {headline.text}")
    
    print("Network test completed successfully!")
    
except Exception as e:
    print(f"Network test failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    result = executor.execute_code(network_code, {}, timeout=60)
    
    print(f"🌐 Network test result:")
    print(f"  ✅ Success: {result.get('success')}")
    print(f"  🔄 Retry count: {result.get('retry_count', 0)}")
    print(f"  📝 Error: {repr(result.get('error'))}")
    
    if result.get('stdout'):
        print(f"  📤 Output:")
        print("    " + "\\n    ".join(result['stdout'].split('\\n')))

def test_6_original_failing_code():
    """Test the original failing web scraping code with diagnostics."""
    print("\\n" + "=" * 60)
    print("TEST 6: Original Failing Code with Diagnostics")
    print("=" * 60)
    
    executor = SandboxExecutor()
    
    # Enhanced version of the failing code with diagnostics
    diagnostic_code = '''
import json
import sys
import os

print("=== DIAGNOSTIC WEB SCRAPING TEST ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Environment variables: HOME={os.environ.get('HOME')}, PYTHONPATH={os.environ.get('PYTHONPATH')}")

try:
    print("\\n1. Testing imports...")
    import pandas as pd
    print("✅ pandas imported successfully")
    
    import requests
    print("✅ requests imported successfully")
    
    from bs4 import BeautifulSoup
    print("✅ BeautifulSoup imported successfully")
    
    print("\\n2. Testing network access...")
    url = "https://news.ycombinator.com/"
    print(f"Requesting: {url}")
    
    response = requests.get(url, timeout=15)
    print(f"✅ Response received: Status {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Content length: {len(response.content)} bytes")
    
    print("\\n3. Testing HTML parsing...")
    soup = BeautifulSoup(response.content, 'html.parser')
    print(f"✅ BeautifulSoup parsing successful")
    
    title = soup.find('title')
    print(f"Page title: {title.text if title else 'No title found'}")
    
    print("\\n4. Looking for headlines...")
    # Try multiple selectors
    selectors_to_try = [
        ('a.storylink', 'storylink class'),
        ('a.titlelink', 'titlelink class'), 
        ('.titleline > a', 'titleline child'),
        ('tr.athing .titleline a', 'athing titleline'),
        ('a[href*="item?id="]', 'item links')
    ]
    
    headlines_found = []
    
    for selector, description in selectors_to_try:
        elements = soup.select(selector)
        print(f"  {description}: found {len(elements)} elements")
        
        if elements:
            headlines_found = [elem.get_text().strip() for elem in elements[:10]]
            print(f"  ✅ Using {description} - found {len(headlines_found)} headlines")
            break
    
    if not headlines_found:
        print("  ❌ No headlines found with any selector")
        # Fallback: get all links
        all_links = soup.find_all('a')
        print(f"  Total links found: {len(all_links)}")
        if all_links:
            print("  Sample links:")
            for i, link in enumerate(all_links[:5]):
                print(f"    {i+1}. {link.get_text().strip()[:50]}")
    
    print("\\n5. Generating final result...")
    if headlines_found:
        final_result = json.dumps(headlines_found, indent=2)
        print("✅ Final result generated successfully")
        print(f"Result preview: {final_result[:200]}...")
        print("\\n=== FINAL RESULT ===")
        print(final_result)
    else:
        error_result = {"error": "No headlines found", "debug_info": f"Total links: {len(soup.find_all('a'))}"}
        print("❌ No headlines found")
        print(json.dumps(error_result, indent=2))

except Exception as e:
    print(f"\\n❌ ERROR OCCURRED: {e}")
    import traceback
    print("\\n=== FULL TRACEBACK ===")
    traceback.print_exc()
    
    error_result = {
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    }
    print("\\n=== ERROR DETAILS ===")
    print(json.dumps(error_result, indent=2))

print("\\n=== DIAGNOSTIC TEST COMPLETE ===")
'''
    
    result = executor.execute_code(diagnostic_code, {}, timeout=90)
    
    print(f"🔍 Diagnostic result:")
    print(f"  ✅ Success: {result.get('success')}")
    print(f"  🔄 Retry count: {result.get('retry_count', 0)}")
    print(f"  📝 Error: {repr(result.get('error'))}")
    
    if result.get('stdout'):
        print(f"  📤 Full Output:")
        print("-" * 50)
        print(result['stdout'])
        print("-" * 50)

def main():
    """Run all diagnostic tests."""
    print("🔍 AI DATA AGENT SANDBOX DIAGNOSTIC SUITE")
    print("🔍 Running comprehensive diagnostics...")
    
    try:
        test_1_package_installation()
        test_2_internet_access()
        test_3_sandbox_environment_setup()
        test_4_subprocess_execution()
        test_5_network_in_sandbox()
        test_6_original_failing_code()
        
        print("\\n" + "=" * 60)
        print("🎯 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print("All diagnostic tests completed!")
        print("Check the output above for specific issues and solutions.")
        
    except Exception as e:
        print(f"\\n❌ Diagnostic suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
