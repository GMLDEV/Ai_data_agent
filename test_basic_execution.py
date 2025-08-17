#!/usr/bin/env python3
"""
Simple test of the retry system without LLM fixing for debugging.
"""

from core.sandbox_executor import SandboxExecutor
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_simple_execution():
    """Test simple code execution first."""
    print("Testing simple code execution...")
    
    simple_code = '''
print("Hello, World!")
result = 2 + 2
print(f"2 + 2 = {result}")
'''
    
    executor = SandboxExecutor()
    
    # Use the old method first to see if basic execution works
    try:
        # Create a temporary directory and test basic execution
        with tempfile.TemporaryDirectory() as temp_dir:
            executor._setup_sandbox_environment(temp_dir, {}, simple_code)
            result = executor._run_code_in_sandbox(temp_dir, 30)
            
            print(f"Basic execution result:")
            print(f"  Success: {result.get('success')}")
            print(f"  Error: {repr(result.get('error'))}")
            print(f"  Stdout: {repr(result.get('stdout'))}")
            print(f"  Stderr: {repr(result.get('stderr'))}")
            print(f"  Return code: {result.get('return_code')}")
            
    except Exception as e:
        print(f"Exception during basic test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import tempfile
    test_simple_execution()
