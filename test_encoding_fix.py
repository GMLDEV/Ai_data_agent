#!/usr/bin/env python3
"""
Quick test of the encoding fixes.
"""

from core.sandbox_executor import SandboxExecutor

def test_encoding_fix():
    executor = SandboxExecutor()
    
    test_code = '''
print("Hello World from sandbox!")
import requests
print("Requests imported successfully")
result = {"status": "success", "message": "Test completed"}
print(result)
'''
    
    result = executor.execute_code(test_code, {}, timeout=60)
    print(f'Success: {result.get("success")}')
    print(f'Error: {result.get("error")}')
    print(f'Stdout: {result.get("stdout", "")[:200]}')

if __name__ == "__main__":
    test_encoding_fix()
