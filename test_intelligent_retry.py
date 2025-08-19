#!/usr/bin/env python3
"""
Test script to verify the intelligent retry system works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sandbox_executor import SandboxExecutor

def test_basic_execution():
    """Test that basic code execution still works"""
    print("🧪 Testing basic code execution...")
    
    executor = SandboxExecutor()
    
    # Simple working code
    working_code = """
import json
result = {"status": "working", "value": 42}
print(json.dumps(result))
"""
    
    result = executor.execute_code(working_code, {})
    
    print(f"✅ Success: {result.get('success', False)}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    print(f"🤖 Fixed by LLM: {result.get('was_fixed_by_llm', False)}")
    
    if result.get('success'):
        print("✅ Basic execution test PASSED")
        return True
    else:
        print("❌ Basic execution test FAILED")
        print(f"Error: {result.get('error')}")
        return False

def test_pandas_append_fix():
    """Test that pandas append is automatically fixed"""
    print("\n🧪 Testing pandas append auto-fix...")
    
    executor = SandboxExecutor()
    
    # Code with deprecated pandas append (should be auto-fixed)
    problematic_code = """
import pandas as pd
import json

df1 = pd.DataFrame({'A': [1, 2, 3]})
df2 = pd.DataFrame({'A': [4, 5, 6]})

# This should be automatically fixed from df.append() to pd.concat()
result_df = df1.append(df2)

result = {
    "shape": result_df.shape,
    "values": result_df['A'].tolist()
}
print(json.dumps(result))
"""
    
    result = executor.execute_code(problematic_code, {})
    
    print(f"✅ Success: {result.get('success', False)}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    print(f"🤖 Fixed by LLM: {result.get('was_fixed_by_llm', False)}")
    
    if result.get('success'):
        print("✅ Pandas append auto-fix test PASSED")
        return True
    else:
        print("❌ Pandas append auto-fix test FAILED")
        print(f"Error: {result.get('error')}")
        return False

def test_missing_import_fix():
    """Test that missing imports are automatically added"""
    print("\n🧪 Testing missing import auto-fix...")
    
    executor = SandboxExecutor()
    
    # Code with missing import (should be auto-fixed)
    code_missing_import = """
# Missing: import base64
import json

data = "Hello, World!"
encoded = base64.b64encode(data.encode()).decode()

result = {
    "original": data,
    "encoded": encoded
}
print(json.dumps(result))
"""
    
    result = executor.execute_code(code_missing_import, {})
    
    print(f"✅ Success: {result.get('success', False)}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    print(f"🤖 Fixed by LLM: {result.get('was_fixed_by_llm', False)}")
    
    if result.get('success'):
        print("✅ Missing import auto-fix test PASSED")
        return True
    else:
        print("❌ Missing import auto-fix test FAILED")
        print(f"Error: {result.get('error')}")
        return False

def main():
    print("=" * 50)
    print("🚀 INTELLIGENT RETRY SYSTEM TESTS")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_basic_execution():
        tests_passed += 1
    
    if test_pandas_append_fix():
        tests_passed += 1
    
    if test_missing_import_fix():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED! Intelligent retry system is working correctly.")
        return True
    else:
        print("❌ Some tests FAILED. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
