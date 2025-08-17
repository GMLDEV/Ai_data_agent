#!/usr/bin/env python3
"""
Simple demonstration of the intelligent retry system.
"""

from core.sandbox_executor import SandboxExecutor
import logging

# Enable logging to see the retry process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def demonstrate_retry_system():
    """Demonstrate the intelligent retry system with a simple example."""
    
    print("🔄 Intelligent Retry System Demonstration")
    print("=" * 50)
    
    # Create a code snippet that will likely fail initially but can be fixed
    problematic_code = """
# This code has multiple issues that LLM should be able to fix:
# 1. Missing import
# 2. Syntax error
# 3. Logic error

def process_data(data):
    # Missing pandas import
    df = pd.DataFrame(data)
    
    # Syntax error - missing closing parenthesis
    result = df.describe(
    
    # Logic error - will cause key error
    summary = {
        'count': result['nonexistent_column'],
        'mean': result.mean()
    }
    
    return summary

# Test data
test_data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]}
result = process_data(test_data)
print("Processing complete:", result)
"""
    
    print("Original code (with intentional errors):")
    print("-" * 40)
    print(problematic_code)
    print("-" * 40)
    print("\\nExecuting with intelligent retry system...")
    print("The system will:")
    print("1. ✨ Automatically install missing packages (pandas)")
    print("2. 🔧 Use LLM to fix syntax errors")
    print("3. 🧠 Use LLM to fix logic errors")
    print("4. 🔄 Retry execution until successful")
    print()
    
    # Execute with the enhanced sandbox
    executor = SandboxExecutor()
    
    result = executor.execute_code(
        code=problematic_code,
        files={},
        timeout=300  # Give it time to install packages and retry
    )
    
    print("\\n" + "=" * 50)
    print("📊 EXECUTION RESULTS")
    print("=" * 50)
    
    print(f"Success: {result.get('success', False)}")
    print(f"Retry attempts: {result.get('retry_count', 0)}")
    
    if result.get('success'):
        print("✅ SUCCESS! The code was automatically fixed and executed.")
        print(f"\\nFixed code preview:")
        print("-" * 30)
        fixed_code = result.get('fixed_code', 'Not available')
        if fixed_code and fixed_code != 'Not available':
            print(fixed_code[:500] + '...' if len(fixed_code) > 500 else fixed_code)
        print("-" * 30)
        
        print(f"\\nOutput:")
        print(result.get('stdout', 'No output'))
    else:
        print("❌ FAILED: Could not fix the code automatically.")
        print(f"Final error: {result.get('error', 'Unknown error')}")
        
        if 'error_history' in result:
            print(f"\\nError history ({len(result['error_history'])} attempts):")
            for i, error in enumerate(result['error_history'], 1):
                print(f"  Attempt {i}: {error.get('error', 'Unknown error')}")
    
    print("\\n" + "=" * 50)
    print("🎯 KEY FEATURES DEMONSTRATED")
    print("=" * 50)
    print("✨ Automatic package installation")
    print("🔧 LLM-powered code fixing") 
    print("🔄 Intelligent retry logic")
    print("📝 Comprehensive error tracking")
    print("🧠 Context-aware error resolution")
    
    return result

if __name__ == "__main__":
    demonstrate_retry_system()
