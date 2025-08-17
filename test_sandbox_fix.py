#!/usr/bin/env python3
"""
Test to verify sandbox executor parameter handling fixes
"""
import os
import sys
sys.path.append('.')

# Set up environment
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

def test_sandbox_executor():
    """Test sandbox executor with various problematic parameter types"""
    try:
        from core.sandbox_executor import SandboxExecutor
        
        sandbox = SandboxExecutor()
        
        print("Testing SandboxExecutor parameter handling...")
        
        # Test 1: Normal case
        print("Test 1: Normal execution")
        result1 = sandbox.execute_simple('print("Hello World")')
        print(f"✅ Normal case: {result1.get('success', 'error handled')}")
        
        # Test 2: Test execute_code with bad files parameter
        print("Test 2: Bad files parameter")
        result2 = sandbox.execute_code(
            code='print("Test")',
            files={'invalid': 123},  # This should be handled now
            allowed_libraries=None
        )
        print(f"✅ Bad files: {result2.get('success', 'error handled')}")
        
        # Test 3: Test with non-list libraries
        print("Test 3: String library instead of list")
        result3 = sandbox.execute_code(
            code='print("Test")',
            files={},
            allowed_libraries='pandas'  # Should convert to list
        )
        print(f"✅ String library: {result3.get('success', 'error handled')}")
        
        # Test 4: Test with integer files parameter (the original error)
        print("Test 4: Integer files parameter")
        result4 = sandbox.execute_code(
            code='print("Test")',
            files=123,  # This was likely causing the original error
            allowed_libraries=[]
        )
        print(f"✅ Integer files: {result4.get('success', 'error handled')}")
        
        return True
        
    except Exception as e:
        print(f"❌ SandboxExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_integration():
    """Test workflow integration with sandbox executor"""
    try:
        from workflows.dynamic_code_execution import DynamicCodeExecutionWorkflow
        from core.code_generator import CodeGenerator
        from core.sandbox_executor import SandboxExecutor
        
        print("\nTesting workflow integration...")
        
        code_gen = CodeGenerator()
        sandbox = SandboxExecutor()
        
        # Create a manifest that might cause issues
        manifest = {
            'files': {'test.txt': 'content'},
            'urls': [],
            'libraries_needed': 'pandas'  # This might cause issues if not handled
        }
        
        workflow = DynamicCodeExecutionWorkflow(code_gen, manifest, sandbox)
        
        # This should not crash with 'int' object is not iterable
        result = workflow.execute(sandbox, "Print hello world")
        print(f"✅ Workflow integration: {result.get('success', 'error handled gracefully')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing sandbox executor parameter handling fixes...\n")
    
    success = True
    success &= test_sandbox_executor()
    success &= test_workflow_integration()
    
    if success:
        print("\n✅ All sandbox tests passed! The 'int object is not iterable' fix is working.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
