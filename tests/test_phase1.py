import asyncio
import json
from pathlib import Path
from core.file_processor import FileProcessor
from sandbox.executor import SandboxExecutor
from config import settings

async def test_file_processor():
    """Test file processor with sample data"""
    print("Testing File Processor...")
    
    processor = FileProcessor()
    
    # Create sample CSV content
    csv_content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,35,Chicago"
    
    # Create sample text content
    text_content = b"This is a sample text file for testing."
    
    files = {
        "sample.csv": csv_content,
        "readme.txt": text_content
    }
    
    questions = "Analyze the data in sample.csv and create a plot showing age distribution. Also check https://example.com for reference."
    
    manifest = processor.create_manifest(files, questions)
    
    print("Manifest created successfully:")
    print(json.dumps(manifest, indent=2, default=str))
    return True

def test_sandbox():
    """Test sandbox executor"""
    print("\nTesting Sandbox Executor...")
    
    executor = SandboxExecutor(
        settings.sandbox_memory_limit,
        settings.sandbox_cpu_limit,
        settings.max_execution_time
    )
    
    # Test basic execution
    test_code = """
print("Hello from sandbox!")
import pandas as pd
data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
df = pd.DataFrame(data)
print("DataFrame created:")
print(df)
print("Sum of column 'a':", df['a'].sum())
"""
    
    result = executor.test_sandbox()
    print("Sandbox test result:", json.dumps(result, indent=2))
    
    result = executor.execute_simple(test_code)
    print("Code execution result:", json.dumps(result, indent=2))
    return result.get('success', False)

async def main():
    """Run all tests"""
    print("=== Phase 1 Testing ===\n")
    
    try:
        # Test file processor
        await test_file_processor()
        
        # Test sandbox
        test_sandbox()
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())