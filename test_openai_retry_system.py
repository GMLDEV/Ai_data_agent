#!/usr/bin/env python3
"""
Test the enhanced OpenAI-powered retry system with visualization capabilities.
"""

import os
import logging
from core.sandbox_executor import SandboxExecutor
from core.code_generator import CodeGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_openai_retry_system():
    """Test the OpenAI-powered intelligent retry system."""
    
    print("=" * 60)
    print("TESTING OPENAI-POWERED RETRY SYSTEM")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key to test the enhanced retry system")
        print("   Falling back to local LLM testing...")
        return test_local_retry_system()
    
    print("✅ OpenAI API key found - testing enhanced system")
    
    # Test code with deliberate errors that OpenAI should fix
    test_code_with_errors = '''
# Intentionally flawed data analysis code that needs fixing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (this will have an error)
df = pd.read_csv("nonexistent_file.csv")  # File doesn't exist - should be fixed

# Analysis with type errors
numeric_columns = df.select_dtypes(include=[number])  # 'number' should be 'number'
correlations = df.corr()

# Visualization with errors
plt.figure(figsize=(10, 6))
plt.plot(df.date, df.value)  # Missing column references
plt.title("Data Analysis")
plt.show()

# Save results
results = {
    "analysis": correlations.to_dict(),
    "summary": "Analysis complete"
}

print("Final result:", results)
'''
    
    # Create file manifest for testing
    test_manifest = {
        "files": {
            "test_data.csv": {
                "type": "csv",
                "path": "test_data.csv",
                "preview": "date,value,category\n2024-01-01,100,A\n2024-01-02,150,B\n2024-01-03,120,A",
                "size": 1000
            }
        },
        "total_files": 1,
        "processing_errors": []
    }
    
    # Test the enhanced retry system
    executor = SandboxExecutor()
    
    print("\n🔧 Testing intelligent retry with error fixing...")
    result = executor.execute_code(
        code=test_code_with_errors,
        files=test_manifest,
        timeout=180,
        allowed_libraries=['pandas', 'matplotlib', 'seaborn', 'numpy']
    )
    
    print("\n" + "=" * 50)
    print("OPENAI RETRY SYSTEM RESULTS")
    print("=" * 50)
    print(f"✅ Success: {result.get('success')}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    print(f"🤖 Fixed by LLM: {result.get('was_fixed_by_llm', False)}")
    print(f"📊 Return code: {result.get('return_code', 'N/A')}")
    
    if result.get('error_history'):
        print(f"\n📝 Error History ({len(result['error_history'])} attempts):")
        for i, attempt in enumerate(result['error_history']):
            print(f"   Attempt {i+1}: {attempt.get('error', 'Unknown')[:100]}...")
    
    stdout_content = result.get('stdout', '')
    if stdout_content:
        print(f"\n📤 Output length: {len(stdout_content)} characters")
        print("📤 Output preview:")
        print("-" * 40)
        print(stdout_content[:800] + "..." if len(stdout_content) > 800 else stdout_content)
        print("-" * 40)
    
    return result

def test_visualization_generation():
    """Test the enhanced visualization code generation."""
    
    print("\n" + "=" * 60)
    print("TESTING VISUALIZATION GENERATION")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for visualization testing")
        return None
    
    # Test visualization generation
    code_gen = CodeGenerator()
    
    task = "Create comprehensive visualizations for sales data analysis"
    manifest = {
        "files": {
            "sales_data.csv": {
                "type": "csv",
                "path": "sales_data.csv", 
                "preview": "date,sales,region,product\n2024-01-01,1000,North,A\n2024-01-02,1500,South,B",
                "size": 5000
            }
        }
    }
    
    print("🎨 Generating visualization code...")
    viz_code = code_gen.generate_visualization_code(task, manifest)
    
    print("\n📊 GENERATED VISUALIZATION CODE:")
    print("-" * 50)
    print(viz_code[:1000] + "..." if len(viz_code) > 1000 else viz_code)
    print("-" * 50)
    
    # Test the generated code
    executor = SandboxExecutor()
    
    print("\n🧪 Testing generated visualization code...")
    result = executor.execute_code(
        code=viz_code,
        files=manifest,
        timeout=180,
        allowed_libraries=['pandas', 'matplotlib', 'seaborn', 'plotly', 'numpy']
    )
    
    print(f"\n✅ Visualization test success: {result.get('success')}")
    return result

def test_local_retry_system():
    """Fallback test using local LLM when OpenAI is not available."""
    
    print("\n🔧 Testing local LLM retry system...")
    
    # Simple test code with a common error
    test_code = '''
import json

# Simple test with a syntax error
data = {"test": "value"  # Missing closing bracket

print("Result:", data)
final_result = data
'''
    
    executor = SandboxExecutor()
    
    result = executor.execute_code(
        code=test_code,
        files={},
        timeout=60
    )
    
    print(f"Local retry result: {result.get('success')}")
    return result

def test_code_enhancement():
    """Test code enhancement capabilities."""
    
    print("\n" + "=" * 60)
    print("TESTING CODE ENHANCEMENT")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for code enhancement testing")
        return None
    
    # Basic code that needs enhancement
    basic_code = '''
import requests
import json

url = "https://api.github.com/users/octocat"
response = requests.get(url)
data = response.json()
print(data)
'''
    
    code_gen = CodeGenerator()
    
    print("🔧 Enhancing code with robustness improvements...")
    enhanced_code = code_gen.enhance_code_with_retries(basic_code, "robustness")
    
    print("\n📋 ENHANCED CODE:")
    print("-" * 50)
    print(enhanced_code[:800] + "..." if len(enhanced_code) > 800 else enhanced_code)
    print("-" * 50)
    
    return enhanced_code

if __name__ == "__main__":
    print("🚀 Starting Enhanced AI Data Agent Testing")
    print("=" * 60)
    
    # Test 1: OpenAI retry system
    retry_result = test_openai_retry_system()
    
    # Test 2: Visualization generation
    viz_result = test_visualization_generation()
    
    # Test 3: Code enhancement
    enhanced_code = test_code_enhancement()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print(f"✅ Retry System: {'PASS' if retry_result and retry_result.get('success') else 'NEEDS_WORK'}")
    print(f"🎨 Visualizations: {'PASS' if viz_result and viz_result.get('success') else 'NEEDS_OPENAI_KEY'}")
    print(f"🔧 Code Enhancement: {'PASS' if enhanced_code else 'NEEDS_OPENAI_KEY'}")
    
    print("\n💡 To fully test the enhanced features, set your OPENAI_API_KEY environment variable")
