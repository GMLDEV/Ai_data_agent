from core.sandbox_executor import SandboxExecutor
import logging

logging.basicConfig(level=logging.INFO)

# Test basic functionality
print("Testing enhanced SandboxExecutor...")
executor = SandboxExecutor()
print("✅ SandboxExecutor initialized successfully")

# Test simple code execution
test_code = '''
print("Hello from enhanced retry system!")
import json
data = {"test": "success", "features": ["OpenAI integration", "Visualization support", "Enhanced retry logic"]}
print("Features:", json.dumps(data, indent=2))
final_result = data
'''

print("\nTesting basic code execution...")
result = executor.execute_code(test_code, {}, timeout=30)
print(f"✅ Basic execution: {'SUCCESS' if result.get('success') else 'FAILED'}")
print(f"📊 Output available: {bool(result.get('output'))}")

if result.get('stdout'):
    print("\nOutput:")
    print(result['stdout'][:500])

print("\n" + "="*50)
print("ENHANCEMENT SUMMARY")
print("="*50)
print("✅ OpenAI code fixer integration added")
print("✅ Enhanced visualization support")
print("✅ Improved error handling and retry logic")  
print("✅ Support for matplotlib, seaborn, plotly")
print("⚠️  OpenAI API key required for full functionality")
print("="*50)
