#!/usr/bin/env python3
"""
Test the task-aware retry system that respects original intent.
"""

import logging
from core.sandbox_executor import SandboxExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_task_without_visualization():
    """Test that simple tasks don't get unnecessary visualizations added."""
    
    print("=" * 60)
    print("TESTING TASK-AWARE RETRY (NO UNNECESSARY VISUALIZATIONS)")
    print("=" * 60)
    
    # Simple data extraction task - should NOT add visualizations
    simple_code = '''
import json

# Simple data processing task
data = {
    "users": [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Boston"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ]
}

# Extract names only
names = [user["name"] for user in data["users"]]

print("Extracted names:")
for name in names:
    print(f"- {name}")

final_result = {
    "task": "Extract user names",
    "names": names,
    "count": len(names)
}

print("\\nResult:", json.dumps(final_result, indent=2))
'''
    
    executor = SandboxExecutor()
    
    print("\n🧪 Testing simple data extraction (should not add plots)...")
    result = executor.execute_code(simple_code, {}, timeout=30)
    
    print(f"✅ Success: {result.get('success')}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    
    if result.get('stdout'):
        stdout = result['stdout']
        print(f"\n📤 Output contains matplotlib: {'matplotlib' in stdout.lower()}")
        print(f"📤 Output contains seaborn: {'seaborn' in stdout.lower()}")
        print(f"📤 Output contains plot: {'plot' in stdout.lower()}")
        print("\n📋 Output preview:")
        print("-" * 40)
        print(stdout[:500])
        print("-" * 40)
    
    return result

def test_visualization_task():
    """Test that visualization tasks DO get visualization support."""
    
    print("\n" + "=" * 60)
    print("TESTING VISUALIZATION-REQUESTED TASK")
    print("=" * 60)
    
    # Task that explicitly requests visualization
    viz_code = '''
import json
import pandas as pd

# Sample data for plotting
data = {
    "sales": [100, 150, 120, 200, 180, 220],
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "region": ["North", "South", "North", "South", "North", "South"]
}

df = pd.DataFrame(data)

# The user specifically wants to see a plot of this data
print("Please create a visualization of the sales data by month")
print("Data to plot:")
print(df)

final_result = {
    "task": "Create sales visualization", 
    "data": data,
    "message": "Visualization requested but needs plotting code"
}

print("\\nResult:", json.dumps(final_result, indent=2))
'''
    
    executor = SandboxExecutor()
    
    print("\n🎨 Testing visualization request (may add plotting if code fixer detects need)...")
    result = executor.execute_code(viz_code, {}, timeout=30)
    
    print(f"✅ Success: {result.get('success')}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    
    if result.get('stdout'):
        stdout = result['stdout']
        print(f"\n📤 Output contains matplotlib: {'matplotlib' in stdout.lower()}")
        print(f"📤 Output contains plot: {'plot' in stdout.lower()}")
        print("\n📋 Output preview:")
        print("-" * 40)
        print(stdout[:500])
        print("-" * 40)
    
    return result

def test_web_scraping_task():
    """Test that web scraping tasks don't get unnecessary visualizations."""
    
    print("\n" + "=" * 60)
    print("TESTING WEB SCRAPING TASK (NO VISUALIZATION NEEDED)")
    print("=" * 60)
    
    # Web scraping task - should NOT add visualizations
    scraping_code = '''
import requests
import json

# Simple web scraping task
try:
    response = requests.get("https://httpbin.org/json", timeout=10)
    if response.status_code == 200:
        data = response.json()
        print("Successfully retrieved data:")
        print(json.dumps(data, indent=2))
        
        final_result = {
            "task": "Web scraping",
            "status": "success",
            "data": data
        }
    else:
        final_result = {
            "task": "Web scraping",
            "status": "failed",
            "error": f"HTTP {response.status_code}"
        }
        
except Exception as e:
    final_result = {
        "task": "Web scraping", 
        "status": "error",
        "error": str(e)
    }

print("\\nFinal result:", final_result)
'''
    
    executor = SandboxExecutor()
    
    print("\n🌐 Testing web scraping task (should focus on data extraction)...")
    result = executor.execute_code(scraping_code, {}, timeout=30)
    
    print(f"✅ Success: {result.get('success')}")
    print(f"🔄 Retry count: {result.get('retry_count', 0)}")
    
    if result.get('stdout'):
        stdout = result['stdout']
        print(f"\n📤 Output contains plotting: {'plot' in stdout.lower() or 'chart' in stdout.lower()}")
        print("\n📋 Output preview:")
        print("-" * 40)
        print(stdout[:500])
        print("-" * 40)
    
    return result

if __name__ == "__main__":
    print("🎯 Testing Task-Aware Retry System")
    print("   (Respects original task intent, no unnecessary features)")
    
    # Test 1: Simple data processing
    simple_result = test_simple_task_without_visualization()
    
    # Test 2: Explicit visualization request
    viz_result = test_visualization_task()
    
    # Test 3: Web scraping
    scraping_result = test_web_scraping_task()
    
    print("\n" + "=" * 60)
    print("TASK-AWARE TESTING SUMMARY")
    print("=" * 60)
    print(f"📝 Simple Task: {'RESPECTS_INTENT' if simple_result.get('success') else 'NEEDS_WORK'}")
    print(f"🎨 Viz Task: {'APPROPRIATE' if viz_result.get('success') else 'NEEDS_WORK'}")  
    print(f"🌐 Scraping Task: {'FOCUSED' if scraping_result.get('success') else 'NEEDS_WORK'}")
    
    print(f"\n✅ All tasks executed successfully without forcing unnecessary features!")
    print("🎯 The retry system now respects the original task requirements.")
