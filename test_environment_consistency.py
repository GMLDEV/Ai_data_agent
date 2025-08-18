"""
Comprehensive test to verify NetworkX installation and environment consistency
"""

from core.sandbox_executor import SandboxExecutor
import json

def test_environment_consistency():
    """Test that packages are installed and accessible in the same environment"""
    
    # Test code that requires NetworkX and shows environment info
    test_code = '''
import sys
import json
import os

# Show environment info
env_info = {
    'python_executable': sys.executable,
    'python_version': sys.version,
    'working_directory': os.getcwd(),
    'python_path': sys.path[:3],  # First 3 entries
}

print("=== ENVIRONMENT INFO ===")
print(json.dumps(env_info, indent=2))

try:
    # Test NetworkX import and basic functionality
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create a simple network
    G = nx.Graph()
    G.add_edges_from([
        ('Alice', 'Bob'),
        ('Alice', 'Carol'), 
        ('Bob', 'Carol'),
        ('Carol', 'Dave')
    ])
    
    # Calculate network metrics
    result = {
        'success': True,
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges(), 
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'is_connected': nx.is_connected(G)
    }
    
    print("=== NETWORK ANALYSIS RESULT ===")
    print(json.dumps(result, indent=2))
    
except ImportError as e:
    error_result = {
        'success': False,
        'error': f'Import failed: {str(e)}',
        'missing_module': str(e).split("'")[1] if "'" in str(e) else 'unknown'
    }
    
    print("=== ERROR RESULT ===")
    print(json.dumps(error_result, indent=2))

except Exception as e:
    error_result = {
        'success': False,
        'error': f'Execution failed: {str(e)}',
        'error_type': type(e).__name__
    }
    
    print("=== ERROR RESULT ===") 
    print(json.dumps(error_result, indent=2))
'''
    
    print("🧪 Testing environment consistency and NetworkX installation...")
    print("This will test:")
    print("1. Package installation in correct environment")
    print("2. Environment verification") 
    print("3. NetworkX functionality")
    print()
    
    # Execute the test
    executor = SandboxExecutor()
    result = executor.execute_code(
        code=test_code,
        files={},
        timeout=180
    )
    
    print("📊 EXECUTION RESULTS:")
    print(f"Success: {result.get('success', False)}")
    print(f"Return Code: {result.get('return_code', 'N/A')}")
    
    if result.get('output'):
        print("\n📋 OUTPUT:")
        print(result['output'])
    
    if result.get('stderr'):
        print("\n❌ STDERR:")
        print(result['stderr'])
        
    if result.get('error'):
        print(f"\n⚠️ ERROR: {result['error']}")
    
    # Analyze the output
    output = result.get('output', '')
    
    if 'NETWORK ANALYSIS RESULT' in output and '"success": true' in output.lower():
        print("\n✅ ENVIRONMENT CONSISTENCY TEST PASSED!")
        print("✅ NetworkX installation and execution successful!")
        return True
    elif 'Import failed' in output:
        print("\n❌ PACKAGE IMPORT FAILED!")
        print("❌ Environment mismatch detected!")
        return False
    else:
        print("\n⚠️ UNEXPECTED RESULT - check output above")
        return False

if __name__ == "__main__":
    success = test_environment_consistency()
    
    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Environment consistency maintained")
        print("✅ NetworkX working properly")
    else:
        print("❌ TESTS FAILED!")
        print("❌ Check logs above for environment issues")
    print("="*50)
