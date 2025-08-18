"""
Simple test to verify NetworkX installation works in sandbox
"""

from core.sandbox_executor import SandboxExecutor

def test_networkx_installation():
    """Test that NetworkX can be installed and used in sandbox"""
    
    # Simple NetworkX test code
    test_code = '''
import networkx as nx
import json

# Create a simple graph
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

# Calculate some basic metrics
result = {
    'node_count': G.number_of_nodes(),
    'edge_count': G.number_of_edges(),
    'density': nx.density(G)
}

print(json.dumps(result))
'''
    
    print("🧪 Testing NetworkX installation in sandbox...")
    
    # Execute the test
    executor = SandboxExecutor()
    result = executor.execute_code(
        code=test_code,
        files={},
        timeout=180
    )
    
    print(f"✅ Execution success: {result.get('success', False)}")
    
    if result.get('success'):
        output = result.get('output', '')
        print(f"📊 Output: {output}")
        
        if 'node_count' in output and 'edge_count' in output:
            print("✅ NetworkX test PASSED - NetworkX working correctly!")
            return True
        else:
            print("❌ NetworkX test FAILED - Unexpected output")
            return False
    else:
        print("❌ Execution failed:")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Stderr: {result.get('stderr', '')}")
        return False

if __name__ == "__main__":
    success = test_networkx_installation()
    print(f"\n🎯 NetworkX test {'PASSED' if success else 'FAILED'}")
