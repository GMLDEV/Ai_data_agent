from core.sandbox_executor import SandboxExecutor
import json

executor = SandboxExecutor()

# Test code that requires networkx
test_code = """
import networkx as nx
import json

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

# Basic graph analysis
result = {
    "nodes": list(G.nodes()),
    "edges": list(G.edges()),
    "num_nodes": G.number_of_nodes(),
    "num_edges": G.number_of_edges(),
    "is_connected": nx.is_connected(G),
    "average_clustering": nx.average_clustering(G),
    "density": nx.density(G)
}

print(json.dumps(result, indent=2))
"""

print('🧪 Testing NetworkX execution with fixed sandbox...')
result = executor.execute_code(test_code, {})
print(f'✅ Success: {result["success"]}')
if result['success']:
    print('📊 Output:', result['output'])
else:
    print('❌ Error:', result['error'])
    if result['stderr']:
        print('🔍 Stderr:', result['stderr'][:200])
