"""
Test script to verify NetworkX is now properly detected and installed in sandbox
"""

import requests
import json
import tempfile
import os

def test_networkx_detection():
    """Test that NetworkX imports are detected and installed"""
    
    # Sample network analysis code that should trigger NetworkX installation
    test_code = '''
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO

# Create a simple network
edges_data = [
    ('Alice', 'Bob'),
    ('Alice', 'Carol'),
    ('Bob', 'Carol'),
    ('Carol', 'Dave'),
    ('Dave', 'Eve'),
    ('Alice', 'Eve'),
    ('Bob', 'Eve')
]

# Create graph
G = nx.Graph()
G.add_edges_from(edges_data)

# Calculate network metrics
result = {
    'edge_count': G.number_of_edges(),
    'node_count': G.number_of_nodes(),
    'density': nx.density(G),
    'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
}

print(json.dumps(result, indent=2))
'''
    
    # Test the sandbox executor with NetworkX code
    print("🧪 Testing NetworkX detection in sandbox...")
    print("Generated code uses NetworkX imports:")
    print("- import networkx as nx")
    print("")
    
    # Check if requirements detection works
    from core.sandbox_executor import SandboxExecutor
    executor = SandboxExecutor()
    
    # Create temp dir to test requirements detection
    with tempfile.TemporaryDirectory() as temp_dir:
        executor._create_requirements_file(temp_dir, test_code)
        
        requirements_path = os.path.join(temp_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print("✅ requirements.txt created")
            with open(requirements_path, 'r') as f:
                content = f.read()
                print(f"📋 Requirements content:\n{content}")
                
                if 'networkx' in content:
                    print("✅ NetworkX detected and added to requirements!")
                    return True
                else:
                    print("❌ NetworkX NOT detected in requirements")
                    return False
        else:
            print("❌ requirements.txt not created")
            return False

def test_api_call():
    """Test the actual API with edges.csv network analysis"""
    
    print("\n🌐 Testing full API call with network analysis...")
    
    # Create edges.csv content
    edges_csv = """source,target
Alice,Bob
Alice,Carol
Bob,Carol
Carol,Dave
Dave,Eve
Alice,Eve
Bob,Eve"""
    
    # Test query for network analysis
    query = """Use the undirected network in `edges.csv`.

Return a JSON object with keys:
- `edge_count`: number
- `highest_degree_node`: string
- `average_degree`: number
- `density`: number
- `shortest_path_alice_eve`: number
- `network_graph`: base64 PNG string under 100kB
- `degree_histogram`: base64 PNG string under 100kB

Answer:
1. How many edges are in the network?
2. Which node has the highest degree?
3. What is the average degree of the network?
4. What is the network density?
5. What is the length of the shortest path between Alice and Eve?
6. Draw the network with nodes labelled and edges shown. Encode as base64 PNG.
7. Plot the degree distribution as a bar chart with green bars. Encode as base64 PNG."""

    try:
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(edges_csv)
            csv_path = f.name
        
        # Make API call
        url = "http://localhost:8000/api/v1/process-request"
        
        with open(csv_path, 'rb') as f:
            files = {'file': ('edges.csv', f, 'text/csv')}
            data = {'query': query}
            
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API call successful!")
                
                if result.get('success', False):
                    print("✅ Network analysis completed successfully!")
                    answer = result.get('answer', {})
                    print(f"📊 Edge count: {answer.get('edge_count', 'N/A')}")
                    print(f"📊 Highest degree node: {answer.get('highest_degree_node', 'N/A')}")
                    return True
                else:
                    print("❌ Network analysis failed:")
                    print(f"Error: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(response.text)
                return False
                
        # Clean up
        os.unlink(csv_path)
        
    except requests.exceptions.ConnectionError:
        print("⚠️ API server not running on localhost:8000")
        print("Start the server with: python main.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing NetworkX fixes...\n")
    
    # Test 1: Requirements detection
    test1_passed = test_networkx_detection()
    
    # Test 2: Full API call
    test2_passed = test_api_call()
    
    print(f"\n📋 Test Results:")
    print(f"Requirements Detection: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"API Network Analysis: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! NetworkX should now work properly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
