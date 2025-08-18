"""
Test the fallback system with format compliance scoring
"""

import requests
import json
import tempfile
import os

def test_fallback_system():
    """Test the fallback system with different scenarios"""
    
    print("🧪 Testing Fallback System...")
    print("This will test:")
    print("1. Primary workflow success (high compliance)")
    print("2. Primary workflow failure → LLM fallback")
    print("3. Complete failure → Emergency response")
    print("")
    
    # Test cases
    test_cases = [
        {
            "name": "NetworkX Analysis (Should work)",
            "csv_content": "source,target\nAlice,Bob\nAlice,Carol\nBob,Carol\nCarol,Dave",
            "query": """Use the undirected network in edges.csv.

Return a JSON object with keys:
- edge_count: number
- highest_degree_node: string  
- average_degree: number
- density: number

Answer: How many edges and what's the network density?""",
            "expected_score": ">= 0.7"
        },
        {
            "name": "Impossible Analysis (Should fallback)",
            "csv_content": "invalid,data\nbroken,format",
            "query": """Analyze quantum entanglement in the data using non-existent libraries.

Return a JSON object with keys:
- quantum_state: string
- entanglement_level: number
- probability_amplitude: number
- measurement_result: boolean

Answer: What is the quantum state?""",
            "expected_score": "fallback"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔬 Test {i}: {test_case['name']}")
        print(f"Expected: {test_case['expected_score']}")
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_case['csv_content'])
            csv_path = f.name
        
        try:
            # Make API call
            url = "http://localhost:8000/api/v1/process-request"
            
            with open(csv_path, 'rb') as f:
                files = {'file': ('edges.csv', f, 'text/csv')}
                data = {'query': test_case['query']}
                
                response = requests.post(url, files=files, data=data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ API Response received")
                    
                    # Check response format
                    if result.get('success', False):
                        answer = result.get('answer', {})
                        print(f"📊 Answer type: {type(answer).__name__}")
                        
                        if isinstance(answer, dict):
                            print(f"📊 Keys: {list(answer.keys())}")
                            
                            # Check for fallback indicators
                            if answer.get('_emergency_response'):
                                print("🚨 EMERGENCY RESPONSE detected")
                            elif answer.get('fallback'):
                                print("🔄 LLM FALLBACK detected")
                            else:
                                print("✅ PRIMARY WORKFLOW success")
                        
                        print(f"📝 Sample answer: {str(answer)[:200]}...")
                    else:
                        print(f"❌ API returned error: {result.get('error', 'Unknown')}")
                else:
                    print(f"❌ API call failed: {response.status_code}")
                    
        except requests.exceptions.ConnectionError:
            print("⚠️ API server not running on localhost:8000")
            print("Start server: python main.py")
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        # Clean up
        os.unlink(csv_path)
        print("-" * 50)

def test_format_compliance_manually():
    """Test format compliance scoring manually"""
    
    print("\n🎯 Testing Format Compliance Scoring...")
    
    # Mock question with JSON structure
    questions = """Return a JSON object with keys:
- edge_count: number
- node_name: string
- is_connected: boolean"""
    
    test_answers = [
        {
            "name": "Perfect Match",
            "answer": {"edge_count": 5, "node_name": "Alice", "is_connected": True},
            "expected": "High (>= 0.9)"
        },
        {
            "name": "Missing Key",
            "answer": {"edge_count": 5, "node_name": "Alice"},
            "expected": "Medium (0.7-0.9)"
        },
        {
            "name": "Extra Key",
            "answer": {"edge_count": 5, "node_name": "Alice", "is_connected": True, "extra": "data"},
            "expected": "Good (0.8-0.9)"
        },
        {
            "name": "Error Response", 
            "answer": {"error": "Workflow failed"},
            "expected": "Very Low (0.1)"
        }
    ]
    
    # This would require importing the orchestrator class to test manually
    # For now, just show the test structure
    for test in test_answers:
        print(f"📊 {test['name']}: {test['answer']}")
        print(f"   Expected score: {test['expected']}")
        print()

if __name__ == "__main__":
    print("🔧 Fallback System Test Suite")
    print("="*50)
    
    # Test 1: API integration test
    test_fallback_system()
    
    # Test 2: Format compliance test
    test_format_compliance_manually()
    
    print("\n🎯 Key Logging Features:")
    print("✅ FORMAT COMPLIANCE scoring (0.0-1.0)")
    print("✅ EMERGENCY RESPONSE detection")
    print("✅ LLM FALLBACK detection") 
    print("✅ Docker-friendly structured logs")
    print("✅ Critical failure tracking")
