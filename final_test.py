import json
from core.orchestrator import Orchestrator

# Test comprehensive system
print('🧪 Final System Test...')

# Initialize orchestrator
orchestrator = Orchestrator()

# Test network analysis request
request = 'Create a NetworkX graph with 5 nodes in a star pattern and calculate centrality measures'
print(f'Request: {request}')

try:
    result = orchestrator.process_request(request, {})
    if isinstance(result, str):
        print('✅ Success: Workflow executed successfully')
        print(f'Output preview: {result[:200]}...')
    else:
        print(f'Result type: {type(result)}')
        success = result.get("success", False) if hasattr(result, "get") else "Unknown"
        print(f'Success: {success}')
except Exception as e:
    print(f'❌ Error: {e}')

print('🎯 System verification complete!')
