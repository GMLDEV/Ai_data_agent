import json
import tempfile
import os
from core.orchestrator import Orchestrator

print('🧪 Testing File Path Resolution Fix...')

# Create a test CSV file
test_csv_content = """source,target,weight
A,B,0.5
B,C,0.3
C,A,0.8
A,D,0.2
D,B,0.6"""

# Create temporary CSV file  
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write(test_csv_content)
    temp_csv_path = f.name

print(f'Created test CSV at: {temp_csv_path}')

# Read the file as bytes (simulating file upload)
with open(temp_csv_path, 'rb') as f:
    file_content = f.read()

# Initialize orchestrator
orchestrator = Orchestrator()

# Test with file provided
files = {'edges.csv': file_content}
request = 'Analyze the network data in edges.csv and calculate basic graph statistics'

print(f'Request: {request}')
print(f'Files provided: {list(files.keys())}')

try:
    result = orchestrator.process_request(request, files)
    if isinstance(result, str):
        print('✅ Success: Workflow executed successfully')
        print(f'Output preview: {result[:300]}...')
    else:
        print(f'Result type: {type(result)}')
        print(f'Full result: {result}')
        success = result.get("success", False) if hasattr(result, "get") else "Unknown"
        print(f'Success: {success}')
        if not success and hasattr(result, "get"):
            error = result.get("error", "Unknown error")
            stderr = result.get("stderr", "")
            print(f'Error: {error}')
            if stderr:
                print(f'Stderr: {stderr}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

# Cleanup
try:
    os.unlink(temp_csv_path)
    print('🧹 Cleaned up test file')
except:
    pass

print('🎯 File path resolution test complete!')
