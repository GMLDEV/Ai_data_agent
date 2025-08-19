#!/usr/bin/env python3
"""
Test script to simulate evaluator requests using the same format as curl -F
"""

import requests
import tempfile
import os

def test_evaluator_format():
    """Test the /process-request endpoint with evaluator format"""
    
    # Create temporary files to simulate evaluator files
    questions_content = """Analyze `sample-sales.csv`.

Return a JSON object with keys:
- `total_sales`: number
- `top_region`: string

Answer:
1. What is the total sales across all regions?
2. Which region has the highest total sales?
"""
    
    csv_content = """region,sales,date
North,1000,2023-01-01
South,1500,2023-01-02
East,1200,2023-01-03
West,800,2023-01-04
North,1100,2023-01-05
"""
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(questions_content)
        questions_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_file = f.name
    
    try:
        # Test the request format that evaluators use
        url = "http://localhost:8000/process-request"
        
        # Prepare files like curl -F does
        files = {
            'questions.txt': ('questions.txt', open(questions_file, 'rb'), 'text/plain'),
            'sample-sales.csv': ('sample-sales.csv', open(csv_file, 'rb'), 'text/csv')
        }
        
        print("🧪 Testing evaluator request format...")
        print(f"📡 Sending POST to {url}")
        print("📁 Files being sent:")
        print("   - questions.txt (text/plain)")
        print("   - sample-sales.csv (text/csv)")
        
        # Send the request
        response = requests.post(url, files=files)
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Request processed successfully!")
            result = response.json()
            print(f"📈 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        else:
            print("❌ FAILED: Request failed!")
            print(f"🚨 Error: {response.text}")
        
        # Close file handles
        for file_tuple in files.values():
            file_tuple[1].close()
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Test Error: {e}")
    finally:
        # Cleanup temporary files
        try:
            os.unlink(questions_file)
            os.unlink(csv_file)
        except:
            pass

if __name__ == "__main__":
    test_evaluator_format()
