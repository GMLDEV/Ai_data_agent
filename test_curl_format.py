#!/usr/bin/env python3
"""
Test script to simulate evaluator's curl request format
"""
import requests
import os

def test_curl_format():
    """Test the /process-request endpoint with curl-style multipart form data"""
    
    # Prepare the files
    questions_content = """Analyze `sample-sales.csv`.

Return a JSON object with keys:
- `total_sales`: number
- `top_region`: string
- `day_sales_correlation`: number
- `bar_chart`: base64 PNG string under 100kB
- `median_sales`: number
- `total_sales_tax`: number
- `cumulative_sales_chart`: base64 PNG string under 100kB

Answer:
1. What is the total sales across all regions?
2. Which region has the highest total sales?
3. What is the correlation between day of month and sales? (Use the date column.)
4. Plot total sales by region as a bar chart with blue bars. Encode as base64 PNG.
5. What is the median sales amount across all orders?
6. What is the total sales tax if the tax rate is 10%?
7. Plot cumulative sales over time as a line chart with a red line. Encode as base64 PNG.
"""
    
    # Create temporary files
    with open("temp_questions.txt", "w", encoding="utf-8") as f:
        f.write(questions_content)
    
    # Prepare the multipart form data exactly like curl does
    files = {
        'questions.txt': ('questions.txt', open('temp_questions.txt', 'r', encoding='utf-8'), 'text/plain'),
        'sample-sales.csv': ('sample-sales.csv', open('sample-sales.csv', 'rb'), 'text/csv')
    }
    
    try:
        print("🧪 Testing /process-request endpoint with curl-style format...")
        print("📡 Sending multipart form data:")
        print("   - questions.txt (form field)")
        print("   - sample-sales.csv (form field)")
        
        response = requests.post(
            "http://localhost:8000/process-request",
            files=files,
            timeout=60
        )
        
        print(f"📊 Response Status: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Files were properly processed:")
            print(f"   - Files detected: {len(result.get('files', []))}")
            if 'files' in result:
                for file_info in result['files']:
                    print(f"     * {file_info['name']} ({file_info['size']} bytes)")
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    finally:
        # Close and cleanup
        for file_obj in files.values():
            if hasattr(file_obj[1], 'close'):
                file_obj[1].close()
        
        # Remove temporary file
        if os.path.exists("temp_questions.txt"):
            os.remove("temp_questions.txt")

if __name__ == "__main__":
    test_curl_format()
