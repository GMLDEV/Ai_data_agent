#!/usr/bin/env python3
"""
Dynamic JSON Structure Parser
Parses questions.txt to understand the expected JSON response structure.
"""

import re
import json
from typing import Dict, List, Any, Optional

class QuestionParser:
    """Parse questions.txt to extract expected JSON response structure"""
    
    def __init__(self):
        pass
    
    def parse_json_structure(self, questions_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse the questions content to extract expected JSON structure.
        
        Returns:
            Dict with 'keys' (list of expected keys) and 'types' (dict of key->type mappings)
        """
        # Look for "Return a JSON object with keys:" section
        json_section_pattern = r'Return a JSON object with keys:\s*\n((?:- `[^`]+`:[^\n]+\n?)*)'
        match = re.search(json_section_pattern, questions_content, re.MULTILINE | re.IGNORECASE)
        
        if not match:
            return None
        
        keys_section = match.group(1)
        
        # Parse individual key definitions
        key_pattern = r'- `([^`]+)`:\s*(.+)'
        key_matches = re.findall(key_pattern, keys_section)
        
        expected_keys = []
        key_types = {}
        
        for key_name, key_type_desc in key_matches:
            expected_keys.append(key_name)
            
            # Determine the expected data type
            key_type_desc = key_type_desc.lower().strip()
            if 'number' in key_type_desc:
                key_types[key_name] = 'number'
            elif 'string' in key_type_desc:
                key_types[key_name] = 'string'
            elif 'base64' in key_type_desc or 'png' in key_type_desc:
                key_types[key_name] = 'base64_image'
            else:
                key_types[key_name] = 'unknown'
        
        return {
            'keys': expected_keys,
            'types': key_types,
            'total_keys': len(expected_keys)
        }
    
    def extract_analysis_questions(self, questions_content: str) -> List[str]:
        """Extract the numbered analysis questions"""
        # Look for "Answer:" section followed by numbered questions
        answer_section_pattern = r'Answer:\s*\n((?:\d+\..*\n?)*)'
        match = re.search(answer_section_pattern, questions_content, re.MULTILINE)
        
        if not match:
            return []
        
        answer_section = match.group(1)
        
        # Extract numbered questions
        question_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)'
        questions = re.findall(question_pattern, answer_section, re.DOTALL)
        
        # Clean up questions
        cleaned_questions = []
        for q in questions:
            cleaned = q.strip().replace('\n', ' ')
            cleaned_questions.append(cleaned)
        
        return cleaned_questions

def test_question_parser():
    """Test the question parser with sample questions"""
    print("🧪 Testing Question Parser")
    print("=" * 50)
    
    # Test with sample questions
    sample_questions = [
        {
            "name": "Sales Analysis",
            "content": """
Analyze `sample-sales.csv`.

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
        },
        {
            "name": "Network Analysis", 
            "content": """
Use the undirected network in `edges.csv`.

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
7. Plot the degree distribution as a bar chart with green bars. Encode as base64 PNG.
"""
        },
        {
            "name": "Weather Analysis",
            "content": """
Analyze `sample-weather.csv`.

Return a JSON object with keys:
- `average_temp_c`: number
- `max_precip_date`: string
- `min_temp_c`: number
- `temp_precip_correlation`: number
- `average_precip_mm`: number
- `temp_line_chart`: base64 PNG string under 100kB
- `precip_histogram`: base64 PNG string under 100kB

Answer:
1. What is the average temperature in Celsius?
2. On which date was precipitation highest?
3. What is the minimum temperature recorded?
4. What is the correlation between temperature and precipitation?
5. What is the average precipitation in millimeters?
6. Plot temperature over time as a line chart with a red line. Encode as base64 PNG.
7. Plot precipitation as a histogram with orange bars. Encode as base64 PNG.
"""
        }
    ]
    
    parser = QuestionParser()
    
    for i, sample in enumerate(sample_questions, 1):
        print(f"\n📝 Test {i}: {sample['name']}")
        
        # Parse JSON structure
        structure = parser.parse_json_structure(sample['content'])
        
        if structure:
            print(f"✅ Found JSON structure with {structure['total_keys']} keys:")
            for key in structure['keys']:
                key_type = structure['types'].get(key, 'unknown')
                print(f"   - {key}: {key_type}")
            
            # Parse analysis questions
            questions = parser.extract_analysis_questions(sample['content'])
            print(f"📋 Found {len(questions)} analysis questions:")
            for j, q in enumerate(questions[:3], 1):  # Show first 3
                print(f"   {j}. {q[:60]}{'...' if len(q) > 60 else ''}")
        else:
            print("❌ Failed to parse JSON structure")
    
    return True

if __name__ == "__main__":
    test_question_parser()
