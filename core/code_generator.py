from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
import logging
from config import settings

logger = logging.getLogger(__name__)

class CodeGenerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = OpenAI(
            api_key=self.api_key,
            temperature=0.2,  # Low temperature for consistent code generation
            max_tokens=2000
        )
        
        self.code_generation_prompt = PromptTemplate(
            input_variables=["task_description", "file_manifest", "output_format", "workflow_type"],
            template="""
You are an expert Python programmer. Generate Python code to complete the user's task.

TASK: {task_description}

AVAILABLE FILES: {file_manifest}

WORKFLOW TYPE: {workflow_type}

OUTPUT FORMAT REQUIRED: {output_format}

REQUIREMENTS:
1. Write complete, executable Python code
2. Use only these allowed libraries: pandas, numpy, json, csv, re, os, sys, io, base64, math, datetime, collections
3. Handle file loading automatically (files will be available in current directory)
4. Include proper error handling
5. Generate the exact output format requested
6. Add comments explaining key steps
7. Print results clearly

IMPORTANT RULES:
- For CSV files: Use pandas to load and analyze
- Always print your final results
- Handle missing values appropriately
- If creating visualizations, describe what would be plotted (matplotlib not available yet)
- Store final results in a variable called 'final_result'

Generate ONLY the Python code, no explanations or markdown:
"""
        )
        
        self.repair_prompt = PromptTemplate(
            input_variables=["original_code", "error_message", "task_description"],
            template="""
The following Python code failed with an error. Please fix it.

ORIGINAL CODE:
{original_code}

ERROR MESSAGE:
{error_message}

ORIGINAL TASK: {task_description}

Please provide the corrected Python code. Fix the specific error while maintaining the original intent.
Generate ONLY the corrected Python code, no explanations:
"""
        )
    
    def generate_code(self, task_description: str, manifest: Dict[str, Any], 
                     workflow_type: str, output_format: str = "json") -> str:
        """Generate Python code for the given task"""
        try:
            # Prepare file manifest summary
            file_summary = self._create_file_summary(manifest)
            
            prompt = self.code_generation_prompt.format(
                task_description=task_description,
                file_manifest=file_summary,
                workflow_type=workflow_type,
                output_format=output_format
            )
            
            logger.info(f"Generating code for {workflow_type} workflow...")
            code = self.llm.invoke(prompt)
            
            # Clean up the response
            code = self._clean_code_response(code)
            
            logger.info("Code generation completed")
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Return a basic template as fallback
            return self._generate_fallback_code(task_description, manifest)
    
    def repair_code(self, original_code: str, error_message: str, task_description: str) -> str:
        """Attempt to repair failed code"""
        try:
            prompt = self.repair_prompt.format(
                original_code=original_code,
                error_message=error_message,
                task_description=task_description
            )
            
            logger.info("Attempting code repair...")
            repaired_code = self.llm.invoke(prompt)
            
            return self._clean_code_response(repaired_code)
            
        except Exception as e:
            logger.error(f"Code repair failed: {e}")
            return original_code  # Return original if repair fails
    
    def _create_file_summary(self, manifest: Dict[str, Any]) -> str:
        """Create a summary of available files for code generation"""
        if not manifest.get('files'):
            return "No files available"
        
        summaries = []
        for filename, info in manifest['files'].items():
            # Ensure info is a dictionary
            if not isinstance(info, dict):
                summaries.append(f"- {filename}: unknown file (invalid format)")
                continue
                
            file_type = info.get('type', 'unknown')
            
            if file_type == 'csv':
                shape = info.get('shape', [0, 0])
                columns = info.get('columns', [])
                summary = f"- {filename}: CSV with {shape[0]} rows, {shape[1]} columns"
                if columns:
                    summary += f"\n  Columns: {', '.join(columns)}"
                if info.get('sample_rows'):
                    summary += f"\n  Sample data: {info['sample_rows'][0] if info['sample_rows'] else 'None'}"
            
            elif file_type == 'image':
                dims = info.get('dimensions', [0, 0])
                summary = f"- {filename}: Image file ({dims[0]}x{dims[1]} pixels)"
            
            elif file_type == 'text':
                lines = info.get('line_count', 0)
                summary = f"- {filename}: Text file ({lines} lines)"
            
            else:
                summary = f"- {filename}: {file_type} file ({info.get('size', 0)} bytes)"
            
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def _clean_code_response(self, code: str) -> str:
        """Clean up the LLM's code response"""
        # Remove markdown code blocks
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        
        if code.endswith("```"):
            code = code[:-3]
        
        # Remove extra whitespace
        code = code.strip()
        
        return code
    
    def _generate_fallback_code(self, task_description: str, manifest: Dict[str, Any]) -> str:
        """Generate basic fallback code when LLM fails"""
        files = manifest.get('files', {})
        has_csv = False
        
        # Check for CSV files with proper validation
        for filename, file_info in files.items():
            if isinstance(file_info, dict) and file_info.get('type') == 'csv':
                has_csv = True
                break
        
        if has_csv:
            # CSV analysis fallback
            csv_file = next((name for name, info in manifest['files'].items() if info.get('type') == 'csv'), None)
            return f'''
import pandas as pd
import json

# Load the CSV file
try:
    df = pd.read_csv("{csv_file}")
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\\nFirst few rows:")
    print(df.head())
    
    # Basic analysis
    print("\\nBasic statistics:")
    print(df.describe())
    
    final_result = {{
        "message": "Basic CSV analysis completed",
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "summary": "Fallback analysis - LLM code generation failed"
    }}
    
    print("\\nFinal result:")
    print(json.dumps(final_result, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
    final_result = {{"error": str(e)}}
'''
        else:
            return f'''
print("Task: {task_description}")
print("Fallback code executed - LLM generation failed")

final_result = {{
    "message": "Fallback execution",
    "task": "{task_description}",
    "status": "LLM code generation failed, using basic template"
}}

print("Final result:", final_result)
'''