from typing import Dict, List, Any, Optional
import json
import re
from workflows.base import BaseWorkflow
from core.code_generator import CodeGenerator
from core.sandbox_executor import SandboxExecutor
import logging

logger = logging.getLogger(__name__)

class DynamicCodeExecutionWorkflow(BaseWorkflow):
    """
    General-purpose workflow that can handle any request by generating
    and executing Python code dynamically.
    """
    
    def __init__(self, code_generator: CodeGenerator, sandbox_executor: SandboxExecutor):
        super().__init__()
        self.code_generator = code_generator
        self.sandbox_executor = sandbox_executor
        self.max_retries = 3
        self.execution_timeout = 180  # 3 minutes
        
    def plan(self, questions: List[str], file_manifest: Dict[str, Any], 
             keywords: List[str], urls: List[str]) -> Dict[str, Any]:
        """
        Analyze the request and create an execution plan.
        
        Returns:
            Dict containing:
            - data_sources: List of data sources to use
            - analysis_type: Type of analysis needed
            - output_format: Expected output format
            - visualization_needed: Whether plots are required
            - libraries_needed: Python libraries to include
        """
        logger.info("Planning dynamic execution workflow")
        
        plan = {
            "data_sources": [],
            "analysis_type": "general",
            "output_format": "json",
            "visualization_needed": False,
            "libraries_needed": ["pandas", "numpy"]
        }
        
        # Analyze file manifest
        for filename, file_info in file_manifest.items():
            if file_info.get("type") == "csv":
                plan["data_sources"].append({
                    "type": "csv",
                    "filename": filename,
                    "columns": file_info.get("columns", [])
                })
                plan["analysis_type"] = "data_analysis"
                
            elif file_info.get("type") in ["png", "jpg", "jpeg"]:
                plan["data_sources"].append({
                    "type": "image",
                    "filename": filename
                })
                plan["analysis_type"] = "image_analysis"
                plan["libraries_needed"].extend(["PIL", "cv2"])
                
        # Analyze URLs
        if urls:
            for url in urls:
                plan["data_sources"].append({
                    "type": "url",
                    "url": url
                })
                plan["analysis_type"] = "web_scraping"
                plan["libraries_needed"].extend(["requests", "beautifulsoup4"])
        
        # Detect output format from questions
        questions_text = " ".join(questions).lower()
        
        if "json" in questions_text or "array" in questions_text:
            plan["output_format"] = "json"
        elif "csv" in questions_text:
            plan["output_format"] = "csv"
        elif "plot" in questions_text or "chart" in questions_text or "graph" in questions_text:
            plan["visualization_needed"] = True
            plan["libraries_needed"].extend(["matplotlib", "seaborn"])
            
        # Detect specific analysis keywords
        if any(word in questions_text for word in ["correlation", "regression", "statistics"]):
            plan["analysis_type"] = "statistical_analysis"
            plan["libraries_needed"].append("scipy")
            
        logger.info(f"Generated plan: {plan}")
        return plan
    
    def generate_code(self, questions: List[str], file_manifest: Dict[str, Any], 
                     plan: Dict[str, Any]) -> str:
        """Generate Python code based on the execution plan."""
        
        code_prompt = self._build_code_generation_prompt(questions, file_manifest, plan)
        
        try:
            code = self.code_generator.generate_code(
                prompt=code_prompt,
                context={
                    "questions": questions,
                    "file_manifest": file_manifest,
                    "plan": plan
                }
            )
            logger.info("Generated code successfully")
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    def execute(self, code: str, file_manifest: Dict[str, Any], 
               plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generated code with retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Execution attempt {attempt + 1}/{self.max_retries + 1}")
                
                result = self.sandbox_executor.execute_code(
                    code=code,
                    files=file_manifest,
                    timeout=self.execution_timeout,
                    allowed_libraries=plan.get("libraries_needed", [])
                )
                
                if result["success"]:
                    logger.info("Code executed successfully")
                    return result
                else:
                    logger.warning(f"Execution failed: {result.get('error', 'Unknown error')}")
                    if attempt < self.max_retries:
                        # Try to repair the code
                        code = self.repair(code, result, plan)
                    else:
                        return result
                        
            except Exception as e:
                logger.error(f"Execution attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "output": None,
                        "artifacts": {}
                    }
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def validate(self, result: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the execution result against expected output format."""
        
        if not result["success"]:
            return {
                "valid": False,
                "errors": [f"Execution failed: {result.get('error', 'Unknown error')}"]
            }
        
        validation_errors = []
        output = result.get("output")
        
        # Validate output format
        expected_format = plan.get("output_format", "json")
        
        if expected_format == "json":
            try:
                if isinstance(output, str):
                    json.loads(output)
                elif not isinstance(output, (dict, list)):
                    validation_errors.append("Output should be valid JSON")
            except json.JSONDecodeError:
                validation_errors.append("Output is not valid JSON")
                
        elif expected_format == "csv":
            if not isinstance(output, str) or not output.strip():
                validation_errors.append("Output should be non-empty CSV string")
        
        # Validate visualization requirements
        if plan.get("visualization_needed") and not result.get("artifacts", {}).get("plots"):
            validation_errors.append("Visualization was requested but no plots were generated")
        
        # Check output size constraints
        if output and len(str(output)) > 100000:  # 100KB limit
            validation_errors.append("Output exceeds size limit (100KB)")
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors
        }
    
    def repair(self, code: str, execution_result: Dict[str, Any], 
              plan: Dict[str, Any]) -> str:
        """Fix the code based on execution errors."""
        
        error_info = execution_result.get("error", "")
        stderr = execution_result.get("stderr", "")
        
        repair_prompt = f"""
The following Python code failed to execute:

```python
{code}
```

Error details:
- Error: {error_info}
- Stderr: {stderr}

Expected output format: {plan.get('output_format', 'json')}
Available libraries: {', '.join(plan.get('libraries_needed', []))}

Please fix the code to:
1. Handle the error that occurred
2. Produce output in the expected format
3. Follow Python best practices
4. Include proper error handling

Return only the corrected Python code:
"""
        
        try:
            fixed_code = self.code_generator.generate_code(
                prompt=repair_prompt,
                context={
                    "original_code": code,
                    "error": error_info,
                    "plan": plan
                }
            )
            logger.info("Code repaired successfully")
            return fixed_code
            
        except Exception as e:
            logger.error(f"Code repair failed: {e}")
            return code  # Return original code if repair fails
    
    def _build_code_generation_prompt(self, questions: List[str], 
                                    file_manifest: Dict[str, Any], 
                                    plan: Dict[str, Any]) -> str:
        """Build the prompt for code generation."""
        
        files_info = []
        for filename, file_info in file_manifest.items():
            files_info.append(f"- {filename}: {file_info.get('type', 'unknown')} file")
            if file_info.get('columns'):
                files_info.append(f"  Columns: {', '.join(file_info['columns'][:5])}")  # First 5 columns
        
        data_sources_info = []
        for source in plan["data_sources"]:
            if source["type"] == "csv":
                data_sources_info.append(f"CSV file: {source['filename']}")
            elif source["type"] == "url":
                data_sources_info.append(f"Web URL: {source['url']}")
            elif source["type"] == "image":
                data_sources_info.append(f"Image file: {source['filename']}")
        
        prompt = f"""
Generate Python code to answer the following questions:

Questions:
{chr(10).join(f"- {q}" for q in questions)}

Available data sources:
{chr(10).join(data_sources_info) if data_sources_info else "- None specified"}

Available files:
{chr(10).join(files_info) if files_info else "- None"}

Requirements:
- Output format: {plan.get('output_format', 'json')}
- Visualization needed: {plan.get('visualization_needed', False)}
- Available libraries: {', '.join(plan.get('libraries_needed', []))}

Instructions:
1. Load and process the data sources
2. Perform the requested analysis
3. Generate output in the specified format
4. If visualization is needed, save plots as artifacts
5. Handle errors gracefully
6. Keep output under 100KB

Important:
- Files are available in the current directory
- For CSV files, use pandas.read_csv()
- For web URLs, use requests and BeautifulSoup
- For images, use PIL or cv2
- Print the final result to stdout
- Save any plots as PNG files

Return only the Python code:
"""
        
        return prompt