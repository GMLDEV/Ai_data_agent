from typing import Dict, List, Any, Optional
import json
import re
from workflows.base import BaseWorkflow
from core.code_generator import CodeGenerator
from sandbox.executor import SandboxExecutor
from core.llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)

class DynamicCodeExecutionWorkflow(BaseWorkflow):
    """
    General-purpose workflow that can handle any request by generating
    and executing Python code dynamically.
    """
    
    def __init__(self, code_generator: CodeGenerator, manifest: Dict[str, Any], sandbox_executor: Optional[SandboxExecutor] = None, llm_client=None):
        super().__init__(code_generator=code_generator, manifest=manifest)
        self.code_generator = code_generator
        self.sandbox_executor = sandbox_executor
        self.llm_client = llm_client or LLMClient()
        self.max_retries = 3
        self.execution_timeout = 180  # 3 minutes
        
    def get_workflow_type(self) -> str:
        return "dynamic_code_execution"
        
    def plan(self, questions: List[str], file_manifest: Dict[str, Any], 
             keywords: List[str] = None, urls: List[str] = None) -> Dict[str, Any]:
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
        
        keywords = keywords or []
        urls = urls or []

        if not isinstance(file_manifest, dict):
            if isinstance(file_manifest, list):
                logger.warning(f"[DynamicCodeExecutionWorkflow.plan] file_manifest is a list, converting to dict: {file_manifest}")
                file_manifest = {str(i): v for i, v in enumerate(file_manifest)}
            else:
                logger.error(f"[DynamicCodeExecutionWorkflow.plan] file_manifest is not a dict, got {type(file_manifest)}: {file_manifest}")
                raise TypeError(f"[DynamicCodeExecutionWorkflow.plan] file_manifest must be a dict, got {type(file_manifest)}")
        
        plan = {
            "data_sources": [],
            "analysis_type": "general",
            "output_format": "json",
            "visualization_needed": False,
            "libraries_needed": ["pandas", "numpy"]
        }
        
        # Extract actual files from manifest structure
        actual_files = file_manifest.get('files', {}) if isinstance(file_manifest, dict) else file_manifest
        
        # Analyze file manifest - only iterate over actual files
        if isinstance(actual_files, dict):
            for filename, file_info in actual_files.items():
                # Ensure file_info is a dictionary
                if not isinstance(file_info, dict):
                    logger.warning(f"Skipping file {filename}: file_info is not a dict, got {type(file_info)}")
                    continue
                    
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
        
        # If no files found but CSV is mentioned in questions, add it as a data source with context
        if not plan["data_sources"]:
            questions_text = " ".join(questions) if isinstance(questions, list) else str(questions)
            if "edges.csv" in questions_text or ".csv" in questions_text:
                import re
                csv_matches = re.findall(r'`?([^`\s]+\.csv)`?', questions_text)
                for csv_file in csv_matches:
                    # Infer likely columns based on context
                    columns = []
                    if "edges" in csv_file.lower() or "edge" in questions_text.lower():
                        columns = ["source", "target"]  # Common edge list format
                    elif "node" in csv_file.lower() or "vertex" in questions_text.lower():
                        columns = ["id", "label"]  # Common node list format
                    
                    plan["data_sources"].append({
                        "type": "csv",
                        "filename": csv_file,
                        "columns": columns,
                        "is_virtual": True,  # Mark as mentioned but not provided
                        "needs_generation": True  # Flag for synthetic data generation
                    })
                    plan["analysis_type"] = "data_analysis"
                
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
        
        # Use expected JSON structure if available
        output_format = plan.get("output_format", "json")
        if hasattr(self, 'expected_json_structure') and self.expected_json_structure:
            if self.expected_json_structure.get('keys'):
                output_format = f"JSON object with keys: {self.expected_json_structure['keys']}"
                logger.info(f"📋 Dynamic code execution targeting JSON structure: {len(self.expected_json_structure['keys'])} keys")
        
        code_prompt = self._build_code_generation_prompt(questions, file_manifest, plan)
        
        try:
            code = self.code_generator.generate_code(
                task_description=" ".join(questions),
                manifest=file_manifest,
                workflow_type="dynamic_code_execution",
                output_format=output_format
            )
            logger.info("Generated code successfully")
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise
    
    def execute(self, sandbox_executor, task_description: str) -> Dict[str, Any]:
        """Execute the workflow: build plan, generate code, and run with retry/repair."""
        questions = [task_description]
        file_manifest = self.manifest
        
        # Ensure URLs are properly handled
        urls = file_manifest.get('urls', [])
        if not isinstance(urls, (list, tuple)):
            if isinstance(urls, str):
                urls = [urls]
            else:
                urls = []
                
        plan = self.plan(questions, file_manifest, keywords=[], urls=urls)

        code = self.generate_code(questions, file_manifest, plan)

        executor = sandbox_executor or self.sandbox_executor
        if executor is None:
            raise RuntimeError("No sandbox executor available for DynamicCodeExecutionWorkflow")

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Execution attempt {attempt + 1}/{self.max_retries + 1}")

                # Prefer execute_code if available for richer options
                if hasattr(executor, 'execute_code'):
                    # Ensure libraries_needed is a proper list
                    libraries_needed = plan.get('libraries_needed', [])
                    if not isinstance(libraries_needed, (list, tuple)):
                        if isinstance(libraries_needed, str):
                            libraries_needed = [libraries_needed]
                        else:
                            libraries_needed = []
                    
                    # Ensure file_manifest has the right structure for sandbox
                    sandbox_files = {}
                    if isinstance(file_manifest, dict):
                        # Check if this is a full manifest with 'files' key
                        if 'files' in file_manifest:
                            files_dict = file_manifest['files']
                            for filename, file_info in files_dict.items():
                                if isinstance(file_info, dict) and "path" in file_info:
                                    sandbox_files[filename] = file_info
                        else:
                            # Assume file_manifest is already just the files dict
                            for filename, file_info in file_manifest.items():
                                if isinstance(file_info, dict) and "path" in file_info:
                                    sandbox_files[filename] = file_info
                    
                    result = executor.execute_code(
                        code=code,
                        files=sandbox_files,
                        timeout=self.execution_timeout,
                        allowed_libraries=libraries_needed
                    )
                else:
                    result = executor.execute_simple(code)

                if result.get('success'):
                    logger.info(f"Code executed successfully on attempt {attempt + 1}")
                    enhanced_result = {
                        "success": True,
                        "result": result,
                        "workflow": self.get_workflow_type(),
                        "retry_count": result.get("retry_count", attempt),
                        "was_fixed_by_llm": result.get("retry_count", attempt) > 0,
                        "original_code": code if attempt == 0 else None,
                        "final_code": result.get("fixed_code", code)
                    }
                    
                    if result.get("retry_count", 0) > 0 or attempt > 0:
                        logger.info(f"Code was successfully executed after {max(result.get('retry_count', 0), attempt)} attempts")
                    
                    return enhanced_result
                else:
                    logger.warning(f"Execution failed: {result.get('error', 'Unknown error')}")
                    if attempt < self.max_retries:
                        code = self.repair(code, result, plan)
                    else:
                        return {
                            "success": False,
                            "error": result.get('error', 'Execution failed'),
                            "workflow": self.get_workflow_type(),
                            "result": result
                        }

            except Exception as e:
                logger.error(f"Execution attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    return {
                        "success": False,
                        "error": str(e),
                        "workflow": self.get_workflow_type()
                    }

        return {"success": False, "error": "Max retries exceeded", "workflow": self.get_workflow_type()}
    
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
                task_description=repair_prompt,
                manifest=self.manifest,
                workflow_type="dynamic_code_execution",
                output_format=plan.get("output_format", "json")
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
            # Ensure file_info is a dictionary
            if not isinstance(file_info, dict):
                files_info.append(f"- {filename}: unknown file (invalid format)")
                continue
                
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