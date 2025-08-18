from typing import Dict, Any
import logging
from .classifier import WorkflowClassifier
from .file_processor import FileProcessor
from .code_generator import CodeGenerator
from workflows.init import get_workflow
from .sandbox_executor import SandboxExecutor
from config import settings

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self, workflows=None):
        # Initialize components
        self.file_processor = FileProcessor()
        
        try:
            self.classifier = WorkflowClassifier(settings.openai_api_key)
            self.code_generator = CodeGenerator(settings.openai_api_key)
            self.llm_available = True
            logger.info("LLM components initialized successfully")
        except Exception as e:
            logger.warning(f"LLM initialization failed: {e}. Falling back to deterministic mode.")
            self.classifier = None
            self.code_generator = None
            self.llm_available = False
        
        self.sandbox = SandboxExecutor(
            settings.sandbox_memory_limit,
            settings.sandbox_cpu_limit,
            settings.max_execution_time
        )
        
        # Store provided workflows or initialize empty dict
        self.workflows = workflows or {}
    
    def process_request(self, questions: str, files: Dict[str, Any]) -> Dict[str, Any]:
        """Main request processing pipeline - returns clean answer"""
        try:
            # Generate a request ID using timestamp and first few chars of question
            import uuid
            request_id = str(uuid.uuid4())
            
            # Step 1: Create file manifest
            manifest = self.file_processor.create_manifest(files, questions)
            logger.info(f"Created manifest for {len(files)} files")
            
            # Step 2: Classify workflow
            if self.llm_available:
                classification = self.classifier.classify(manifest)
            else:
                classification = self._fallback_classification(manifest)
            
            logger.info(f"Workflow classification: {classification['workflow']} (confidence: {classification.get('confidence', 0)})")
            
            # Step 3: Execute workflow
            workflow_result = self._execute_workflow(classification, manifest, questions)
            
            # Step 4: Extract the final answer from the workflow result
            final_answer = self._extract_final_answer(workflow_result, questions)
            
            # Return clean response with just the answer
            if final_answer is not None:
                return final_answer
            else:
                # Fallback: return minimal response structure
                return {
                    "success": True,
                    "result": workflow_result.get("result", {}).get("output", "No output generated")
                }
            
        except Exception as e:
            logger.error(f"Request processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_final_answer(self, workflow_result: Dict[str, Any], questions: str) -> Any:
        """
        Extract just the final answer from workflow result based on what the question asks for.
        """
        try:
            # Get the actual execution result
            execution_result = workflow_result.get("result", {})
            if not execution_result.get("success", False):
                return {
                    "success": False, 
                    "error": execution_result.get("error", "Execution failed")
                }
            
            # Try to get the parsed output (final_result from the executed code)
            output = execution_result.get("output")
            if output is not None:
                # If the output is already in the expected format, return it directly
                if isinstance(output, (list, dict, str, int, float)):
                    return output
            
            # Look for specific answer patterns in stdout
            stdout = execution_result.get("stdout", "")
            
            # Check for base64 image data in the output (for chart/plot requests)
            if "data:image" in stdout:
                import re
                # Extract base64 image data
                base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                matches = re.findall(base64_pattern, stdout)
                if matches:
                    return f"data:image/png;base64,{matches[0]}"
            
            # Look for JSON array patterns in stdout
            import json
            import re
            
            # Try to find final_result assignments with JSON arrays
            final_result_pattern = r'final_result\s*=\s*(\[.*?\])'
            matches = re.findall(final_result_pattern, stdout, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    continue
            
            # Try to find any JSON arrays in stdout
            json_array_pattern = r'\[(?:[^[\]]*(?:\[[^\]]*\])*)*[^[\]]*\]'
            json_matches = re.findall(json_array_pattern, stdout)
            
            for match in json_matches:
                try:
                    # Skip simple arrays that look like coordinates or basic lists
                    if match.count(',') > 0 and ('"' in match or any(char.isalpha() for char in match)):
                        parsed = json.loads(match)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return parsed
                except:
                    continue
            
            # Look for the final answer printed at the end
            lines = stdout.strip().split('\n')
            for i in range(len(lines) - 1, -1, -1):  # Search from the end
                line = lines[i].strip()
                if not line:
                    continue
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, (list, dict)) and parsed:
                        return parsed
                except:
                    pass
                
                # Look for specific patterns indicating final results
                if any(keyword in line.lower() for keyword in ['final result', 'answer', 'result:']):
                    # Try to extract JSON from this line
                    json_start = line.find('[')
                    if json_start >= 0:
                        json_end = line.rfind(']') + 1
                        if json_end > json_start:
                            try:
                                result_json = line[json_start:json_end]
                                return json.loads(result_json)
                            except:
                                pass
                    
                    # Try to extract from after colon
                    if ':' in line:
                        after_colon = line.split(':', 1)[1].strip()
                        try:
                            return json.loads(after_colon)
                        except:
                            pass
            
            # If we have artifacts (like generated files), check for specific answer files
            artifacts = execution_result.get("artifacts", {})
            if artifacts:
                # If there are plots, include them in response for visualization tasks
                plots = artifacts.get("plots", [])
                if plots and any(word in questions.lower() for word in ["plot", "chart", "graph", "visualize", "draw"]):
                    plot_info = plots[0] if plots else None
                    if plot_info:
                        # For plot requests, return a base64 placeholder or file reference
                        return f"data:image/png;base64,{plot_info.get('filename', 'chart.png')}"
            
            # If nothing specific found, return the raw output or the cleanest part of stdout
            if output:
                return output
            elif stdout.strip():
                # Return the last meaningful line of output that looks like an answer
                lines = [line.strip() for line in stdout.split('\n') if line.strip()]
                if lines:
                    # Look for the most answer-like line
                    for line in reversed(lines):
                        if any(char.isalnum() for char in line) and len(line) < 200:
                            try:
                                # Try to parse as JSON first
                                return json.loads(line)
                            except:
                                # Return as string if not JSON
                                return line
                    return lines[-1]
            
            return "No answer found in output"
            
        except Exception as e:
            logger.error(f"Error extracting final answer: {e}")
            return {
                "success": False,
                "error": f"Error extracting answer: {str(e)}"
            }
    
    def _execute_workflow(self, classification: Dict[str, Any], manifest: Dict[str, Any], questions: str) -> Dict[str, Any]:
        """Execute the appropriate workflow"""
        import traceback
        try:
            workflow_type = classification['workflow']
            logger.info(f"[orchestrator] Executing workflow: {workflow_type}")
            logger.info(f"[orchestrator] Manifest type: {type(manifest)}; Manifest value: {manifest}")
            if not isinstance(manifest, dict):
                logger.error(f"[orchestrator] Manifest is not a dict, got {type(manifest)}: {manifest}")
                raise TypeError(f"[orchestrator] Manifest must be a dict, got {type(manifest)}")

            # Only execute the selected workflow
            task_description = " ".join(questions) if isinstance(questions, list) else str(questions)
            
            if workflow_type in self.workflows:
                workflow = self.workflows[workflow_type]
                result = workflow.execute(self.sandbox, task_description)
            elif self.llm_available:
                workflow = get_workflow(
                    workflow_type,
                    self.code_generator,
                    manifest,
                    sandbox_executor=self.sandbox,
                    llm_client=None  # Will use default LLMClient in workflows
                )
                result = workflow.execute(self.sandbox, task_description)
            else:
                result = self._execute_basic_analysis(manifest, questions)

            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": f"Workflow execution failed: {str(e)}"
            }
    
    def _fallback_classification(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback classification when LLM is not available"""
        file_types = manifest.get('file_types', [])
        
        if 'csv' in file_types:
            return {
                "workflow": "data_analysis",
                "confidence": 0.8,
                "reasoning": "CSV file detected, using data analysis workflow",
                "method": "fallback"
            }
        else:
            return {
                "workflow": "dynamic",
                "confidence": 0.6, 
                "reasoning": "No clear file type, using dynamic workflow",
                "method": "fallback"
            }
    
    def _execute_basic_analysis(self, manifest: Dict[str, Any], questions: str) -> Dict[str, Any]:
        """Basic analysis when LLM is not available"""
        # Generate simple analysis code
        csv_files = [name for name, info in manifest.get('files', {}).items() if info.get('type') == 'csv']
        
        if csv_files:
            code = f'''
import pandas as pd
import json

# Load the first CSV file
df = pd.read_csv("{csv_files[0]}")
print("Data loaded successfully!")
print(f"Shape: {{df.shape}}")
print("\\nColumns:", df.columns.tolist())
print("\\nFirst few rows:")
print(df.head())

print("\\nBasic statistics:")
print(df.describe())

# Try to answer the question with basic operations
final_result = {{
    "message": "Basic CSV analysis completed (LLM not available)",
    "question": "{questions}",
    "file_analyzed": "{csv_files[0]}",
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else "No numeric columns"
}}

print("\\nFinal result:")
print(json.dumps(final_result, indent=2, default=str))
'''
        else:
            code = f'''
print("Question: {questions}")
print("No CSV files available for analysis")

final_result = {{
    "message": "No data files available for analysis",
    "question": "{questions}",
    "available_files": {list(manifest.get('files', {}).keys())}
}}

print("Final result:", final_result)
'''
        
        return self.sandbox.execute_simple(code)