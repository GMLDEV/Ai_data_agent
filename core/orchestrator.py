from typing import Dict, Any, Optional
import logging
from .classifier import WorkflowClassifier
from .file_processor import FileProcessor
from .code_generator import CodeGenerator
from .question_parser import QuestionParser
from workflows.init import get_workflow
from .sandbox_executor import SandboxExecutor

logger = logging.getLogger(__name__)

class Orchestrator:
    """Main orchestrator that coordinates all AI data agent operations with comprehensive logging"""
    
    def __init__(self, llm_available: bool = True):
        self.llm_available = llm_available
        
        if llm_available:
            try:
                from .llm_client import LLMClient
                self.llm_client = LLMClient()
                logger.info("✅ LLM client initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LLM client: {e}")
                self.llm_client = None
                self.llm_available = False
        else:
            self.llm_client = None
            logger.info("ℹ️ Running in no-LLM mode")
        
        # Initialize components
        self.file_processor = FileProcessor()
        self.classifier = WorkflowClassifier(self.llm_client) if self.llm_client else None
        self.code_generator = CodeGenerator(self.llm_client) if self.llm_client else None
        self.question_parser = QuestionParser()
        
        # Initialize sandbox executor with enhanced capabilities
        self.sandbox = SandboxExecutor(
            max_memory_mb=512,
            max_cpu_time_seconds=180
        )
        # Set LLM client reference for logging
        if hasattr(self.sandbox, 'llm_client'):
            self.sandbox.llm_client = self.llm_client
        
        # Initialize workflow registry
        self.workflow_registry = {}
        self._init_workflows()
        
        logger.info("🚀 Orchestrator initialized with comprehensive logging")

    def _init_workflows(self):
        """Initialize all available workflows"""
        try:
            from workflows.data_analysis import DataAnalysisWorkflow
            from workflows.web_scraping import WebScrapingWorkflow
            from workflows.image_analysis import ImageAnalysisWorkflow
            from workflows.dynamic_code_execution import DynamicCodeExecutionWorkflow
            
            # Create a dummy manifest for workflow initialization
            dummy_manifest = {'files': {}, 'urls': [], 'keywords': [], 'file_types': []}
            
            self.workflow_registry = {
                'data_analysis': DataAnalysisWorkflow(
                    code_generator=self.code_generator, 
                    manifest=dummy_manifest,
                    sandbox_executor=self.sandbox, 
                    llm_client=self.llm_client
                ),
                'web_scraping': WebScrapingWorkflow(
                    code_generator=self.code_generator, 
                    manifest=dummy_manifest,
                    sandbox_executor=self.sandbox, 
                    llm_client=self.llm_client
                ),
                'image_analysis': ImageAnalysisWorkflow(
                    code_generator=self.code_generator, 
                    manifest=dummy_manifest,
                    sandbox_executor=self.sandbox, 
                    llm_client=self.llm_client
                ),
                'dynamic_code_execution': DynamicCodeExecutionWorkflow(
                    code_generator=self.code_generator, 
                    manifest=dummy_manifest,
                    sandbox_executor=self.sandbox, 
                    llm_client=self.llm_client
                ),
                'general': DataAnalysisWorkflow(
                    code_generator=self.code_generator, 
                    manifest=dummy_manifest,
                    sandbox_executor=self.sandbox, 
                    llm_client=self.llm_client
                )  # Fallback
            }
            logger.info(f"📋 Initialized {len(self.workflow_registry)} workflows")
        except Exception as e:
            logger.error(f"❌ Failed to initialize workflows: {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            # Fallback: create an empty registry and handle this in execution
            self.workflow_registry = {}

    def process_request(self, questions: str, files: Dict[str, Any]) -> Dict[str, Any]:
        """Main request processing pipeline - returns clean answer"""
        try:
            # Generate a request ID using timestamp and first few chars of question
            import uuid
            request_id = str(uuid.uuid4())
            
            # Log detailed request start
            logger.info(f"🚀 REQUEST START [ID: {request_id}]")
            logger.info(f"📝 Questions: {questions[:200]}{'...' if len(questions) > 200 else ''}")
            logger.info(f"📁 Files provided: {len(files)} files")
            if files:
                for filename in files.keys():
                    logger.info(f"   - {filename}")
            
            # Step 1: Create file manifest
            manifest = self.file_processor.create_manifest(files, questions)
            logger.info(f"📋 Created manifest for {len(files)} files")
            logger.debug(f"📋 Full manifest: {manifest}")
            
            # Step 2: Classify workflow
            if self.llm_available:
                classification = self.classifier.classify(manifest)
            else:
                classification = self._fallback_classification(manifest)
            
            logger.info(f"🎯 Workflow classification: {classification['workflow']} (confidence: {classification.get('confidence', 0):.2f})")
            logger.info(f"💭 Classification reasoning: {classification.get('reasoning', 'No reasoning provided')}")
            
            # Step 3: Execute workflow with detailed logging
            logger.info(f"⚙️ Starting workflow execution: {classification['workflow']}")
            workflow_result = self._execute_workflow_with_logging(classification, manifest, questions, request_id)
            
            # Step 4: Extract the final answer from the workflow result
            final_answer = self._extract_final_answer(workflow_result, questions)
            
            # Log the complete workflow result for debugging
            self._log_complete_workflow_details(request_id, workflow_result, final_answer)
            
            # Return clean response with just the answer
            logger.info(f"✅ REQUEST COMPLETE [ID: {request_id}] - Clean response returned")
            if final_answer is not None:
                return final_answer
            else:
                # Fallback: return minimal response structure
                logger.warning(f"⚠️ Fallback response used [ID: {request_id}]")
                return {
                    "success": True,
                    "result": workflow_result.get("result", {}).get("output", "No output generated")
                }
            
        except Exception as e:
            logger.error(f"❌ REQUEST FAILED [ID: {request_id if 'request_id' in locals() else 'unknown'}]: {str(e)}")
            import traceback
            logger.error(f"🔍 Full traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }

    def _execute_workflow_with_logging(self, classification: Dict[str, str], manifest: Dict[str, Any], questions: str, request_id: str) -> Dict[str, Any]:
        """Execute workflow with comprehensive logging"""
        workflow_name = classification['workflow']
        logger.info(f"⚙️ Workflow '{workflow_name}' started [ID: {request_id}]")
        
        try:
            # Execute the workflow based on classification
            workflow = self.workflow_registry.get(workflow_name)
            if workflow is None:
                raise ValueError(f"Workflow '{workflow_name}' not found in registry")
            
            # Parse expected JSON structure from questions
            expected_structure = None
            if hasattr(self, 'question_parser'):
                try:
                    expected_structure = self.question_parser.parse_json_structure(questions)
                    logger.info(f"📋 Parsed expected JSON structure: {expected_structure}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse JSON structure: {e}")
            
            # Log workflow execution details
            logger.info(f"📊 Executing workflow: {workflow.__class__.__name__}")
            logger.debug(f"📊 Manifest data: {manifest}")
            
            # Update workflow manifest and pass JSON structure if available
            workflow.manifest = manifest
            if expected_structure and hasattr(workflow, 'set_expected_structure'):
                workflow.set_expected_structure(expected_structure)
            
            result = workflow.execute(self.sandbox, questions)
            
            # Log workflow completion
            if result.get("success", False):
                logger.info(f"✅ Workflow '{workflow_name}' completed successfully [ID: {request_id}]")
                if "code_attempts" in result:
                    logger.info(f"🔄 Code generation attempts: {len(result.get('code_attempts', []))}")
                if "retry_history" in result:
                    logger.info(f"🔄 Retry attempts: {len(result.get('retry_history', []))}")
            else:
                logger.warning(f"⚠️ Workflow '{workflow_name}' completed with issues [ID: {request_id}]")
                if "error" in result:
                    logger.error(f"❌ Workflow error: {result['error']}")
                    
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow '{workflow_name}' failed [ID: {request_id}]: {str(e)}")
            import traceback
            logger.error(f"🔍 Workflow traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "workflow": workflow_name
            }

    def _log_complete_workflow_details(self, request_id: str, workflow_result: Dict[str, Any], final_answer: Any) -> None:
        """Log complete workflow details for debugging"""
        logger.debug(f"📊 COMPLETE WORKFLOW DETAILS [ID: {request_id}]")
        logger.debug(f"📊 Raw workflow result keys: {list(workflow_result.keys())}")
        
        # Log code generation attempts
        if "code_attempts" in workflow_result:
            attempts = workflow_result["code_attempts"]
            logger.debug(f"🔄 Code Generation Attempts ({len(attempts)}):")
            for i, attempt in enumerate(attempts, 1):
                logger.debug(f"   Attempt {i}: {attempt[:100]}...")
        
        # Log retry history
        if "retry_history" in workflow_result:
            retries = workflow_result["retry_history"]
            logger.debug(f"🔄 Retry History ({len(retries)}):")
            for i, retry in enumerate(retries, 1):
                logger.debug(f"   Retry {i}: {retry.get('reason', 'No reason')} - {retry.get('status', 'Unknown status')}")
        
        # Log execution details
        if "execution_details" in workflow_result:
            details = workflow_result["execution_details"]
            logger.debug(f"⚙️ Execution Details: {details}")
        
        # Log any errors
        if "errors" in workflow_result:
            errors = workflow_result["errors"]
            logger.debug(f"❌ Errors encountered: {errors}")
        
        # Log the final extracted answer
        logger.debug(f"📤 Final extracted answer type: {type(final_answer)}")
        if isinstance(final_answer, (dict, list)):
            logger.debug(f"📤 Final answer structure: {str(final_answer)[:200]}...")
        else:
            logger.debug(f"📤 Final answer: {str(final_answer)[:200]}...")
        
        # Log full result for debugging if needed
        logger.debug(f"🔍 Full workflow result: {workflow_result}")

    def _extract_final_answer(self, workflow_result: Dict[str, Any], questions: str) -> Any:
        """Extract clean final answer from workflow result that matches the question format"""
        try:
            logger.debug(f"🔍 Extracting final answer for question: {questions[:100]}...")
            
            # Get the execution result
            if not workflow_result.get("success", False):
                logger.warning("⚠️ Workflow was not successful")
                return {"error": workflow_result.get("error", "Workflow execution failed")}
            
            result = workflow_result.get("result", {})
            if not result.get("success", False):
                logger.warning("⚠️ Code execution was not successful")
                return {"error": result.get("error", "Code execution failed")}
            
            # Get stdout content for parsing
            stdout = result.get("stdout", "")
            logger.debug(f"📜 Raw stdout length: {len(stdout)} chars")
            
            # Parse the questions to understand expected JSON structure
            json_structure = self.question_parser.parse_json_structure(questions)
            expects_json_object = json_structure is not None
            
            if json_structure:
                logger.info(f"📋 Expecting JSON object with {json_structure['total_keys']} keys: {json_structure['keys']}")
            else:
                # Fallback to old detection method
                expects_json_object = any(key_word in questions.lower() for key_word in [
                    'json object', 'return a json', 'json with keys', 'object with keys'
                ])
            
            # Look for structured data patterns in stdout
            import json
            import re
            
            # Strategy 1: Look for JSON object patterns
            if expects_json_object:
                json_object_patterns = [
                    r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # JSON object pattern
                    r'final_result\s*=\s*(\{.*?\})',
                    r'result\s*=\s*(\{.*?\})',
                    r'answer\s*=\s*(\{.*?\})',
                    r'output\s*=\s*(\{.*?\})'
                ]
                
                for pattern in json_object_patterns:
                    matches = re.findall(pattern, stdout, re.DOTALL)
                    for match in matches:
                        # Skip processing messages
                        if any(skip_text in match.lower() for skip_text in [
                            'processing', 'package', 'install', 'records'
                        ]):
                            continue
                            
                        try:
                            # First try ast.literal_eval for Python dict notation (single quotes)
                            import ast
                            parsed = ast.literal_eval(match)
                            if isinstance(parsed, dict) and len(parsed) > 0:
                                # Validate against expected structure if available
                                if json_structure:
                                    validated = self._validate_json_structure(parsed, json_structure)
                                    if validated:
                                        logger.info(f"✅ Found Python dict object answer: {len(parsed)} keys (validated)")
                                        return validated
                                logger.info(f"✅ Found Python dict object answer: {len(parsed)} keys")
                                return parsed
                        except (ValueError, SyntaxError):
                            try:
                                # Then try JSON parsing for standard JSON
                                parsed = json.loads(match)
                                if isinstance(parsed, dict) and len(parsed) > 0:
                                    # Validate against expected structure if available
                                    if json_structure:
                                        validated = self._validate_json_structure(parsed, json_structure)
                                        if validated:
                                            logger.info(f"✅ Found JSON object answer: {len(parsed)} keys (validated)")
                                            return validated
                                    logger.info(f"✅ Found JSON object answer: {len(parsed)} keys")
                                    return parsed
                            except json.JSONDecodeError:
                                continue
            
            # Strategy 2: Look for list patterns (original functionality)
            final_result_patterns = [
                r'final_result\s*=\s*(\[.*?\])',
                r'result\s*=\s*(\[.*?\])', 
                r'answer\s*=\s*(\[.*?\])',
                r'output\s*=\s*(\[.*?\])'
            ]
            
            for pattern in final_result_patterns:
                matches = re.findall(pattern, stdout, re.DOTALL)
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            logger.info(f"✅ Found structured answer via variable assignment: {type(parsed)} with {len(parsed)} items")
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Strategy 3: Look for JSON objects or arrays printed at the end
            lines = [line.strip() for line in stdout.split('\n') if line.strip()]
            
            # Search from the last line backwards for the first valid JSON structure
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                
                # Skip processing messages and built-in module messages
                if any(skip_text in line.lower() for skip_text in [
                    'processing', 'records', 'skipping built-in', 'missing package',
                    'successfully installed', 'failed to install', 'attempting to install'
                ]):
                    continue
                
                # Try to parse the line as JSON object first (if expected)
                if expects_json_object and line.startswith('{') and (line.endswith('}') or '...' in line):
                    # Handle potentially truncated lines
                    clean_line = line.rstrip('...')
                    
                    try:
                        # First try ast.literal_eval for Python dict notation
                        import ast
                        parsed = ast.literal_eval(clean_line)
                        if isinstance(parsed, dict) and len(parsed) > 0:
                            # Validate against expected structure if available
                            if json_structure:
                                validated = self._validate_json_structure(parsed, json_structure)
                                if validated:
                                    logger.info(f"✅ Found Python dict answer at line {i}: {len(parsed)} keys (validated)")
                                    return validated
                            logger.info(f"✅ Found Python dict answer at line {i}: {len(parsed)} keys")
                            return parsed
                    except (ValueError, SyntaxError):
                        try:
                            # Then try JSON parsing
                            parsed = json.loads(clean_line)
                            if isinstance(parsed, dict) and len(parsed) > 0:
                                logger.info(f"✅ Found JSON object answer at line {i}: {len(parsed)} keys")
                                return parsed
                        except json.JSONDecodeError:
                            continue
                
                # Try to parse the line as JSON array
                if line.startswith('[') and line.endswith(']'):
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            logger.info(f"✅ Found structured answer at line {i}: {type(parsed)} with {len(parsed)} items")
                            return parsed
                    except json.JSONDecodeError:
                        continue
                
                # Look for patterns like "Final result: {...}" or "Answer: {...}"
                answer_patterns = [
                    r'(?:final result|answer|result):\s*(\{.*\})',  # JSON object
                    r'(?:final result|answer|result):\s*(\[.*\])',  # JSON array
                    r'(?:final result|answer|result):\s*(.+)$'     # Any other format
                ]
                
                for pattern in answer_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            answer_str = match.group(1).strip()
                            parsed = json.loads(answer_str)
                            if isinstance(parsed, (list, dict)) and parsed:
                                logger.info(f"✅ Found answer via pattern matching: {type(parsed)}")
                                return parsed
                        except json.JSONDecodeError:
                            # If not JSON, return the string value
                            return answer_str
            
            # Strategy 4: Look for any JSON structures in the output
            if expects_json_object:
                # Look for JSON objects
                json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_obj_matches = re.findall(json_object_pattern, stdout)
                
                for match in reversed(json_obj_matches):  # Start from the last match
                    # Skip if it looks like processing info
                    if any(skip_text in match.lower() for skip_text in [
                        'records', 'processing', 'package', 'install'
                    ]):
                        continue
                    
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, dict) and len(parsed) > 0:
                            logger.info(f"✅ Found JSON object via pattern search: {len(parsed)} keys")
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Also look for JSON arrays (original functionality)
            json_array_pattern = r'\[(?:[^\[\]]*(?:\[[^\]]*\])*)*[^\[\]]*\]'
            json_matches = re.findall(json_array_pattern, stdout)
            
            # Filter and validate JSON matches
            for match in reversed(json_matches):  # Start from the last match
                # Skip if it looks like processing info
                if any(skip_text in match.lower() for skip_text in [
                    'records', 'processing', 'package', 'install'
                ]):
                    continue
                
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Additional validation: check if it looks like an answer
                        if (len(parsed) >= 2 and 
                            not all(isinstance(x, str) and any(word in x.lower() for word in ['package', 'install', 'module']) for x in parsed)):
                            logger.info(f"✅ Found structured answer via JSON search: {type(parsed)} with {len(parsed)} items")
                            return parsed
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Handle simple outputs that might be direct answers
            if result.get("output"):
                output = result["output"]
                if isinstance(output, (list, dict, int, float)):
                    logger.info(f"✅ Using direct output: {type(output)}")
                    return output
                elif isinstance(output, str):
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(output)
                        logger.info(f"✅ Parsed output string as JSON: {type(parsed)}")
                        return parsed
                    except json.JSONDecodeError:
                        # Return the string if it's not processing noise
                        if not any(noise in output.lower() for noise in ['processing', 'package', 'install', 'module']):
                            logger.info(f"✅ Using string output: {output[:50]}...")
                            return output
            
            # Strategy 5: Last resort - look for the cleanest line in stdout
            clean_lines = []
            for line in reversed(lines):
                if (line and 
                    not line.lower().startswith(('processing', 'skipping', 'missing', 'successfully', 'failed')) and
                    not 'package' in line.lower() and
                    not 'install' in line.lower() and
                    len(line) < 500):  # Avoid very long lines
                    clean_lines.append(line)
            
            if clean_lines:
                # Try to parse the cleanest line
                for line in clean_lines[:3]:  # Check up to 3 clean lines
                    try:
                        parsed = json.loads(line)
                        logger.info(f"✅ Found answer in clean line: {type(parsed)}")
                        return parsed
                    except json.JSONDecodeError:
                        pass
                
                # Return the cleanest line as string if nothing else works
                logger.info(f"✅ Using cleanest line: {clean_lines[0][:50]}...")
                return clean_lines[0]
            
            # No structured answer found
            logger.warning("⚠️ Could not extract structured answer from output")
            return {"error": "Could not parse structured answer from execution output"}
            
        except Exception as e:
            logger.error(f"❌ Error extracting final answer: {e}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return {"error": f"Error extracting answer: {str(e)}"}

    def _fallback_classification(self, manifest: Dict[str, Any]) -> Dict[str, str]:
        """Fallback classification when LLM is not available"""
        # Simple rule-based classification
        files = manifest.get('files', {})
        if isinstance(files, dict):
            # Check if any files are CSV type
            if any(file_info.get('type') == 'csv' for file_info in files.values() if isinstance(file_info, dict)):
                return {
                    'workflow': 'data_analysis',
                    'confidence': 0.8,
                    'reasoning': 'CSV files detected - using data analysis workflow'
                }
        
        # Check for web-related keywords in the user request
        user_request = str(manifest.get('questions', ''))
        if any(keyword in user_request.lower() for keyword in ['url', 'scrape', 'website', 'web']):
            return {
                'workflow': 'web_scraping',
                'confidence': 0.7,
                'reasoning': 'Web-related keywords detected - using web scraping workflow'
            }
        else:
            return {
                'workflow': 'general',
                'confidence': 0.5,
                'reasoning': 'No specific patterns detected - using general workflow'
            }
    
    def _validate_json_structure(self, parsed_json: Dict[str, Any], expected_structure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate that the parsed JSON has the expected structure and keys.
        
        Args:
            parsed_json: The parsed JSON object
            expected_structure: Structure info from question parser
            
        Returns:
            Validated JSON object or None if validation fails
        """
        expected_keys = expected_structure.get('keys', [])
        key_types = expected_structure.get('types', {})
        
        # Check if all expected keys are present
        missing_keys = [key for key in expected_keys if key not in parsed_json]
        
        if missing_keys:
            logger.warning(f"⚠️ JSON validation failed - missing keys: {missing_keys}")
            # Don't fail completely, just log the warning
        
        # Validate key types if possible
        for key, value in parsed_json.items():
            if key in key_types:
                expected_type = key_types[key]
                
                if expected_type == 'number' and not isinstance(value, (int, float)):
                    logger.warning(f"⚠️ Key '{key}' expected number, got {type(value).__name__}")
                elif expected_type == 'string' and not isinstance(value, str):
                    logger.warning(f"⚠️ Key '{key}' expected string, got {type(value).__name__}")
                elif expected_type == 'base64_image' and not isinstance(value, str):
                    logger.warning(f"⚠️ Key '{key}' expected base64 image string, got {type(value).__name__}")
                elif expected_type == 'base64_image' and isinstance(value, str):
                    if not (value.startswith('data:image/') or 'base64' in value.lower()):
                        logger.warning(f"⚠️ Key '{key}' may not be valid base64 image data")
        
        logger.info(f"✅ JSON structure validated: {len(parsed_json)} keys present")
        return parsed_json
    
    def set_llm_client(self, llm_client):
        """Set or update the LLM client"""
        self.llm_client = llm_client
        self.llm_available = llm_client is not None
        if self.classifier:
            self.classifier.llm_client = llm_client
        if self.code_generator:
            self.code_generator.llm_client = llm_client
        
        # Update sandbox executor
        if hasattr(self.sandbox, 'llm_client'):
            self.sandbox.llm_client = llm_client
        logger.info("✅ LLM client updated in orchestrator and all components")