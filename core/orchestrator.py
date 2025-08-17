from typing import Dict, Any
import logging
from .classifier import WorkflowClassifier
from .file_processor import FileProcessor
from .code_generator import CodeGenerator
from workflows.init import get_workflow
from sandbox.executor import SandboxExecutor
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
        """Main request processing pipeline"""
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
            result = self._execute_workflow(classification, manifest, questions)
            
            # Step 4: Format response
            return {
                "success": True,
                "request_id": request_id,
                "workflow_used": classification['workflow'],
                "confidence": classification.get('confidence', 0),
                "reasoning": classification.get('reasoning', 'No reasoning provided'),
                "llm_available": self.llm_available,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Request processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "llm_available": self.llm_available
            }
    
    def _execute_workflow(self, classification: Dict[str, Any], manifest: Dict[str, Any], questions: str) -> Dict[str, Any]:
        """Execute the appropriate workflow"""
        try:
            workflow_type = classification['workflow']
            
            if workflow_type in self.workflows:
                # Use provided workflow instance
                workflow = self.workflows[workflow_type]
                result = workflow.execute(self.sandbox, questions)
            elif self.llm_available:
                # Fallback to creating new workflow if not provided
                workflow = get_workflow(
                    workflow_type, 
                    self.code_generator, 
                    manifest
                )
                result = workflow.execute(self.sandbox, questions)
            else:
                # Use basic deterministic approach
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