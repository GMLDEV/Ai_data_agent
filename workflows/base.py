from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

class BaseWorkflow(ABC):
    def __init__(self, code_generator, manifest: Dict[str, Any]):
        self.code_generator = code_generator
        self.manifest = manifest
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = 3
    
    @abstractmethod
    def get_workflow_type(self) -> str:
        """Return the workflow type identifier"""
        pass
    
    def plan(self) -> List[str]:
        """Generate execution plan steps"""
        return [
            "Analyze user request and available files",
            "Generate appropriate Python code", 
            "Execute code in sandbox environment",
            "Validate and return results"
        ]
    
    def generate_code(self, task_description: str, output_format: str = "json") -> str:
        """Generate Python code for the task"""
        return self.code_generator.generate_code(
            task_description=task_description,
            manifest=self.manifest,
            workflow_type=self.get_workflow_type(),
            output_format=output_format
        )
    
    def execute(self, sandbox_executor, task_description: str) -> Dict[str, Any]:
        """Execute the workflow with retry logic"""
        plan = self.plan()
        self.logger.info(f"Execution plan: {plan}")
        
        # Generate initial code
        code = self.generate_code(task_description)
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Execution attempt {attempt + 1}")
                result = sandbox_executor.execute_simple(code)
                
                if result.get('success'):
                    self.logger.info("Execution successful")
                    return {
                        "success": True,
                        "result": result,
                        "workflow": self.get_workflow_type(),
                        "attempts": attempt + 1,
                        "code_used": code
                    }
                
                # Attempt repair if failed and we have retries left
                if attempt < self.max_retries - 1:
                    self.logger.info("Attempting code repair...")
                    error_msg = result.get('error', 'Unknown error')
                    code = self.code_generator.repair_code(code, error_msg, task_description)
                
            except Exception as e:
                self.logger.error(f"Execution attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "workflow": self.get_workflow_type(),
                        "attempts": attempt + 1
                    }
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "workflow": self.get_workflow_type(),
            "attempts": self.max_retries
        }