from .base import BaseWorkflow
from .data_analysis import DataAnalysisWorkflow
from .web_scraping import WebScrapingWorkflow
from .dynamic_code_execution import DynamicCodeExecutionWorkflow
from .image_analysis import ImageAnalysisWorkflow

# Workflow registry
WORKFLOW_REGISTRY = {
    'data_analysis': DataAnalysisWorkflow,
    'dynamic': DataAnalysisWorkflow,
    'web_scraping': WebScrapingWorkflow,
    'dynamic_code_execution': DynamicCodeExecutionWorkflow,
    'image_analysis': ImageAnalysisWorkflow,
 
}

def get_workflow_class(workflow_name: str):
    """Get workflow class by name"""
    return WORKFLOW_REGISTRY.get(workflow_name, DataAnalysisWorkflow)

def get_workflow(workflow_name: str, code_generator, manifest, sandbox_executor=None, llm_client=None):
    """Create workflow instance with proper parameters"""
    if code_generator is None:
        raise ValueError(f"code_generator cannot be None for workflow: {workflow_name}")
    if manifest is None:
        raise ValueError(f"manifest cannot be None for workflow: {workflow_name}")
    
    workflow_class = get_workflow_class(workflow_name)
    
    # All workflows now have consistent constructor signatures
    return workflow_class(code_generator, manifest, sandbox_executor=sandbox_executor, llm_client=llm_client)