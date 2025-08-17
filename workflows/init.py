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

def get_workflow(workflow_name: str, code_generator, manifest):
    """Create workflow instance"""
    workflow_class = get_workflow_class(workflow_name)
    return workflow_class(code_generator, manifest)