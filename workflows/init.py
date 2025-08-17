from .base import BaseWorkflow
from .data_analysis import DataAnalysisWorkflow

# Workflow registry
WORKFLOW_REGISTRY = {
    'data_analysis': DataAnalysisWorkflow,
    'dynamic': DataAnalysisWorkflow,  # Use data analysis as fallback for now
    # More workflows will be added in future phases
}

def get_workflow_class(workflow_name: str):
    """Get workflow class by name"""
    return WORKFLOW_REGISTRY.get(workflow_name, DataAnalysisWorkflow)

def get_workflow(workflow_name: str, code_generator, manifest):
    """Create workflow instance"""
    workflow_class = get_workflow_class(workflow_name)
    return workflow_class(code_generator, manifest)