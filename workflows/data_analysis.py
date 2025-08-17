from typing import Dict, List, Any
import pandas as pd
import json
import logging
from workflows.base import BaseWorkflow
from workflows.dynamic_code_execution import DynamicCodeExecutionWorkflow
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class DataAnalysisWorkflow(BaseWorkflow):
    """
    Specialized workflow for CSV data analysis tasks.
    Inherits from DynamicCodeExecutionWorkflow for code generation and execution.
    """
    
    def __init__(self, code_generator, manifest, sandbox_executor=None, llm_client=None):
        super().__init__(code_generator=code_generator, manifest=manifest)
        if not isinstance(manifest, dict):
            logger.error(f"[DataAnalysisWorkflow.__init__] manifest is not a dict, got {type(manifest)}: {manifest}")
            raise TypeError(f"[DataAnalysisWorkflow.__init__] manifest must be a dict, got {type(manifest)}")
        self.dynamic_workflow = DynamicCodeExecutionWorkflow(code_generator=code_generator, manifest=manifest, sandbox_executor=sandbox_executor)
        self.sandbox_executor = sandbox_executor
        self.llm_client = llm_client or LLMClient()

    def get_workflow_type(self):
        return "data_analysis"

    def plan(self, questions: List[str], file_manifest: Dict[str, Any], 
             keywords: List[str], urls: List[str]) -> Dict[str, Any]:
        """
        Create a specialized plan for data analysis tasks.
        """
        logger.info("Planning data analysis workflow")
        
        # Start with dynamic workflow plan
        base_plan = self.dynamic_workflow.plan(questions, file_manifest, keywords, urls)
        
        # Enhance with data analysis specific details
        csv_files = [f for f, info in file_manifest.items() if info.get("type") == "csv"]
        
        if not csv_files:
            raise ValueError("No CSV files found for data analysis workflow")
        
        questions_text = " ".join(questions).lower()
        
        analysis_plan = {
            **base_plan,
            "analysis_type": "data_analysis",
            "csv_files": csv_files,
            "primary_csv": csv_files[0],  # Use first CSV as primary
            "operations": self._detect_operations(questions_text),
            "statistical_tests": self._detect_statistical_tests(questions_text),
            "visualization_types": self._detect_visualization_types(questions_text),
            "grouping_variables": self._extract_grouping_variables(questions_text, file_manifest),
            "target_variables": self._extract_target_variables(questions_text, file_manifest),
            "column_types": self._infer_column_types(csv_files[0]),
            "data_cleaning_steps": self._suggest_data_cleaning(csv_files[0])
        }
        
        # Add specialized libraries
        analysis_plan.setdefault("libraries_needed", []).extend([
            "scipy.stats", "statsmodels", "sklearn"
        ])
        
        logger.info(f"Enhanced data analysis plan: {analysis_plan}")
        return analysis_plan
    
    def generate_code(self, questions: List[str], file_manifest: Dict[str, Any], 
                     plan: Dict[str, Any]) -> str:
        """Generate specialized data analysis code."""
        data_analysis_prompt = self._build_data_analysis_prompt(questions, file_manifest, plan)
        try:
            code = self.code_generator.generate_code(
                prompt=data_analysis_prompt,
                context={
                    "questions": questions,
                    "file_manifest": file_manifest,
                    "plan": plan,
                    "workflow_type": "data_analysis"
                }
            )
            return code
        except Exception as e:
            logger.error(f"Data analysis code generation failed: {e}")
            raise

    def execute(self, sandbox_executor, task_description: str) -> Dict[str, Any]:
        """Execute the data analysis workflow using the provided sandbox executor.

        The method will build a plan from the workflow's manifest, generate code,
        and execute it in the provided executor (or the one stored on the workflow).
        """
        logger.info("Executing data analysis workflow")

        questions = [task_description]
        file_manifest = self.manifest

        # Build the plan and generate code
        plan = self.plan(questions, file_manifest, keywords=[], urls=file_manifest.get('urls', []))
        code = self.generate_code(questions, file_manifest, plan)

        executor = sandbox_executor or self.sandbox_executor
        if executor is None:
            raise RuntimeError("No sandbox executor available for DataAnalysisWorkflow")

        result = executor.execute_simple(code)

        return {
            "success": result.get("success", False),
            "result": result,
            "workflow": self.get_workflow_type()
        }

    def validate(self, result: Dict[str, Any], plan: Dict[str, Any]) -> bool:
        """Validate output format and constraints."""
        logger.info("Validating data analysis output")
        # Example: Check output size, required keys, and format
        if not result.get("success"):
            return False
        if len(result.get("stdout", "")) > 100_000:
            logger.warning("Output exceeds size limit")
            return False
        # Add more schema/format checks as needed
        return True

    def repair(self, code: str, error: str, plan: Dict[str, Any]) -> str:
        """Repair code using error feedback and LLM."""
        logger.info("Repairing code using error feedback")
        repair_prompt = (
            f"The following code failed with error:\n{error}\n"
            f"Original code:\n{code}\n"
            "Please fix the code for the data analysis task described in the plan."
        )
        repaired_code = self.code_generator.generate_code(
            prompt=repair_prompt,
            context={"plan": plan, "workflow_type": "data_analysis"}
        )
        return repaired_code

    # --- Helper methods for analysis plan ---
    def _detect_operations(self, questions_text: str) -> List[str]:
        # Example: Detect requested operations (mean, sum, correlation, etc.)
        ops = []
        if "correlation" in questions_text:
            ops.append("correlation")
        if "regression" in questions_text:
            ops.append("regression")
        if "mean" in questions_text:
            ops.append("mean")
        if "sum" in questions_text:
            ops.append("sum")
        if "group" in questions_text:
            ops.append("groupby")
        # Add more detection logic as needed
        return ops

    def _detect_statistical_tests(self, questions_text: str) -> List[str]:
        tests = []
        if "t-test" in questions_text or "ttest" in questions_text:
            tests.append("t-test")
        if "anova" in questions_text:
            tests.append("anova")
        if "chi-square" in questions_text or "chi2" in questions_text:
            tests.append("chi-square")
        # Add more detection logic as needed
        return tests

    def _detect_visualization_types(self, questions_text: str) -> List[str]:
        vis = []
        if "plot" in questions_text or "visualize" in questions_text:
            vis.append("plot")
        if "histogram" in questions_text:
            vis.append("histogram")
        if "scatter" in questions_text:
            vis.append("scatter")
        if "bar" in questions_text:
            vis.append("bar")
        # Add more detection logic as needed
        return vis

    def _extract_grouping_variables(self, questions_text: str, file_manifest: Dict[str, Any]) -> List[str]:
        prompt = (
            f"Given the question: '{questions_text}' and columns: {list(file_manifest.values())[0].get('columns', [])}, "
            "list the grouping variables as a Python list."
        )
        response = self.llm_client.generate(prompt)
        try:
            return eval(response)
        except Exception:
            return []

    def _extract_target_variables(self, questions_text: str, file_manifest: Dict[str, Any]) -> List[str]:
        prompt = (
            f"Given the question: '{questions_text}' and columns: {list(file_manifest.values())[0].get('columns', [])}, "
            "list the target variables as a Python list."
        )
        response = self.llm_client.generate(prompt)
        try:
            return eval(response)
        except Exception:
            return []

    def _infer_column_types(self, csv_file: str) -> Dict[str, str]:
        # Infer column types from the primary CSV file
        try:
            df = pd.read_csv(csv_file, nrows=100)
            return {col: str(dtype) for col, dtype in df.dtypes.items()}
        except Exception as e:
            logger.warning(f"Could not infer column types: {e}")
            return {}

    def _suggest_data_cleaning(self, csv_file: str) -> List[str]:
        # Suggest basic data cleaning steps
        steps = []
        try:
            df = pd.read_csv(csv_file, nrows=100)
            if df.isnull().any().any():
                steps.append("Handle missing values")
            # Add more cleaning suggestions as needed
        except Exception as e:
            logger.warning(f"Could not analyze data cleaning needs: {e}")
        return steps

    def _build_data_analysis_prompt(self, questions: List[str], file_manifest: Dict[str, Any], plan: Dict[str, Any]) -> str:
        # Build a prompt for LLM code generation
        prompt = (
            "You are a Python data analyst. Given the following questions and CSV files, "
            "write code to perform the requested analysis, including statistical tests and visualizations if needed.\n"
            f"Questions: {questions}\n"
            f"CSV files: {plan.get('csv_files')}\n"
            f"Operations: {plan.get('operations')}\n"
            f"Statistical tests: {plan.get('statistical_tests')}\n"
            f"Visualizations: {plan.get('visualization_types')}\n"
            f"Column types: {plan.get('column_types')}\n"
            f"Data cleaning steps: {plan.get('data_cleaning_steps')}\n"
            "Return results in JSON format and save any plots as PNG files."
        )
        return prompt
