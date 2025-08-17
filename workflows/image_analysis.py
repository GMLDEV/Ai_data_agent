from typing import Dict, List, Any
import logging
from workflows.base import BaseWorkflow
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class ImageAnalysisWorkflow(BaseWorkflow):
    def __init__(self, code_generator, manifest, sandbox_executor=None, llm_client=None):
        super().__init__(code_generator=code_generator, manifest=manifest)
        self.sandbox_executor = sandbox_executor
        self.llm_client = llm_client or LLMClient()

    def get_workflow_type(self):
        return "image_analysis"
    """
    Workflow for image analysis tasks.
    Supports OCR, basic computer vision, and LLM-guided code generation.
    """

    def plan(self, questions: List[str], file_manifest: Dict[str, Any], keywords: List[str] = None, image_files: List[str] = None) -> Dict[str, Any]:
        """
        Analyze requirements and image file types using LLM.
        Return plan with analysis targets (OCR, object detection, etc.).
        """
        logger.info("Planning image analysis workflow")
        image_files = image_files or []
        keywords = keywords or []
        if not image_files:
            raise ValueError("No image files provided for image analysis workflow")
        # Build prompt for LLM
        prompt = self._build_image_plan_prompt(questions, image_files)
        plan_text = self.llm_client.generate(prompt)
        try:
            import json
            # Try to parse as JSON first
            if plan_text.strip().startswith('{'):
                plan = json.loads(plan_text)
            else:
                plan = {'tasks': []}
        except (json.JSONDecodeError, Exception):
            plan = {'tasks': []}
        plan['image_files'] = image_files
        return plan

    def generate_code(self, questions: List[str], file_manifest: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Prompt LLM to write a Python script for image analysis.
        """
        prompt = self._build_image_code_prompt(questions, plan)
        code = self.code_generator.generate_code(
            task_description=" ".join(questions),
            manifest=file_manifest,
            workflow_type="image_analysis",
            output_format="json"
        )
        return code

    def execute(self, sandbox_executor, task_description: str) -> Dict[str, Any]:
        """Execute the image analysis workflow using the provided sandbox executor."""
        logger.info("Executing image analysis workflow")

        questions = [task_description]
        file_manifest = self.manifest
        image_files = [f for f, info in file_manifest.items() if info.get('type') in ('png', 'jpg', 'jpeg')]

        plan = self.plan(questions, file_manifest, keywords=[], image_files=image_files)
        code = self.generate_code(questions, file_manifest, plan)

        executor = sandbox_executor or self.sandbox_executor
        if executor is None:
            raise RuntimeError("No sandbox executor available for ImageAnalysisWorkflow")

        result = executor.execute_simple(code)
        return {
            "success": result.get("success", False),
            "result": result,
            "workflow": self.get_workflow_type()
        }

    def validate(self, result: Dict[str, Any], plan: Dict[str, Any]) -> bool:
        """
        Check output format and completeness.
        """
        logger.info("Validating image analysis output")
        if not result.get("success"):
            return False
        if len(result.get("stdout", "")) > 100_000:
            logger.warning("Output exceeds size limit")
            return False
        # Add more schema/format checks as needed
        return True

    def repair(self, code: str, error: str, plan: Dict[str, Any]) -> str:
        """
        Use LLM to fix code based on error feedback.
        """
        logger.info("Repairing image analysis code using error feedback")
        repair_prompt = (
            f"The following code failed with error:\n{error}\n"
            f"Original code:\n{code}\n"
            "Please fix the code for the image analysis task described in the plan."
        )
        repaired_code = self.code_generator.generate_code(
            task_description=repair_prompt,
            manifest=self.manifest,
            workflow_type="image_analysis",
            output_format="json"
        )
        return repaired_code

    # --- Helper methods ---
    def _build_image_plan_prompt(self, questions: List[str], image_files: List[str]) -> str:
        prompt = (
            "You are a Python image analysis expert. Given the following requirements and image files, "
            "analyze what tasks should be performed (OCR, object detection, classification, etc.). "
            "Return a Python dict with keys 'tasks' (list of analysis tasks).\n"
            f"Questions: {questions}\n"
            f"Image files: {image_files}\n"
        )
        return prompt

    def _build_image_code_prompt(self, questions: List[str], plan: Dict[str, Any]) -> str:
        prompt = (
            "You are a Python image analysis expert. Given the following requirements and plan, "
            "write a Python script using Pillow, OpenCV, and pytesseract to perform the required analysis. "
            "If OCR is needed, use pytesseract. If object detection or classification is needed, use OpenCV. "
            "Save results as JSON and any processed images as PNG.\n"
            f"Questions: {questions}\n"
            f"Plan: {plan}\n"
            "Do not use your own analysis logic; only generate Python code."
        )
        return prompt
