from typing import Dict, List, Any
import logging
from workflows.base import BaseWorkflow

logger = logging.getLogger(__name__)

class ImageAnalysisWorkflow(BaseWorkflow):
    def __init__(self, code_generator, manifest, sandbox_executor=None, llm_client=None):
        super().__init__(code_generator=code_generator, manifest=manifest)
        self.sandbox_executor = sandbox_executor
        self.llm_client = llm_client

    def get_workflow_type(self):
        return "image_analysis"
    """
    Workflow for image analysis tasks.
    Supports OCR, basic computer vision, and LLM-guided code generation.
    """

    def plan(self, questions: List[str], file_manifest: Dict[str, Any], keywords: List[str], image_files: List[str]) -> Dict[str, Any]:
        """
        Analyze requirements and image file types using LLM.
        Return plan with analysis targets (OCR, object detection, etc.).
        """
        logger.info("Planning image analysis workflow")
        if not image_files:
            raise ValueError("No image files provided for image analysis workflow")
        # Build prompt for LLM
        prompt = self._build_image_plan_prompt(questions, image_files)
        plan_text = self.llm_client.generate(prompt)
        try:
            plan = eval(plan_text) if plan_text.strip().startswith('{') else {'tasks': []}
        except Exception:
            plan = {'tasks': []}
        plan['image_files'] = image_files
        return plan

    def generate_code(self, questions: List[str], file_manifest: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Prompt LLM to write a Python script for image analysis.
        """
        prompt = self._build_image_code_prompt(questions, plan)
        code = self.code_generator.generate_code(
            prompt=prompt,
            context={
                "questions": questions,
                "plan": plan,
                "workflow_type": "image_analysis"
            }
        )
        return code

    def execute(self, code: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run script in sandbox.
        """
        logger.info("Executing image analysis code in sandbox")
        result = self.sandbox_executor.execute_simple(code)
        return result

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
            prompt=repair_prompt,
            context={"plan": plan, "workflow_type": "image_analysis"}
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
