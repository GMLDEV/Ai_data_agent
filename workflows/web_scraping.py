from typing import Dict, List, Any
import requests
from bs4 import BeautifulSoup
import logging
from workflows.base import BaseWorkflow

logger = logging.getLogger(__name__)

class WebScrapingWorkflow(BaseWorkflow):
    def __init__(self, code_generator, manifest, sandbox_executor=None, llm_client=None):
        super().__init__(code_generator, manifest)
        self.sandbox_executor = sandbox_executor
        self.llm_client = llm_client

    def get_workflow_type(self):
        return "web_scraping"
    """
    Workflow for web scraping tasks.
    Uses LLM to analyze page structure and generate scraping scripts.
    """

    def plan(self, questions: List[str], file_manifest: Dict[str, Any], keywords: List[str], urls: List[str]) -> Dict[str, Any]:
        """
        Analyze requirements and page structure using LLM.
        Optionally fetch HTML and summarize for LLM.
        Return plan with extraction targets, navigation steps, etc.
        """
        logger.info("Planning web scraping workflow")
        if not urls:
            raise ValueError("No URLs provided for web scraping workflow")
        # Fetch HTML for each URL
        html_summaries = {}
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                # Summarize HTML structure for LLM
                html_summaries[url] = self._summarize_html(soup)
            except Exception as e:
                html_summaries[url] = f"Error fetching: {e}"
        # Build prompt for LLM
        prompt = self._build_scraping_plan_prompt(questions, urls, html_summaries)
        plan_text = self.llm_client.generate(prompt)
        try:
            plan = eval(plan_text) if plan_text.strip().startswith('{') else {'targets': [], 'navigation': []}
        except Exception:
            plan = {'targets': [], 'navigation': []}
        plan['urls'] = urls
        plan['html_summaries'] = html_summaries
        return plan

    def generate_code(self, questions: List[str], file_manifest: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        Prompt LLM to write a Python script for scraping and navigation.
        """
        prompt = self._build_scraping_code_prompt(questions, plan)
        code = self.code_generator.generate_code(
            prompt=prompt,
            context={
                "questions": questions,
                "plan": plan,
                "workflow_type": "web_scraping"
            }
        )
        return code

    def execute(self, code: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run script in sandbox.
        """
        logger.info("Executing web scraping code in sandbox")
        result = self.sandbox_executor.execute_simple(code)
        return result

    def validate(self, result: Dict[str, Any], plan: Dict[str, Any]) -> bool:
        """
        Check output format and completeness.
        """
        logger.info("Validating web scraping output")
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
        logger.info("Repairing web scraping code using error feedback")
        repair_prompt = (
            f"The following code failed with error:\n{error}\n"
            f"Original code:\n{code}\n"
            "Please fix the code for the web scraping task described in the plan."
        )
        repaired_code = self.code_generator.generate_code(
            prompt=repair_prompt,
            context={"plan": plan, "workflow_type": "web_scraping"}
        )
        return repaired_code

    # --- Helper methods ---
    def _summarize_html(self, soup: BeautifulSoup) -> str:
        """
        Summarize HTML structure for LLM prompt.
        """
        tags = [tag.name for tag in soup.find_all()]
        tag_counts = {t: tags.count(t) for t in set(tags)}
        summary = f"Tags: {tag_counts}\n"
        tables = soup.find_all('table')
        summary += f"Tables found: {len(tables)}\n"
        links = soup.find_all('a')
        summary += f"Links found: {len(links)}\n"
        return summary

    def _build_scraping_plan_prompt(self, questions: List[str], urls: List[str], html_summaries: Dict[str, str]) -> str:
        prompt = (
            "You are a Python web scraping expert. Given the following requirements and HTML summaries, "
            "analyze what data should be extracted and how to navigate the pages. "
            "Return a Python dict with keys 'targets' (list of data to extract) and 'navigation' (steps to follow).\n"
            f"Questions: {questions}\n"
            f"URLs: {urls}\n"
            f"HTML summaries: {html_summaries}\n"
        )
        return prompt

    def _build_scraping_code_prompt(self, questions: List[str], plan: Dict[str, Any]) -> str:
        prompt = (
            "You are a Python web scraping expert. Given the following requirements and plan, "
            "write a Python script using requests and BeautifulSoup to extract the required data. "
            "If navigation is needed (pagination, multi-step forms), include code to handle it. "
            "Save extracted tables as CSV.\n"
            f"Questions: {questions}\n"
            f"Plan: {plan}\n"
            "Do not use your own scraping logic; only generate Python code."
        )
        return prompt
