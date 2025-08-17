import unittest
from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path
import tempfile
import json
import pytest

from core.request_processor import RequestProcessor
from core.orchestrator import LLMOrchestrator
from core.classifier import WorkflowClassifier
from core.prompt_manager import PromptManager
from core.error_handler import ErrorHandler
from core.code_generator import CodeGenerator
from workflows.data_analysis import DataAnalysisWorkflow
from workflows.web_scraping import WebScrapingWorkflow
from workflows.image_analysis import ImageAnalysisWorkflow

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete AI Data Agent system."""

    @classmethod
    def setUpClass(cls):
        # Explicitly load .env file
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
        """Set up test environment once for all tests."""
        # Create temporary directory for cache
        cls.temp_dir = tempfile.mkdtemp()

        # Initialize components
        cls.prompt_manager = PromptManager(cache_dir=cls.temp_dir)
        cls.error_handler = ErrorHandler()
        from config import Settings
        settings = Settings()
        cls.code_generator = CodeGenerator(api_key=settings.openai_api_key)
        cls.classifier = WorkflowClassifier(api_key=settings.openai_api_key)

        # Initialize workflows with empty manifest
        empty_manifest = {}
        cls.data_workflow = DataAnalysisWorkflow(
            code_generator=cls.code_generator,
            manifest=empty_manifest,
            sandbox_executor=None
        )
        cls.web_workflow = WebScrapingWorkflow(
            code_generator=cls.code_generator,
            manifest=empty_manifest,
            sandbox_executor=None
        )
        cls.image_workflow = ImageAnalysisWorkflow(
            code_generator=cls.code_generator,
            manifest=empty_manifest,
            sandbox_executor=None
        )

        # Initialize orchestrator
        cls.orchestrator = LLMOrchestrator(
            workflows={
                "data_analysis": cls.data_workflow,
                "web_scraping": cls.web_workflow,
                "image_analysis": cls.image_workflow
            }
        )

        # Initialize request processor
        cls.processor = RequestProcessor(
            orchestrator=cls.orchestrator,
            classifier=cls.classifier,
            prompt_manager=cls.prompt_manager,
            error_handler=cls.error_handler,
            cache_dir=cls.temp_dir
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(cls.temp_dir)

    def create_test_files(self):
        """Create test files for different scenarios."""
        # Create questions.txt
        questions_file = Path(self.temp_dir) / "questions.txt"
        questions_file.write_text("Analyze the data in test.csv and create a visualization.")
        
        # Create test.csv
        csv_file = Path(self.temp_dir) / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4")
        
        # Create test image
        image_file = Path(self.temp_dir) / "test.png"
        # Create a simple test image
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_file)
        
        return questions_file, csv_file, image_file

    async def test_data_analysis_workflow(self):
        """Test complete data analysis workflow."""
        questions_file, csv_file, _ = self.create_test_files()
        
        result = await self.processor.process_request(
            questions_file=str(questions_file),
            additional_files=[str(csv_file)]
        )
        
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("workflow_type"), "data_analysis")

    async def test_web_scraping_workflow(self):
        """Test complete web scraping workflow."""
        # Create questions file with URL
        questions_file = Path(self.temp_dir) / "web_questions.txt"
        questions_file.write_text("Scrape data from https://example.com")
        
        result = await self.processor.process_request(
            questions_file=str(questions_file)
        )
        
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("workflow_type"), "web_scraping")

    async def test_image_analysis_workflow(self):
        """Test complete image analysis workflow."""
        questions_file, _, image_file = self.create_test_files()
        
        result = await self.processor.process_request(
            questions_file=str(questions_file),
            additional_files=[str(image_file)]
        )
        
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("workflow_type"), "image_analysis")

    async def test_error_recovery(self):
        """Test error recovery and retry logic."""
        questions_file = Path(self.temp_dir) / "error_test.txt"
        questions_file.write_text("This will cause an error")
        
        # Modify orchestrator to raise an error first time
        original_process = self.orchestrator.process_request
        error_count = 0
        
        async def mock_process(*args, **kwargs):
            nonlocal error_count
            if error_count == 0:
                error_count += 1
                raise Exception("Test error")
            return await original_process(*args, **kwargs)
        
        self.orchestrator.process_request = mock_process
        
        result = await self.processor.process_request(
            questions_file=str(questions_file)
        )
        
        self.assertTrue(result.get("success"))
        self.assertGreater(len(self.error_handler.error_history), 0)

    async def test_cache_usage(self):
        """Test caching mechanism."""
        questions_file, csv_file, _ = self.create_test_files()
        
        # First request
        result1 = await self.processor.process_request(
            questions_file=str(questions_file),
            additional_files=[str(csv_file)]
        )
        
        # Second request with same input
        result2 = await self.processor.process_request(
            questions_file=str(questions_file),
            additional_files=[str(csv_file)]
        )
        
        self.assertEqual(
            result1.get("request_id"),
            result2.get("cached_from")
        )

if __name__ == '__main__':
    pytest.main([__file__])
