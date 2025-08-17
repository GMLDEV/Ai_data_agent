import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from pathlib import Path
import json
from datetime import datetime

from core.request_processor import RequestProcessor
from core.error_handler import ErrorHandler
from core.prompt_manager import PromptManager

class TestRequestProcessor(unittest.TestCase):
    """Test suite for RequestProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_orchestrator = Mock()
        self.mock_orchestrator.process_request = AsyncMock(return_value={"success": True, "output": "test result", "request_id": "test_123"})
        self.mock_classifier = Mock()
        self.mock_prompt_manager = Mock()
        self.error_handler = ErrorHandler()

        self.processor = RequestProcessor(
            orchestrator=self.mock_orchestrator,
            classifier=self.mock_classifier,
            prompt_manager=self.mock_prompt_manager,
            error_handler=self.error_handler
        )

    def tearDown(self):
        """Clean up after tests."""
        pass

    def test_request_preprocessing(self):
        """Test request preprocessing logic."""
        # Create temporary test files
        with self.create_temp_files() as (questions_file, additional_file):
            # Test valid request
            result = asyncio.run(self.processor._preprocess_request(
                questions_file=str(questions_file),
                additional_files=[str(additional_file)],
                request_id="test_123"
            ))
            
            self.assertIn("request_id", result)
            self.assertIn("questions", result)
            self.assertIn("file_manifest", result)
            
            # Test invalid file
            with self.assertRaises(ValueError):
                asyncio.run(self.processor._preprocess_request(
                    questions_file="nonexistent.txt",
                    additional_files=None,
                    request_id="test_456"
                ))

    def test_workflow_classification(self):
        """Test workflow classification."""
        # Mock classifier response
        self.mock_classifier.classify.return_value = "data_analysis"
        
        request_data = {
            "questions": "Analyze this CSV file",
            "file_manifest": {"test.csv": {"type": "csv"}}
        }
        
        result = asyncio.run(self.processor._classify_workflow(request_data))
        self.assertEqual(result, "data_analysis")
        self.mock_classifier.classify.assert_called_once()

    def test_execution_plan_building(self):
        """Test execution plan building."""
        request_data = {
            "questions": "Test question",
            "file_manifest": {}
        }
        
        result = asyncio.run(self.processor._build_execution_plan(
            workflow_type="data_analysis",
            request_data=request_data
        ))
        
        self.assertIn("workflow_type", result)
        self.assertIn("steps", result)
        self.assertIn("constraints", result)
        self.assertEqual(result["workflow_type"], "data_analysis")

    def test_result_processing(self):
        """Test result processing and validation."""
        # Test valid result
        valid_result = {"output": "test"}
        plan = {
            "workflow_type": "test",
            "constraints": {"output_size_limit": 1000}
        }
        
        result = asyncio.run(self.processor._process_result(valid_result, plan))
        self.assertIn("processed_at", result)
        self.assertIn("workflow_type", result)
        
        # Test oversized result
        large_result = {"output": "x" * 2000}
        with self.assertRaises(ValueError):
            asyncio.run(self.processor._process_result(
                large_result,
                {"constraints": {"output_size_limit": 100}}
            ))

    def test_full_request_processing(self):
        """Test complete request processing flow."""
        # Mock successful workflow execution
        self.mock_orchestrator.process_request.return_value = {
            "success": True,
            "request_id": "test_123",
            "workflow_used": "data_analysis",
            "confidence": 0.9,
            "reasoning": "Test reasoning",
            "llm_available": True,
            "result": {"output": "test result"}
        }
        self.mock_classifier.classify.return_value = "data_analysis"
        
        # Create test files
        with self.create_temp_files() as (questions_file, additional_file):
            result = asyncio.run(self.processor.process_request(
                questions_file=str(questions_file),
                additional_files=[str(additional_file)]
            ))
            
            self.assertTrue(result.get("success"))
            self.assertIn("request_id", result)

    def test_error_handling(self):
        """Test error handling in request processing."""
        # Mock error in orchestrator
        self.mock_orchestrator.process_request.side_effect = Exception("Test error")
        
        # Create test file
        with self.create_temp_files() as (questions_file, _):
            result = asyncio.run(self.processor.process_request(
                questions_file=str(questions_file)
            ))
            
            self.assertFalse(result.get("success"))
            self.assertIn("error", result)
            self.assertIn("error_details", result)

    def create_temp_files(self):
        """Create temporary test files."""
        import tempfile
        import contextlib
        
        @contextlib.contextmanager
        def temp_files():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as qf:
                qf.write("Test question")
                questions_file = Path(qf.name)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as af:
                af.write("test,data\n1,2")
                additional_file = Path(af.name)
            
            try:
                yield questions_file, additional_file
            finally:
                questions_file.unlink()
                additional_file.unlink()
        
        return temp_files()

if __name__ == '__main__':
    unittest.main()
