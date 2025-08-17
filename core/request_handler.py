from typing import Dict, List, Any
import logging
from pathlib import Path
import json
import mimetypes
from core.orchestrator import Orchestrator
from core.classifier import WorkflowClassifier

logger = logging.getLogger(__name__)

class RequestHandler:
    """
    Enhanced request handler with validation, sanitization, and queueing.
    """
    def __init__(self, orchestrator: Orchestrator, classifier: WorkflowClassifier):
        self.orchestrator = orchestrator
        self.classifier = classifier
        self.request_queue = []
        self.max_queue_size = 100
        self.supported_file_types = {
            '.txt', '.csv', '.json', '.png', '.jpg', '.jpeg',
            '.pdf', '.xlsx', '.xls', '.db', '.sqlite'
        }

    async def handle_request(self, questions_file: str, additional_files: List[str] = None) -> Dict[str, Any]:
        """
        Handle incoming request with validation and queueing.
        """
        try:
            # Validate and build request
            request = await self._build_request(questions_file, additional_files)
            
            # Add to queue
            if len(self.request_queue) >= self.max_queue_size:
                raise ValueError("Request queue is full, please try again later")
            self.request_queue.append(request)
            
            # Process request
            result = await self._process_request(request)
            
            # Remove from queue
            self.request_queue.remove(request)
            
            return result
        
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": None
            }

    async def _build_request(self, questions_file: str, additional_files: List[str] = None) -> Dict[str, Any]:
        """
        Build and validate request manifest.
        """
        manifest = {
            "questions_file": self._validate_file(questions_file),
            "additional_files": [],
            "urls": [],
            "keywords": []
        }

        # Process additional files
        if additional_files:
            for file in additional_files:
                file_info = self._validate_file(file)
                manifest["additional_files"].append(file_info)

        # Extract URLs and keywords from questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_text = f.read()
            manifest["urls"] = self._extract_urls(questions_text)
            manifest["keywords"] = self._extract_keywords(questions_text)

        return manifest

    def _validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file existence, size, and type.
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        file_size = path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"File too large: {file_path}")

        file_type = path.suffix.lower()
        if file_type not in self.supported_file_types:
            raise ValueError(f"Unsupported file type: {file_type}")

        mime_type, _ = mimetypes.guess_type(file_path)
        
        return {
            "path": str(path.absolute()),
            "size": file_size,
            "type": file_type,
            "mime_type": mime_type
        }

    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text using regex or LLM.
        """
        # TODO: Implement URL extraction
        return []

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords and entities from text using LLM.
        """
        # TODO: Implement keyword extraction
        return []

    async def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process validated request through orchestrator.
        """
        try:
            # Classify workflow
            workflow_type = self.classifier.classify(
                questions=request.get("questions_file"),
                file_manifest=request
            )

            # Process through orchestrator
            result = await self.orchestrator.process_request(
                workflow_type=workflow_type,
                request=request
            )

            return {
                "success": True,
                "result": result,
                "workflow_type": workflow_type
            }

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_type": None
            }
