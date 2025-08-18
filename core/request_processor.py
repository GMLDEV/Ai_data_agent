from typing import Dict, List, Any, Optional
import logging
import asyncio
from pathlib import Path
import json
from datetime import datetime

from core.orchestrator import Orchestrator
from core.classifier import WorkflowClassifier
from core.error_handler import ErrorHandler
from core.prompt_manager import PromptManager
from core.code_generator import CodeGenerator

logger = logging.getLogger(__name__)

class RequestProcessor:
    """
    Central request processing system that integrates all components:
    - Request validation and preprocessing
    - Workflow classification and routing
    - Execution monitoring and result handling
    - Error management and recovery
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        classifier: WorkflowClassifier,
        prompt_manager: PromptManager,
        error_handler: ErrorHandler,
        cache_dir: Optional[str] = None
    ):
        self.orchestrator = orchestrator
        self.classifier = classifier
        self.prompt_manager = prompt_manager
        self.error_handler = error_handler
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_history: List[Dict[str, Any]] = []

    async def process_request(
        self,
        questions_file: str,
        additional_files: Optional[List[str]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a complete request through the system.
        """
        try:
            # Generate request ID if not provided
            request_id = request_id or self._generate_request_id()
            
            # Track request
            self.active_requests[request_id] = {
                "start_time": datetime.now(),
                "status": "processing",
                "questions_file": questions_file,
                "additional_files": additional_files or []
            }

            # 1. Validate and preprocess request
            request_data = await self._preprocess_request(
                questions_file,
                additional_files,
                request_id
            )

            # 2. Classify workflow
            workflow_type = await self._classify_workflow(request_data)
            
            # 3. Build execution plan
            execution_plan = await self._build_execution_plan(
                workflow_type,
                request_data
            )

            # 4. Execute through orchestrator
            result = await self.orchestrator.process_request(
                workflow_type=workflow_type,
                request=request_data,
                plan=execution_plan
            )

            # 5. Validate and format result
            final_result = await self._process_result(result, execution_plan)

            # Update request tracking
            self.active_requests[request_id].update({
                "status": "completed",
                "end_time": datetime.now(),
                "workflow_type": workflow_type
            })

            # Archive request
            self._archive_request(request_id)

            return final_result

        except Exception as e:
            error_ctx = self.error_handler.handle_error(e)
            logger.error(f"Error processing request {request_id}: {error_ctx.message}")
            
            if request_id in self.active_requests:
                self.active_requests[request_id].update({
                    "status": "failed",
                    "end_time": datetime.now(),
                    "error": str(e)
                })
                self._archive_request(request_id)

            return {
                "success": False,
                "error": str(e),
                "request_id": request_id,
                "error_details": error_ctx.__dict__
            }

    async def _preprocess_request(
        self,
        questions_file: str,
        additional_files: Optional[List[str]],
        request_id: str
    ) -> Dict[str, Any]:
        """
        Validate and preprocess the request files.
        """
        # Read and validate questions file
        questions_path = Path(questions_file)
        if not questions_path.exists():
            raise ValueError(f"Questions file not found: {questions_file}")

        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_text = f.read()

        # Process additional files
        file_manifest = {}
        if additional_files:
            for file_path in additional_files:
                path = Path(file_path)
                if not path.exists():
                    raise ValueError(f"File not found: {file_path}")
                
                file_manifest[str(path)] = {
                    "size": path.stat().st_size,
                    "type": path.suffix.lower(),
                    "last_modified": datetime.fromtimestamp(path.stat().st_mtime)
                }

        return {
            "request_id": request_id,
            "questions": questions_text,
            "file_manifest": file_manifest,
            "timestamp": datetime.now().isoformat()
        }

    async def _classify_workflow(self, request_data: Dict[str, Any]) -> str:
        """
        Use classifier to determine appropriate workflow.
        """
        return self.classifier.classify(
            questions=request_data["questions"],
            file_manifest=request_data["file_manifest"]
        )

    async def _build_execution_plan(
        self,
        workflow_type: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build an execution plan based on workflow type and request.
        """
        # Get workflow-specific prompt
        plan_prompt = self.prompt_manager.get_prompt(
            f"{workflow_type}_plan",
            request=request_data
        )

        # Generate plan
        return {
            "workflow_type": workflow_type,
            "steps": [
                "preprocess_data",
                "generate_code",
                "execute_code",
                "validate_output"
            ],
            "requirements": request_data,
            "constraints": {
                "max_execution_time": 180,  # 3 minutes
                "max_memory": 512 * 1024 * 1024,  # 512MB
                "output_size_limit": 100 * 1024  # 100KB
            }
        }

    async def _process_result(
        self,
        result: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and validate the execution result.
        """
        # Basic validation
        if not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")

        # Size validation
        result_size = len(json.dumps(result))
        if result_size > plan["constraints"]["output_size_limit"]:
            raise ValueError(f"Result size ({result_size} bytes) exceeds limit")

        # Add metadata
        result.update({
            "processed_at": datetime.now().isoformat(),
            "workflow_type": plan["workflow_type"]
        })

        return result

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"req_{timestamp}"

    def _archive_request(self, request_id: str) -> None:
        """Archive a completed request."""
        if request_id in self.active_requests:
            request_data = self.active_requests.pop(request_id)
            self.request_history.append(request_data)
            
            # Optionally save to cache
            if self.cache_dir:
                cache_file = self.cache_dir / f"request_{request_id}.json"
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(request_data, f, indent=2, default=str)

    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a request."""
        if request_id in self.active_requests:
            return self.active_requests[request_id]
            
        # Check history
        for req in self.request_history:
            if req.get("request_id") == request_id:
                return req
                
        raise ValueError(f"Request {request_id} not found")
