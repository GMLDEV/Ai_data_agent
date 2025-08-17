from typing import Dict, Any, Optional, List
import logging
from enum import Enum
import json
from dataclasses import dataclass
import traceback

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of errors that can occur during execution."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorContext:
    """Context information about an error."""
    error_type: str
    message: str
    traceback: str
    category: ErrorCategory
    attempt_number: int
    code_snippet: Optional[str] = None
    fix_suggestion: Optional[str] = None

class ErrorHandler:
    """Handles errors with retry logic and reporting."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.error_history: List[ErrorContext] = []
        
    def handle_error(self, error: Exception, code: Optional[str] = None, attempt: int = 1) -> ErrorContext:
        """Process an error and categorize it."""
        error_type = type(error).__name__
        message = str(error)
        tb = traceback.format_exc()
        
        # Categorize error
        category = self._categorize_error(error)
        
        # Create error context
        context = ErrorContext(
            error_type=error_type,
            message=message,
            traceback=tb,
            category=category,
            attempt_number=attempt,
            code_snippet=code
        )
        
        # Log error
        logger.error(f"Error occurred (attempt {attempt}/{self.max_retries}): {message}")
        self.error_history.append(context)
        
        return context
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        if error_type in ("SyntaxError", "IndentationError"):
            return ErrorCategory.SYNTAX_ERROR
        elif error_type == "ImportError":
            return ErrorCategory.IMPORT_ERROR
        elif error_type == "MemoryError" or "memory" in error_msg:
            return ErrorCategory.MEMORY_ERROR
        elif "timeout" in error_msg:
            return ErrorCategory.TIMEOUT_ERROR
        elif error_type == "PermissionError":
            return ErrorCategory.PERMISSION_ERROR
        elif error_type == "ValidationError":
            return ErrorCategory.VALIDATION_ERROR
        elif error_type == "RuntimeError":
            return ErrorCategory.RUNTIME_ERROR
        else:
            return ErrorCategory.UNKNOWN_ERROR
    
    def should_retry(self, error_context: ErrorContext) -> bool:
        """Determine if we should retry based on error category and attempt number."""
        if error_context.attempt_number >= self.max_retries:
            return False
            
        # Define which categories are retryable
        retryable_categories = {
            ErrorCategory.SYNTAX_ERROR,
            ErrorCategory.RUNTIME_ERROR,
            ErrorCategory.VALIDATION_ERROR
        }
        
        return error_context.category in retryable_categories
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Generate a summary of errors encountered."""
        return {
            "total_errors": len(self.error_history),
            "categories": {cat.value: 0 for cat in ErrorCategory},
            "latest_error": None if not self.error_history else {
                "type": self.error_history[-1].error_type,
                "message": self.error_history[-1].message,
                "category": self.error_history[-1].category.value
            }
        }

class OutputValidator:
    """Validates output against schemas and constraints."""
    
    def __init__(self, size_limit: int = 100_000):
        self.size_limit = size_limit
        
    def validate_output(self, output: Any, expected_schema: Dict[str, Any]) -> bool:
        """Validate output against expected schema."""
        try:
            # Check size constraint
            if isinstance(output, (str, bytes)):
                if len(output) > self.size_limit:
                    logger.error(f"Output size ({len(output)}) exceeds limit ({self.size_limit})")
                    return False
            
            # Validate structure
            if expected_schema.get("type") == "json":
                return self._validate_json(output, expected_schema.get("schema", {}))
            elif expected_schema.get("type") == "code":
                return self._validate_code(output)
            # Add more validation types as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _validate_json(self, output: str, schema: Dict[str, Any]) -> bool:
        """Validate JSON output against schema."""
        try:
            if isinstance(output, str):
                output = json.loads(output)
            
            # Check required fields
            for key, value_type in schema.items():
                if key not in output:
                    return False
                if not isinstance(output[key], eval(value_type)):
                    return False
            
            return True
            
        except json.JSONDecodeError:
            return False
    
    def _validate_code(self, code: str) -> bool:
        """Validate Python code structure."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False
