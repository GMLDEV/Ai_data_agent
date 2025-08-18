"""
OpenAI-powered intelligent code fixing and retry system.
Handles code error analysis, fixing, and visualization generation.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAICodeFixer:
    """
    OpenAI-powered code fixer that analyzes errors and suggests fixes.
    Also capable of generating visualizations and enhanced code.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the OpenAI code fixer.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: OpenAI model to use for code generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Track retry attempts and fixes
        self.retry_history: List[Dict[str, Any]] = []
        
    def fix_code_error(
        self, 
        code: str, 
        error_message: str, 
        stderr: str, 
        attempt_number: int,
        context: Dict[str, Any] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Fix code errors using OpenAI GPT models.
        
        Args:
            code: The original code that failed
            error_message: Brief error description
            stderr: Full stderr output
            attempt_number: Current retry attempt number
            context: Additional context about the execution environment
            
        Returns:
            Tuple of (fixed_code, metadata)
        """
        logger.info(f"Requesting OpenAI code fix for attempt {attempt_number}")
        
        try:
            # Build comprehensive prompt
            fix_prompt = self._build_fix_prompt(
                code, error_message, stderr, attempt_number, context
            )
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": fix_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            fixed_code = response.choices[0].message.content.strip()
            
            # Extract code from response if wrapped in markdown
            fixed_code = self._extract_code_from_response(fixed_code)
            
            # Validate and enhance the fixed code
            metadata = {
                "attempt_number": attempt_number,
                "original_error": error_message,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens,
                "fix_applied": True,
                "enhancement_type": self._detect_enhancement_type(fixed_code, code)
            }
            
            # Store in retry history
            self.retry_history.append({
                "attempt": attempt_number,
                "original_code": code,
                "fixed_code": fixed_code,
                "error": error_message,
                "metadata": metadata
            })
            
            logger.info(f"OpenAI provided code fix ({response.usage.total_tokens} tokens)")
            return fixed_code, metadata
            
        except Exception as e:
            logger.error(f"OpenAI code fix failed: {e}")
            return None, {"error": str(e), "fix_applied": False}
    
    def generate_visualization_code(
        self,
        data_description: str,
        data_sample: str = "",
        visualization_type: str = "auto"
    ) -> Optional[str]:
        """
        Generate visualization code using OpenAI.
        
        Args:
            data_description: Description of the data to visualize
            data_sample: Sample of the actual data
            visualization_type: Type of visualization (auto, chart, plot, dashboard, etc.)
            
        Returns:
            Generated visualization code
        """
        logger.info(f"Generating visualization code for: {visualization_type}")
        
        try:
            viz_prompt = f"""
Generate Python code to create visualizations for the following data:

Data Description: {data_description}

Data Sample:
{data_sample[:1000] if data_sample else "No sample provided"}

Visualization Requirements:
- Type: {visualization_type}
- Use matplotlib, seaborn, or plotly as appropriate
- Create multiple visualizations if the data supports it
- Include proper titles, labels, and legends
- Handle missing data gracefully
- Save plots to files with descriptive names
- Print summary statistics

Generate complete, runnable Python code:
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data visualization specialist. Generate clean, well-documented Python code for data visualization."
                    },
                    {
                        "role": "user",
                        "content": viz_prompt
                    }
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            viz_code = response.choices[0].message.content.strip()
            viz_code = self._extract_code_from_response(viz_code)
            
            logger.info(f"Generated visualization code ({response.usage.total_tokens} tokens)")
            return viz_code
            
        except Exception as e:
            logger.error(f"Visualization code generation failed: {e}")
            return None
    
    def enhance_code_with_retries(
        self, 
        code: str, 
        enhancement_type: str = "robustness"
    ) -> Optional[str]:
        """
        Enhance code to be more robust with better error handling and retries.
        
        Args:
            code: Original code
            enhancement_type: Type of enhancement (robustness, performance, visualization)
            
        Returns:
            Enhanced code
        """
        logger.info(f"Enhancing code with {enhancement_type} improvements")
        
        enhancement_prompts = {
            "robustness": """
Enhance this code to be more robust by adding:
1. Comprehensive error handling with try-catch blocks
2. Retry logic for network requests
3. Input validation
4. Graceful degradation for failures
5. Better logging and status messages
6. Timeout handling
7. Resource cleanup
            """,
            "performance": """
Optimize this code for better performance by:
1. Adding caching where appropriate
2. Using more efficient algorithms
3. Minimizing memory usage
4. Parallelizing operations where possible
5. Adding progress indicators
6. Optimizing loops and data structures
            """,
            "visualization": """
Enhance this code to include data visualization by:
1. Adding appropriate charts and graphs
2. Creating summary statistics
3. Generating visual reports
4. Adding interactive elements where possible
5. Saving visualizations to files
6. Creating dashboards if applicable
            """
        }
        
        try:
            enhance_prompt = f"""
{enhancement_prompts.get(enhancement_type, enhancement_prompts['robustness'])}

Original Code:
```python
{code}
```

Generate the enhanced version maintaining all original functionality:
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Python developer. Enhance code while maintaining functionality and adding robust error handling."
                    },
                    {
                        "role": "user",
                        "content": enhance_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            enhanced_code = response.choices[0].message.content.strip()
            enhanced_code = self._extract_code_from_response(enhanced_code)
            
            logger.info(f"Code enhanced with {enhancement_type} improvements")
            return enhanced_code
            
        except Exception as e:
            logger.error(f"Code enhancement failed: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for code fixing."""
        return """
You are an expert Python developer and debugger. Your job is to fix Python code errors while preserving the original intent and functionality.

Guidelines:
1. Analyze the error message and code carefully
2. Identify the root cause of the issue
3. Provide a minimal, targeted fix that resolves the error
4. Maintain all original functionality and logic
5. Do NOT add features that weren't in the original code
6. Do NOT add visualizations unless the original code was specifically trying to create them
7. Do NOT change the core purpose or scope of the code
8. Add robust error handling only where necessary for the fix
9. Include helpful comments explaining the fix
10. Return ONLY the corrected Python code
11. Ensure the code is production-ready and handles edge cases
12. Keep the same imports and overall structure unless they're causing the error

Focus on common issues like:
- Import errors and missing packages
- API changes and deprecated methods  
- Type errors and attribute errors
- Network connectivity issues
- File I/O problems
- Data parsing errors
- Syntax errors

IMPORTANT: Your goal is to make the code work as originally intended, not to enhance or extend its functionality.
"""
    
    def _build_fix_prompt(
        self, 
        code: str, 
        error_message: str, 
        stderr: str, 
        attempt_number: int,
        context: Dict[str, Any] = None
    ) -> str:
        """Build a comprehensive prompt for code fixing."""
        
        context_info = ""
        if context:
            context_info = f"""
Execution Context:
- Environment: {context.get('environment', 'Unknown')}
- Allowed libraries: {context.get('allowed_libraries', [])}
- Timeout: {context.get('timeout', 'Not specified')}
- Working directory: {context.get('working_dir', 'Unknown')}
"""

        previous_attempts = ""
        if attempt_number > 1 and self.retry_history:
            previous_attempts = "\nPrevious Retry Attempts:\n"
            for i, attempt in enumerate(self.retry_history[-3:]):  # Last 3 attempts
                previous_attempts += f"Attempt {attempt['attempt']}: {attempt['error']}\n"
        
        return f"""
Fix this Python code that is failing with an error.

Error Message: {error_message}

Full Error Output:
{stderr}

Current Attempt: {attempt_number}
{previous_attempts}
{context_info}

Failing Code:
```python
{code}
```

Provide the corrected Python code:
"""
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from OpenAI response, handling markdown formatting."""
        # Remove markdown code blocks
        code_pattern = r'```(?:python)?\n?(.*?)\n?```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no markdown blocks, return the whole response
        return response.strip()
    
    def _detect_enhancement_type(self, fixed_code: str, original_code: str) -> str:
        """Detect what type of enhancement was applied."""
        fixed_lower = fixed_code.lower()
        original_lower = original_code.lower()
        
        if "matplotlib" in fixed_lower or "seaborn" in fixed_lower or "plotly" in fixed_lower:
            if not ("matplotlib" in original_lower or "seaborn" in original_lower or "plotly" in original_lower):
                return "visualization_added"
        
        if "try:" in fixed_lower and "except" in fixed_lower:
            if not ("try:" in original_lower and "except" in original_lower):
                return "error_handling_added"
        
        if "retry" in fixed_lower or "attempt" in fixed_lower:
            return "retry_logic_added"
        
        if len(fixed_code) > len(original_code) * 1.5:
            return "major_enhancement"
        
        return "bug_fix"
    
    def get_retry_history(self) -> List[Dict[str, Any]]:
        """Get the history of retry attempts."""
        return self.retry_history.copy()
    
    def clear_history(self):
        """Clear the retry history."""
        self.retry_history.clear()
