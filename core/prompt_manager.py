from typing import Dict, Any, Optional
import json
import logging
import tiktoken
from dataclasses import dataclass
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Template for generating prompts with token counting and caching."""
    name: str
    template: str
    max_tokens: int
    expected_output_format: Dict[str, Any]
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class PromptManager:
    """Manages prompt templates with token optimization and caching."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.templates = self._load_default_templates()
        self.cache = {}
        
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default prompt templates."""
        return {
            "code_generation": PromptTemplate(
                name="code_generation",
                template=(
                    "Task: Generate Python code for the following requirement.\n"
                    "Requirements: {requirements}\n"
                    "Context: {context}\n"
                    "Constraints: {constraints}\n"
                    "Generate only the code, no explanations."
                ),
                max_tokens=1000,
                expected_output_format={"type": "code", "language": "python"}
            ),
            "error_analysis": PromptTemplate(
                name="error_analysis",
                template=(
                    "Analyze this error and suggest a fix:\n"
                    "Code: {code}\n"
                    "Error: {error}\n"
                    "Return JSON with 'analysis' and 'fix' keys."
                ),
                max_tokens=500,
                expected_output_format={
                    "type": "json",
                    "schema": {
                        "analysis": "string",
                        "fix": "string"
                    }
                }
            ),
            # Add more templates as needed
        }
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """Get a formatted prompt, using cache if available."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
            
        # Generate cache key
        cache_key = self._generate_cache_key(template_name, kwargs)
        
        # Check cache
        if cache_key in self.cache:
            logger.info(f"Cache hit for prompt {template_name}")
            return self.cache[cache_key]
            
        # Format prompt
        prompt = template.format(**kwargs)
        
        # Count tokens
        token_count = len(self.encoder.encode(prompt))
        if token_count > template.max_tokens:
            logger.warning(f"Prompt exceeds token limit: {token_count} > {template.max_tokens}")
            # Truncate or summarize input if needed
            prompt = self._truncate_prompt(prompt, template.max_tokens)
            
        # Cache result
        self.cache[cache_key] = prompt
        if self.cache_dir:
            self._save_to_cache(cache_key, prompt)
            
        return prompt
    
    def _generate_cache_key(self, template_name: str, kwargs: Dict[str, Any]) -> str:
        """Generate a unique cache key for the prompt."""
        content = f"{template_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Intelligently truncate prompt to fit token limit."""
        while len(self.encoder.encode(prompt)) > max_tokens:
            # First try removing newlines and extra spaces
            prompt = " ".join(prompt.split())
            if len(self.encoder.encode(prompt)) <= max_tokens:
                break
                
            # Then truncate content while preserving structure
            lines = prompt.split("\n")
            if len(lines) > 3:
                # Keep first and last lines, truncate middle
                prompt = f"{lines[0]}\n...\n{lines[-1]}"
            else:
                # Last resort: hard truncate
                tokens = self.encoder.encode(prompt)[:max_tokens]
                prompt = self.encoder.decode(tokens)
                
        return prompt
    
    def _save_to_cache(self, cache_key: str, prompt: str) -> None:
        """Save prompt to disk cache."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.txt"
            cache_file.write_text(prompt, encoding='utf-8')
    
    def validate_output_format(self, template_name: str, output: str) -> bool:
        """Validate if the output matches the expected format."""
        template = self.templates.get(template_name)
        if not template:
            return False
            
        expected_format = template.expected_output_format
        
        try:
            if expected_format["type"] == "json":
                # Validate JSON structure
                output_json = json.loads(output)
                schema = expected_format.get("schema", {})
                return all(key in output_json for key in schema.keys())
                
            elif expected_format["type"] == "code":
                # Basic code validation
                return bool(output.strip())
                
            # Add more format validations as needed
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False
            
        return True
