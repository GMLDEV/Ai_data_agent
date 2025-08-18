from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any, Optional
import json
import logging
from config import settings

logger = logging.getLogger(__name__)

class WorkflowClassifier:
    def __init__(self, llm_client=None):
        # If an LLMClient object is passed, extract the API key
        if hasattr(llm_client, 'api_key'):
            self.api_key = llm_client.api_key
        elif isinstance(llm_client, str):
            # If a string is passed, treat it as the API key
            self.api_key = llm_client
        else:
            # Fallback to settings
            self.api_key = settings.openai_api_key
            
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = OpenAI(
            api_key=self.api_key,
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=500
        )
        
        self.classification_prompt = PromptTemplate(
            input_variables=["questions", "file_types", "urls", "keywords", "file_info"],
            template="""
You are an AI workflow classifier. Analyze the user request and classify it into the most appropriate workflow.

AVAILABLE WORKFLOWS:
1. data_analysis - For CSV/data analysis, statistics, plotting, data manipulation with existing data files
2. web_scraping - For extracting data from websites/URLs  
3. image_analysis - For image processing, OCR, computer vision tasks
4. code_generation - For simple programming tasks, algorithms, utilities
5. multimodal - For tasks requiring multiple data types (CSV + images, etc.)
6. dynamic - For complex programming tasks, network analysis, code generation when no data files, or ambiguous requests

USER REQUEST: {questions}

AVAILABLE FILES: {file_types}
FILE DETAILS: {file_info}
URLS FOUND: {urls}
EXTRACTED KEYWORDS: {keywords}

CLASSIFICATION RULES (in priority order):
1. If URLs present → web_scraping
2. If image files present → image_analysis  
3. If CSV files + clear analysis intent (analyze/plot/statistics keywords) → data_analysis
4. If multiple file types → multimodal
5. If programming/network/graph/code keywords (especially without data files) → dynamic
6. If simple programming keywords → code_generation
7. If unclear or complex requests → dynamic (default)

IMPORTANT: Only use data_analysis if there are CSV files AND clear analysis intent. 
For network analysis, graph creation, or code generation requests → use dynamic workflow.
For general programming without files → use dynamic workflow.

Return ONLY valid JSON in this exact format:
{{
    "workflow": "workflow_name",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this workflow was chosen",
    "fallback": "alternative_workflow_if_unsure",
    "detected_intent": "description_of_user_intent"
}}
"""
        )
    
    def classify(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the request into appropriate workflow"""
        try:
            # Prepare file information summary
            file_info = self._summarize_file_info(manifest.get('files', {}))
            
            # Create the prompt
            prompt = self.classification_prompt.format(
                questions=manifest.get('questions', ''),
                file_types=manifest.get('file_types', []),
                urls=manifest.get('urls', []),
                keywords=manifest.get('keywords', []),
                file_info=file_info
            )
            
            logger.info(f"Classifying request with LLM...")
            response = self.llm.invoke(prompt)
            logger.info(f"LLM Response: {response}")
            
            # Parse JSON response
            try:
                classification = json.loads(response.strip())
                
                # Validate required fields
                required_fields = ['workflow', 'confidence', 'reasoning']
                for field in required_fields:
                    if field not in classification:
                        raise ValueError(f"Missing required field: {field}")
                
                # Validate workflow name
                valid_workflows = ['data_analysis', 'web_scraping', 'image_analysis', 
                                 'code_generation', 'multimodal', 'dynamic']
                if classification['workflow'] not in valid_workflows:
                    raise ValueError(f"Invalid workflow: {classification['workflow']}")
                
                # Apply confidence threshold
                if classification.get('confidence', 0) < 0.5:
                    logger.warning("Low confidence classification, applying fallback")
                    fallback_classification = self._apply_deterministic_fallback(manifest)
                    classification['original_classification'] = classification.copy()
                    classification.update(fallback_classification)
                
                return classification
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"LLM classification parsing failed: {e}, using fallback")
                return self._apply_deterministic_fallback(manifest)
                
        except Exception as e:
            logger.error(f"LLM classification failed: {e}, using deterministic fallback")
            return self._apply_deterministic_fallback(manifest)
    
    def _summarize_file_info(self, files: Dict[str, Any]) -> str:
        """Create a summary of file information for the LLM"""
        if not files:
            return "No files uploaded"
        
        summaries = []
        for filename, info in files.items():
            file_summary = f"- {filename} ({info.get('type', 'unknown')})"
            
            if info.get('type') == 'csv':
                shape = info.get('shape', [0, 0])
                file_summary += f": {shape[0]} rows, {shape[1]} columns"
                if info.get('columns'):
                    file_summary += f", columns: {', '.join(info['columns'][:3])}"
                    if len(info['columns']) > 3:
                        file_summary += "..."
            
            elif info.get('type') == 'image':
                dims = info.get('dimensions', [0, 0])
                file_summary += f": {dims[0]}x{dims[1]} pixels"
            
            summaries.append(file_summary)
        
        return "\n".join(summaries)
    
    def _apply_deterministic_fallback(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic fallback classification rules"""
        file_types = manifest.get('file_types', [])
        urls = manifest.get('urls', [])
        keywords = manifest.get('keywords', [])
        questions = manifest.get('questions', '').lower()
        
        # Rule-based classification with improved logic
        
        # 1. Web scraping - highest priority for URL-related requests
        if urls or any(word in questions for word in ['url', 'scrape', 'website', 'web', 'crawl', 'extract from site']):
            return {
                "workflow": "web_scraping",
                "confidence": 0.9,
                "reasoning": "URLs detected or web scraping keywords found",
                "fallback": "dynamic",
                "method": "deterministic_fallback"
            }
        
        # 2. Image analysis - for image files
        if 'image' in file_types or any(word in questions for word in ['image', 'photo', 'picture', 'ocr', 'vision', 'detect objects']):
            return {
                "workflow": "image_analysis",
                "confidence": 0.9,
                "reasoning": "Image files detected or image processing keywords found",
                "fallback": "dynamic",
                "method": "deterministic_fallback"
            }
        
        # 3. Data analysis - only when clear analysis intent with CSV files
        if 'csv' in file_types and any(word in questions for word in ['analyze', 'analysis', 'plot', 'chart', 'statistics', 'stat', 'correlation', 'mean', 'average', 'visualize', 'trend', 'explore data']):
            return {
                "workflow": "data_analysis",
                "confidence": 0.9,
                "reasoning": "CSV file detected with clear data analysis intent",
                "fallback": "dynamic",
                "method": "deterministic_fallback"
            }
        
        # 4. Multimodal - for multiple file types
        if len(set(file_types)) > 1:
            return {
                "workflow": "multimodal",
                "confidence": 0.8,
                "reasoning": "Multiple file types detected requiring multimodal processing",
                "fallback": "dynamic",
                "method": "deterministic_fallback"
            }
        
        # 5. Dynamic - for programming, code generation, complex tasks
        if any(word in questions for word in [
            'code', 'function', 'algorithm', 'program', 'generate', 'create', 
            'networkx', 'network', 'graph', 'script', 'python', 'programming',
            'implement', 'build', 'develop', 'compute', 'calculate', 'solve'
        ]):
            return {
                "workflow": "dynamic",
                "confidence": 0.8,
                "reasoning": "Programming, code generation, or complex computation keywords detected",
                "fallback": "code_generation",
                "method": "deterministic_fallback"
            }
        
        # 6. Code generation - for simpler programming tasks
        if any(word in questions for word in ['simple', 'basic', 'utility', 'helper', 'quick']):
            return {
                "workflow": "code_generation",
                "confidence": 0.7,
                "reasoning": "Simple programming task keywords detected",
                "fallback": "dynamic",
                "method": "deterministic_fallback"
            }
        
        # 7. Default - dynamic workflow for unclear cases
        return {
            "workflow": "dynamic",
            "confidence": 0.6,
            "reasoning": "No clear pattern detected, using dynamic workflow as versatile default",
            "fallback": "code_generation",
            "method": "deterministic_fallback"
        }