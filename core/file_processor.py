# core/file_processor.py - FIXED VERSION
import os
import re
import mimetypes
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
from PIL import Image
from io import StringIO, BytesIO
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FileProcessor:
    def __init__(self):
        self.supported_formats = {
            'csv': ['.csv'],
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
            'text': ['.txt', '.md']
        }
    
    def create_manifest(self, files: Dict[str, bytes], questions_text: str) -> Dict[str, Any]:
        """Create a manifest of all input files and extract metadata"""
        manifest = {
            'files': {},
            'urls': self._extract_urls(questions_text),
            'keywords': self._extract_keywords(questions_text),
            'questions': questions_text,
            'file_types': [],
            'total_files': len(files),
            'processing_errors': []
        }
        
        for filename, file_content in files.items():
            try:
                file_info = self._analyze_file(filename, file_content)
                manifest['files'][filename] = file_info
                manifest['file_types'].append(file_info['type'])
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                manifest['processing_errors'].append(error_msg)
                logger.error(error_msg)
        
        # Remove duplicates from file_types
        manifest['file_types'] = list(set(manifest['file_types']))
        
        return manifest
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from questions text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms and entities from the questions"""
        keywords = []
        
        # Extract proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        keywords.extend(proper_nouns)
        
        # Extract analysis-related keywords
        analysis_keywords = re.findall(
            r'\b(?:analyze|plot|chart|graph|data|table|statistics|correlation|regression|visualization)\b', 
            text.lower()
        )
        keywords.extend(analysis_keywords)
        
        # Extract file format mentions
        format_keywords = re.findall(r'\b(?:csv|json|image|png|jpg|excel)\b', text.lower())
        keywords.extend(format_keywords)
        
        return list(set(keywords))  # Remove duplicates
    
    def _analyze_file(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Analyze individual file and extract metadata"""
        ext = Path(filename).suffix.lower()
        file_info = {
            'filename': filename,
            'extension': ext,
            'type': self._get_file_type(ext),
            'size': len(content),
            'size_mb': round(len(content) / (1024 * 1024), 2)
        }
        
        # Add specific metadata based on file type
        try:
            if file_info['type'] == 'csv':
                file_info.update(self._analyze_csv(content))
            elif file_info['type'] == 'image':
                file_info.update(self._analyze_image(content))
            elif file_info['type'] == 'text':
                file_info.update(self._analyze_text(content))
        except Exception as e:
            file_info['analysis_error'] = str(e)
            logger.warning(f"Could not analyze {filename}: {e}")
        
        return file_info
    
    def _analyze_csv(self, content: bytes) -> Dict[str, Any]:
        """Analyze CSV file content - FIXED for JSON serialization"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    df = pd.read_csv(StringIO(text_content))
                    break
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
            else:
                return {'error': 'Could not decode CSV file'}
            
            # Convert numpy types to Python native types for JSON serialization
            dtypes_dict = {}
            for col, dtype in df.dtypes.items():
                dtypes_dict[str(col)] = str(dtype)
            
            # Convert boolean values to Python bool
            has_nulls = bool(df.isnull().any().any())
            
            # Get sample rows and convert to JSON-serializable format
            sample_rows = df.head(3).to_dict('records')
            # Convert numpy types in sample rows
            for row in sample_rows:
                for key, value in row.items():
                    if pd.isna(value):
                        row[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        row[key] = value.item()  # Convert to Python native type
                    elif isinstance(value, np.bool_):
                        row[key] = bool(value)
                    elif isinstance(value, np.ndarray):
                        row[key] = value.tolist()
            
            return {
                'shape': list(df.shape),  # Convert tuple to list
                'columns': [str(col) for col in df.columns],  # Ensure strings
                'dtypes': dtypes_dict,
                'sample_rows': sample_rows,
                'has_nulls': has_nulls,
                'memory_usage_mb': round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2),
                'row_count': int(df.shape[0]),
                'column_count': int(df.shape[1])
            }
        except Exception as e:
            return {'error': f'CSV analysis failed: {str(e)}'}
    
    def _analyze_image(self, content: bytes) -> Dict[str, Any]:
        """Analyze image file content"""
        try:
            img = Image.open(BytesIO(content))
            return {
                'dimensions': list(img.size),  # Convert tuple to list
                'mode': str(img.mode),
                'format': str(img.format) if img.format else 'Unknown',
                'has_transparency': bool(img.mode in ('RGBA', 'LA') or 'transparency' in img.info)
            }
        except Exception as e:
            return {'error': f'Image analysis failed: {str(e)}'}
    
    def _analyze_text(self, content: bytes) -> Dict[str, Any]:
        """Analyze text file content"""
        try:
            # Try different encodings
            encoding_used = 'utf-8'  # Default
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return {'error': 'Could not decode text file'}
            
            lines = text_content.split('\n')
            return {
                'line_count': len(lines),
                'character_count': len(text_content),
                'word_count': len(text_content.split()),
                'encoding_used': encoding_used,
                'preview': text_content[:200] + '...' if len(text_content) > 200 else text_content
            }
        except Exception as e:
            return {'error': f'Text analysis failed: {str(e)}'}
    
    def _get_file_type(self, extension: str) -> str:
        """Determine file type from extension"""
        for file_type, extensions in self.supported_formats.items():
            if extension in extensions:
                return file_type
        return 'unknown'
    
    def validate_file(self, filename: str, content: bytes, max_size: int) -> Dict[str, Any]:
        """Validate file before processing"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check file size
        if len(content) > max_size:
            validation_result['valid'] = False
            validation_result['errors'].append(f'File size ({len(content)} bytes) exceeds limit ({max_size} bytes)')
        
        # Check file extension
        ext = Path(filename).suffix.lower()
        all_supported = []
        for extensions in self.supported_formats.values():
            all_supported.extend(extensions)
        
        if ext not in all_supported:
            validation_result['warnings'].append(f'File type {ext} may not be fully supported')
        
        # Check for suspicious filenames
        if any(char in filename for char in ['<', '>', '|', ':', '*', '?', '"']):
            validation_result['valid'] = False
            validation_result['errors'].append('Filename contains invalid characters')
        
        return validation_result