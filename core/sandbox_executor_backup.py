import subprocess
import tempfile
import os
import shutil
import json
import signal
import time
import sys
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None

import platform

logger = logging.getLogger(__name__)

class SandboxExecutor:
    """
    Secure sandbox executor for running Python code with resource limits
    and restricted environment.
    """
    
    # Package mapping for common import/install mismatches
    PACKAGE_MAPPING = {
        'bs4': 'beautifulsoup4',
        'beautifulsoup4': 'beautifulsoup4',
        'pil': 'pillow',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'serial': 'pyserial',
        'psycopg2': 'psycopg2-binary',
        'mysql': 'mysql-connector-python',
        'dateutil': 'python-dateutil',
        'magic': 'python-magic',
        'Crypto': 'pycryptodome',
        'jwt': 'pyjwt',
        'dotenv': 'python-dotenv'
    }
    
    # Packages that are built-in or platform-specific and should not be installed
    BUILTIN_MODULES = {
        'sys', 'os', 'json', 'csv', 're', 'math', 'datetime', 'time', 'io', 'base64', 
        'collections', 'traceback', 'signal', 'platform', 'subprocess', 'threading', 
        'multiprocessing', 'urllib', 'http', 'email', 'html', 'xml', 'sqlite3',
        'hashlib', 'hmac', 'secrets', 'uuid', 'pickle', 'copy', 'itertools', 'functools',
        'operator', 'pathlib', 'tempfile', 'shutil', 'glob', 'fnmatch', 'linecache',
        'textwrap', 'string', 'struct', 'codecs', 'locale', 'calendar', 'zoneinfo',
        'decimal', 'fractions', 'statistics', 'random', 'bisect', 'heapq', 'array',
        'weakref', 'types', 'enum', 'contextlib', 'abc', 'numbers', 'cmath', 'logging',
        'argparse', 'getopt', 'configparser', 'fileinput', 'readline', 'rlcompleter',
        'msvcrt',  # Windows-specific
        'fcntl',   # Unix-specific  
        'termios', # Unix-specific
        'tty',     # Unix-specific
        'pty',     # Unix-specific
        'grp',     # Unix-specific
        'pwd',     # Unix-specific
        'spwd',    # Unix-specific
        'crypt',   # Unix-specific
        'nis',     # Unix-specific
        'syslog',  # Unix-specific
        'resource' # Unix-specific (sometimes available on Windows)
    }
    
    def __init__(self, 
                 max_memory_mb: int = 512,
                 max_cpu_time_seconds: int = 180,
                 allowed_imports: Optional[List[str]] = None):
        """
        Initialize sandbox executor.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_time_seconds: Maximum CPU time in seconds
            allowed_imports: List of allowed import modules
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time_seconds = max_cpu_time_seconds
        # Remove import restrictions
        self.allowed_imports = None
        
        # Initialize OpenAI code fixer
        self.openai_fixer = None
        try:
            from core.openai_code_fixer import OpenAICodeFixer
            self.openai_fixer = OpenAICodeFixer()
            logger.info("OpenAI code fixer initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI code fixer initialization failed: {e}")
        
        # Create base sandbox template
        self.sandbox_template = self._create_sandbox_template()
        
    def execute_code(self, 
                    code: str, 
                    files: Dict[str, Any], 
                    timeout: int = 180,
                    allowed_libraries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment with intelligent error recovery.
        
        Args:
            code: Python code to execute
            files: Dictionary of files to make available
            timeout: Execution timeout in seconds
            allowed_libraries: Additional allowed libraries for this execution
            
        Returns:
            Dict with execution results
        """
        
        # Validate and normalize parameters
        if not isinstance(files, dict):
            logger.error(f"Files parameter must be a dictionary, got {type(files)}: {files}")
            if files is None:
                files = {}
            elif isinstance(files, (list, tuple)):
                files = {f"file_{i}": f for i, f in enumerate(files)}
            elif isinstance(files, (str, int)):
                logger.error(f"Cannot convert {type(files)} to dictionary: {files}")
                files = {}
            else:
                files = {}
        
        # Try execution with intelligent retry
        return self._execute_with_retry(code, files, timeout, allowed_libraries)
