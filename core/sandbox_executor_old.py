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
    
    def execute_simple(self, code: str) -> Dict[str, Any]:
        """
        Simple execution method for backward compatibility.
        Uses execute_code with default parameters.
        """
        return self.execute_code(
            code=code,
            files={},  # No additional files
            timeout=180,
            allowed_libraries=None
        )
    
    def _execute_with_retry(self, 
                          code: str, 
                          files: Dict[str, Any], 
                          timeout: int,
                          allowed_libraries: Optional[List[str]] = None,
                          max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute code with intelligent retry using LLM to fix errors.
        
        Args:
            code: Python code to execute
            files: Dictionary of files to make available
            timeout: Execution timeout
            allowed_libraries: Additional allowed libraries
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict with execution results
        """
        from core.llm_client import LLMClient
        
        original_code = code
        retry_count = 0
        error_history = []
        
        while retry_count <= max_retries:
            # Create temporary directory for execution
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Setup sandbox environment
                    self._setup_sandbox_environment(temp_dir, files, code)
                    
                    # Execute code
                    result = self._run_code_in_sandbox(temp_dir, timeout)
                    
                    if result.get("success", False):
                        # Collect artifacts (plots, files generated)
                        artifacts = self._collect_artifacts(temp_dir)
                        result["artifacts"] = artifacts
                        result["retry_count"] = retry_count
                        result["fixed_code"] = code if retry_count > 0 else None
                        return result
                    else:
                        # Code execution failed, try to fix it
                        error_msg = result.get("error", "Unknown error")
                        stderr = result.get("stderr", "")
                        
                        logger.warning(f"Execution attempt {retry_count + 1} failed: {error_msg}")
                        error_history.append({
                            "attempt": retry_count + 1,
                            "error": error_msg,
                            "stderr": stderr,
                            "code": code
                        })
                        
                        if retry_count >= max_retries:
                            logger.error(f"Max retries ({max_retries}) exceeded. All attempts failed.")
                            return self._create_failure_result(
                                original_code, code, error_history, result
                            )
                        
                        # Try to fix the error using LLM
                        fixed_code = self._fix_code_with_llm(
                            code, error_msg, stderr, error_history
                        )
                        
                        if fixed_code and fixed_code.strip() != code.strip():
                            logger.info(f"LLM suggested code fix for attempt {retry_count + 2}")
                            code = fixed_code
                        else:
                            logger.warning("LLM could not suggest a fix, retrying with original code")
                        
                        retry_count += 1
                        
                except Exception as e:
                    logger.error(f"Critical error in retry attempt {retry_count + 1}: {e}")
                    error_msg = str(e)
                    import traceback
                    stderr = traceback.format_exc()
                    
                    error_history.append({
                        "attempt": retry_count + 1,
                        "error": error_msg,
                        "stderr": stderr,
                        "code": code
                    })
                    
                    # Handle dependency errors specially
                    if self._is_dependency_error(error_msg, stderr):
                        missing_packages = self._extract_missing_packages(error_msg, stderr)
                        if missing_packages:
                            logger.info(f"Installing missing packages: {missing_packages}")
                            self._install_packages(missing_packages)
                            retry_count += 1
                            continue
                    
                    if retry_count >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded after critical error.")
                        return self._create_failure_result(
                            original_code, code, error_history, 
                            {"error": error_msg, "stderr": stderr, "success": False}
                        )
                    
                    # Try to fix the error using LLM
                    fixed_code = self._fix_code_with_llm(
                        code, error_msg, stderr, error_history
                    )
                    
                    if fixed_code and fixed_code.strip() != code.strip():
                        logger.info(f"LLM suggested code fix after critical error")
                        code = fixed_code
                    
                    retry_count += 1
        
        # Should not reach here, but just in case
        return self._create_failure_result(
            original_code, code, error_history, 
            {"error": "Max retries exceeded", "success": False}
        )
    
    def _fix_code_with_llm(self, 
                         code: str, 
                         error_msg: str, 
                         stderr: str, 
                         error_history: List[Dict]) -> Optional[str]:
        """
        Use OpenAI LLM to fix code based on error messages.
        
        Args:
            code: The failing code
            error_msg: Error message
            stderr: Standard error output
            error_history: History of previous errors and attempts
            
        Returns:
            Fixed code or None if LLM couldn't fix it
        """
        logger.info("Attempting code fix with OpenAI...")
        
        # Try OpenAI fixer first
        if self.openai_fixer:
            try:
                attempt_number = len(error_history) + 1
                context = {
                    "environment": "sandbox",
                    "timeout": self.max_cpu_time_seconds,
                    "working_dir": "temp_sandbox"
                }
                
                fixed_code, metadata = self.openai_fixer.fix_code_error(
                    code=code,
                    error_message=error_msg,
                    stderr=stderr,
                    attempt_number=attempt_number,
                    context=context
                )
                
                if fixed_code and len(fixed_code.strip()) > 10:
                    logger.info(f"OpenAI provided code fix (tokens: {metadata.get('tokens_used', 'unknown')})")
                    return fixed_code
                else:
                    logger.warning("OpenAI fix was empty or too short")
                    
            except Exception as e:
                logger.error(f"OpenAI code fixing failed: {e}")
        
        # Fallback to local LLM if OpenAI fails
        logger.info("Falling back to local LLM for code fix...")
        return self._fix_code_with_local_llm(code, error_msg, stderr, error_history)
    
    def _fix_code_with_local_llm(self, 
                               code: str, 
                               error_msg: str, 
                               stderr: str, 
                               error_history: List[Dict]) -> Optional[str]:
        """
        Fallback method using local LLM for code fixing.
        """
        try:
            from core.llm_client import LLMClient
            
            llm_client = LLMClient()
            
            # Build context for LLM
            error_context = f"""
ERROR ANALYSIS AND CODE FIXING REQUEST

Original Code:
```python
{code}
```

Current Error:
{error_msg}

Full Error Details:
{stderr}

Previous Attempts:
"""
            
            for i, attempt in enumerate(error_history):
                error_context += f"""
Attempt {attempt['attempt']}:
Error: {attempt['error']}
Code used: 
```python
{attempt['code'][:500]}{'...' if len(attempt['code']) > 500 else ''}
```
"""
            
            fix_prompt = f"""{error_context}

TASK: Fix the Python code to resolve the error. Focus on:
1. Import errors - add missing imports or install packages
2. Syntax errors - fix syntax issues
3. Runtime errors - handle exceptions and edge cases
4. Logic errors - fix algorithmic issues

REQUIREMENTS:
- Return ONLY the corrected Python code
- Do NOT include explanations or markdown
- Ensure the code is complete and executable
- Add any necessary imports at the top
- Handle edge cases that might cause similar errors

RESPONSE FORMAT: Return only the Python code, nothing else.
"""
            
            logger.info("Requesting code fix from local LLM...")
            
            # Get fixed code from LLM
            try:
                fixed_code_response = llm_client.generate(
                    prompt=fix_prompt,
                    max_tokens=2000
                )
            except Exception as e:
                logger.error(f"LLM client error: {e}")
                return None
            
            if fixed_code_response and isinstance(fixed_code_response, str):
                fixed_code = fixed_code_response.strip()
                
                # Check for error messages that might be returned instead of code
                error_indicators = [
                    'HTTPConnectionPool', 'connection', 'failed', 'unavailable',
                    'Max retries exceeded', 'NameResolutionError', 'getaddrinfo failed'
                ]
                
                if any(indicator in fixed_code for indicator in error_indicators):
                    logger.warning("LLM returned error message instead of code")
                    return None
                
                # Clean up the response (remove markdown if present)
                if fixed_code.startswith('```python'):
                    fixed_code = fixed_code.replace('```python\n', '').replace('```', '')
                elif fixed_code.startswith('```'):
                    fixed_code = fixed_code.replace('```\n', '').replace('```', '')
                
                # Validate that we got actual code
                if len(fixed_code) > 10 and ('import ' in fixed_code or 'def ' in fixed_code or '=' in fixed_code or 'print(' in fixed_code):
                    logger.info("Local LLM provided code fix")
                    return fixed_code
                else:
                    logger.warning(f"Local LLM response doesn't appear to be valid code: {fixed_code[:100]}...")
                    return None
            else:
                logger.warning("No valid response from local LLM for code fix")
                return None
                
        except Exception as e:
            logger.error(f"Error getting code fix from local LLM: {e}")
            return None
    
    def _create_failure_result(self, 
                             original_code: str, 
                             final_code: str, 
                             error_history: List[Dict], 
                             last_result: Dict) -> Dict[str, Any]:
        """Create a comprehensive failure result with all attempt details."""
        return {
            "success": False,
            "error": last_result.get("error", "Unknown error"),
            "stderr": last_result.get("stderr", ""),
            "traceback": last_result.get("traceback", ""),
            "output": None,
            "stdout": "",
            "artifacts": {},
            "retry_attempts": len(error_history),
            "error_history": error_history,
            "original_code": original_code,
            "final_attempted_code": final_code
        }
    
    def _is_dependency_error(self, error_msg: str, stderr: str) -> bool:
        """Check if error is related to missing dependencies."""
        dependency_indicators = [
            "ModuleNotFoundError",
            "ImportError", 
            "No module named",
            "Missing optional dependency",
            "cannot import name",
            "DLL load failed",
            "package not found"
        ]
        
        full_error = f"{error_msg} {stderr}".lower()
        return any(indicator.lower() in full_error for indicator in dependency_indicators)
    
    def _extract_missing_packages(self, error_msg: str, stderr: str) -> List[str]:
        """Extract package names from dependency error messages."""
        import re
        
        packages = []
        full_error = f"{error_msg} {stderr}"
        
        # Common patterns for missing packages
        patterns = [
            r"No module named '([^']+)'",
            r"ModuleNotFoundError: No module named '([^']+)'",
            r"ImportError: No module named ([^\s]+)",
            r"Missing optional dependency '([^']+)'",
            r"cannot import name '([^']+)'"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, full_error)
            packages.extend(matches)
        
        # Clean and deduplicate
        clean_packages = []
        for pkg in packages:
            # Map common package names
            pkg = pkg.strip().replace('"', '').replace("'", "")
            if pkg and pkg not in clean_packages:
                # Skip built-in modules that can't be pip installed
                if pkg in self.BUILTIN_MODULES:
                    logger.debug(f"Skipping built-in module in error detection: {pkg}")
                    continue
                    
                # Map to installable names
                mapped_pkg = self.PACKAGE_MAPPING.get(pkg.lower(), pkg)
                if mapped_pkg not in clean_packages:
                    clean_packages.append(mapped_pkg)
        
        if clean_packages:
            logger.info(f"Detected missing packages: {clean_packages}")
        
        return clean_packages

    def _setup_sandbox_environment(self, temp_dir: str, files: Dict[str, Any], code: str):
        """Setup the sandbox environment with files and code."""
        
        logger.info(f"Setting up sandbox environment in: {temp_dir}")
        logger.info(f"Files to process: {len(files)} items")
        logger.info(f"Code length: {len(code)} characters")
        
        # Log file manifest structure for debugging
        for filename, file_info in files.items():
            logger.debug(f"File manifest entry: {filename} -> {type(file_info)}: {repr(file_info)[:100]}")
        
        # Copy uploaded files to sandbox
        files_copied = 0
        for filename, file_info in files.items():
            # Ensure file_info is a dictionary
            if not isinstance(file_info, dict):
                logger.warning(f"Skipping file {filename}: file_info is not a dict, got {type(file_info)}")
                continue
                
            if "path" in file_info:
                src_path = file_info["path"]
                dst_path = os.path.join(temp_dir, filename)
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_size = os.path.getsize(dst_path)
                    logger.info(f"Copied file {filename} to sandbox ({copied_size} bytes)")
                    files_copied += 1
                except Exception as e:
                    logger.warning(f"Failed to copy file {filename}: {e}")
                    continue
            else:
                logger.warning(f"Skipping file {filename}: no 'path' key in file_info")
        
        logger.info(f"Successfully copied {files_copied} files to sandbox")
        
        # Create the main execution script
        logger.info("Creating main.py execution script...")
        main_script = self._wrap_code_with_safety(code)
        
        main_py_path = os.path.join(temp_dir, "main.py")
        try:
            with open(main_py_path, "w", encoding="utf-8") as f:
                f.write(main_script)
            
            # Verify the file was written correctly
            written_size = os.path.getsize(main_py_path)
            logger.info(f"Created main.py ({written_size} bytes)")
            
            # Log a preview of the generated script for debugging
            logger.debug(f"main.py preview: {main_script[:500]}...")
            
        except Exception as e:
            logger.error(f"Failed to create main.py: {e}")
            raise
        
        # Create requirements.txt if needed
        logger.info("Creating requirements.txt...")
        try:
            self._create_requirements_file(temp_dir, code)
            req_path = os.path.join(temp_dir, "requirements.txt")
            if os.path.exists(req_path):
                req_size = os.path.getsize(req_path)
                logger.info(f"Created requirements.txt ({req_size} bytes)")
                
                # Log requirements content
                with open(req_path, "r", encoding="utf-8") as f:
                    req_content = f.read()
                    if req_content.strip():
                        logger.info(f"Requirements content: {req_content.strip()}")
                    else:
                        logger.info("Requirements file is empty")
            else:
                logger.info("No requirements.txt created (no dependencies detected)")
                
        except Exception as e:
            logger.warning(f"Failed to create requirements.txt: {e}")
        
        # List all files in sandbox for final verification
        try:
            sandbox_files = os.listdir(temp_dir)
            logger.info(f"Final sandbox contents: {sandbox_files}")
            
            # Log file sizes
            for filename in sandbox_files:
                filepath = os.path.join(temp_dir, filename)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    logger.debug(f"  {filename}: {size} bytes")
                    
        except Exception as e:
            logger.warning(f"Failed to list sandbox contents: {e}")
    
    def _wrap_code_with_safety(self, code: str) -> str:
        """Wrap user code with safety measures and output capture."""
        
        wrapped_code = f"""
import sys
import json
import traceback
import signal
import os
import platform
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Try importing resource (only works on Linux/Mac)
try:
    import resource
except ImportError:
    resource = None

# Set resource limits
def set_limits():
    if resource:
        # Memory limit (in bytes) - ensure integer values
        memory_limit = int({self.max_memory_mb * 1024 * 1024})
        cpu_limit = int({self.max_cpu_time_seconds})
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        # CPU time limit (in seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

# Download size cap (1GB)
MAX_DOWNLOAD_SIZE = 1024 * 1024 * 1024  # 1GB

def safe_download(url, *args, **kwargs):
    import requests
    resp = requests.get(url, stream=True, *args, **kwargs)
    total = 0
    chunks = []
    for chunk in resp.iter_content(8192):
        total += len(chunk)
        if total > MAX_DOWNLOAD_SIZE:
            raise Exception(f"Download size exceeded 1GB limit for URL: {{url}}")
        chunks.append(chunk)
    return b"".join(chunks)

# Auto-install missing packages function
def install_missing_package(package_name):
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--quiet'])
        return True
    except:
        return False

# Patch requests.get to use safe_download  
try:
    import requests
    original_get = requests.get
    def patched_get(url, *args, **kwargs):
        if 'stream' in kwargs and kwargs['stream']:
            return original_get(url, *args, **kwargs)
        content = safe_download(url, *args, **kwargs)
        
        # Create a more complete response object
        class MockResponse:
            def __init__(self, content, url):
                self.content = content
                self.text = content.decode('utf-8', errors='ignore')
                self.status_code = 200
                self.url = url
                self.headers = {{'content-type': 'text/html'}}
                self.encoding = 'utf-8'
            
            def json(self):
                import json
                return json.loads(self.text)
            
            def raise_for_status(self):
                pass  # Assume success for downloaded content
        
        return MockResponse(content, url)
    requests.get = patched_get
except Exception:
    pass

# Override import to auto-install missing packages
# Handle __builtins__ being either a dict or module
if isinstance(__builtins__, dict):
    original_import = __builtins__['__import__']
else:
    original_import = __builtins__.__import__

def auto_install_import(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return original_import(name, globals, locals, fromlist, level)
    except ImportError as e:
        # Built-in modules that should not be installed
        builtin_modules = {{
            'sys', 'os', 'json', 'csv', 're', 'math', 'datetime', 'time', 'io', 'base64', 
            'collections', 'traceback', 'signal', 'platform', 'subprocess', 'threading', 
            'multiprocessing', 'urllib', 'http', 'email', 'html', 'xml', 'sqlite3',
            'hashlib', 'hmac', 'secrets', 'uuid', 'pickle', 'copy', 'itertools', 'functools',
            'operator', 'pathlib', 'tempfile', 'shutil', 'glob', 'fnmatch', 'linecache',
            'textwrap', 'string', 'struct', 'codecs', 'locale', 'calendar', 'zoneinfo',
            'decimal', 'fractions', 'statistics', 'random', 'bisect', 'heapq', 'array',
            'weakref', 'types', 'enum', 'contextlib', 'abc', 'numbers', 'cmath', 'logging',
            'argparse', 'getopt', 'configparser', 'fileinput', 'readline', 'rlcompleter',
            'msvcrt', 'fcntl', 'termios', 'tty', 'pty', 'grp', 'pwd', 'spwd', 'crypt', 
            'nis', 'syslog', 'resource'
        }}
        
        # Skip built-in modules
        if name in builtin_modules:
            print(f"Skipping built-in module: {{name}}")
            raise e
        
        # Common package mappings for auto-install
        package_map = {{
            'bs4': 'beautifulsoup4',
            'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn'
        }}
        
        package_to_install = package_map.get(name, name)
        print(f"Missing package {{name}}, attempting to install {{package_to_install}}...")
        
        if install_missing_package(package_to_install):
            print(f"Successfully installed {{package_to_install}}")
            return original_import(name, globals, locals, fromlist, level)
        else:
            print(f"Failed to install {{package_to_install}}")
            raise e

# Set the modified import function
if isinstance(__builtins__, dict):
    __builtins__['__import__'] = auto_install_import
else:
    __builtins__.__import__ = auto_install_import

def main():
    try:
        # Set limits if supported
        set_limits()

        # Set timeout alarm (Unix only)
        if platform.system() != "Windows":
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int({self.max_cpu_time_seconds}))

        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Redirect output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # User code starts here
            exec('''{code}''')

        # Get captured output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()

        # Clear alarm (if set)
        if platform.system() != "Windows":
            signal.alarm(0)

        # Write results to files
        with open("stdout.txt", "w", encoding="utf-8", errors="replace") as f:
            f.write(stdout_content)

        with open("stderr.txt", "w", encoding="utf-8", errors="replace") as f:
            f.write(stderr_content)

        with open("success.txt", "w", encoding="utf-8") as f:
            f.write("true")

        # Print final stdout for capture
        print(stdout_content, end="")

    except Exception as e:
        error_info = {{
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }}

        with open("error.json", "w", encoding="utf-8", errors="replace") as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)

        with open("success.txt", "w", encoding="utf-8") as f:
            f.write("false")

        print(f"ERROR: {{str(e)}}", file=sys.stderr)

if __name__ == "__main__":
    main()
"""
        return wrapped_code

    def _validate_imports(self, code: str, additional_allowed: Optional[List[str]] = None) -> Dict[str, Any]:
        """Validate that only allowed imports are used."""
        
        # Defensive: ensure self.allowed_imports is always a list
        if not isinstance(self.allowed_imports, (list, tuple, set)):
            logger.error(f"self.allowed_imports must be iterable, got {type(self.allowed_imports)}: {self.allowed_imports}")
            allowed_imports = []
        else:
            allowed_imports = self.allowed_imports
        allowed = set(allowed_imports)
        if additional_allowed:
            # Ensure additional_allowed is iterable
            if isinstance(additional_allowed, str):
                additional_allowed = [additional_allowed]
            elif not isinstance(additional_allowed, (list, tuple, set)):
                logger.error(f"additional_allowed must be iterable, got {type(additional_allowed)}: {additional_allowed}")
                additional_allowed = []
            allowed.update(additional_allowed)
        
        # Extract import statements
        import re
        import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
        
        unauthorized_imports = []
        
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                match = re.match(import_pattern, line)
                if match:
                    module = match.group(1) or match.group(2)
                    # Get base module name (before any dots)
                    base_module = module.split('.')[0]
                    
                    if base_module not in allowed:
                        unauthorized_imports.append(base_module)
        
        return {
            "valid": len(unauthorized_imports) == 0,
            "unauthorized": list(set(unauthorized_imports)),
            "error": f"Unauthorized imports found: {', '.join(set(unauthorized_imports))}" if unauthorized_imports else None
        }
    
    def _run_code_in_sandbox(self, temp_dir: str, timeout: int) -> Dict[str, Any]:
        """Run the code in the sandbox and capture results."""
        
        logger.info(f"Starting sandbox execution in {temp_dir} with timeout {timeout}s")
        
        try:
            # Change to sandbox directory
            original_cwd = os.getcwd()
            logger.info(f"Changing directory from {original_cwd} to {temp_dir}")
            os.chdir(temp_dir)
            
            # Check if main.py exists and log its size
            main_py_path = os.path.join(temp_dir, "main.py")
            if os.path.exists(main_py_path):
                file_size = os.path.getsize(main_py_path)
                logger.info(f"main.py exists, size: {file_size} bytes")
                # Log first few lines for debugging
                with open(main_py_path, "r", encoding="utf-8", errors="replace") as f:
                    first_lines = f.read(500)
                    logger.debug(f"main.py preview: {repr(first_lines)}")
            else:
                logger.error("main.py not found!")
                return {
                    "success": False,
                    "error": "main.py not found",
                    "output": None,
                    "stdout": "",
                    "stderr": "main.py not found in sandbox",
                    "return_code": -1
                }
            
            # Set up environment variables for security
            env = os.environ.copy()
            env['PYTHONPATH'] = temp_dir
            env['HOME'] = temp_dir
            logger.info(f"Environment setup: PYTHONPATH={temp_dir}, HOME={temp_dir}")
            
            # Log Python executable being used
            logger.info(f"Using Python executable: {sys.executable}")
            
            # Run the code
            logger.info("Starting subprocess execution...")
            if platform.system() == "Windows":
                # Windows doesn't support os.setsid
                process = subprocess.Popen(
                    [sys.executable, "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=temp_dir
                )
                logger.info(f"Started Windows subprocess with PID: {process.pid}")
            else:
                process = subprocess.Popen(
                    [sys.executable, "main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=temp_dir,
                    preexec_fn=os.setsid  # Create new process group
                )
                logger.info(f"Started Unix subprocess with PID: {process.pid}")
            
            try:
                logger.info(f"Waiting for process completion (timeout: {timeout}s)...")
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
                logger.info(f"Process completed with return code: {return_code}")
                logger.info(f"Raw stdout length: {len(stdout)} chars")
                logger.info(f"Raw stderr length: {len(stderr)} chars")
                
                if stdout:
                    logger.debug(f"Raw stdout preview: {repr(stdout[:300])}")
                if stderr:
                    logger.warning(f"Raw stderr preview: {repr(stderr[:300])}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Process timed out after {timeout}s, terminating...")
                # Kill the process group (Unix) or terminate process (Windows)
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    logger.info("Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate gracefully, killing...")
                    if platform.system() == "Windows":
                        process.kill()
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    stdout, stderr = process.communicate()
                
                return {
                    "success": False,
                    "error": f"Execution timed out after {timeout} seconds",
                    "output": None,
                    "stdout": stdout or "",
                    "stderr": stderr or "Execution timed out",
                    "return_code": -1
                }
            
            finally:
                os.chdir(original_cwd)
                logger.info(f"Restored working directory to {original_cwd}")
            
            # Check for output files and log their existence
            output_files = ["success.txt", "stdout.txt", "stderr.txt", "error.json"]
            files_found = []
            files_missing = []
            
            for filename in output_files:
                filepath = os.path.join(temp_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    files_found.append(f"{filename}({size}b)")
                else:
                    files_missing.append(filename)
            
            logger.info(f"Output files found: {files_found}")
            if files_missing:
                logger.warning(f"Output files missing: {files_missing}")
            
            # Check if execution was successful
            success_file = os.path.join(temp_dir, "success.txt")
            success = False
            
            if os.path.exists(success_file):
                with open(success_file, "r", encoding="utf-8", errors="replace") as f:
                    success_content = f.read().strip()
                    success = success_content == "true"
                    logger.info(f"success.txt content: {repr(success_content)} -> success={success}")
            else:
                logger.warning("success.txt not found")
            
            # Read captured output files
            stdout_file = os.path.join(temp_dir, "stdout.txt")
            stderr_file = os.path.join(temp_dir, "stderr.txt")
            error_file = os.path.join(temp_dir, "error.json")
            
            captured_stdout = ""
            captured_stderr = ""
            error_details = None
            
            if os.path.exists(stdout_file):
                with open(stdout_file, "r", encoding="utf-8", errors="replace") as f:
                    captured_stdout = f.read()
                    logger.info(f"Captured stdout from file: {len(captured_stdout)} chars")
                    if captured_stdout:
                        logger.debug(f"Captured stdout preview: {repr(captured_stdout[:300])}")
            else:
                logger.warning("stdout.txt not found, using raw subprocess stdout")
                captured_stdout = stdout or ""
            
            if os.path.exists(stderr_file):
                with open(stderr_file, "r", encoding="utf-8", errors="replace") as f:
                    captured_stderr = f.read()
                    logger.info(f"Captured stderr from file: {len(captured_stderr)} chars")
                    if captured_stderr:
                        logger.warning(f"Captured stderr preview: {repr(captured_stderr[:300])}")
            else:
                logger.warning("stderr.txt not found, using raw subprocess stderr")
                captured_stderr = stderr or ""
            
            if os.path.exists(error_file):
                with open(error_file, "r", encoding="utf-8", errors="replace") as f:
                    error_details = json.load(f)
                    logger.error(f"Error details from file: {error_details}")
            
            # Determine final output
            output = captured_stdout.strip() if captured_stdout.strip() else None
            
            # Try to parse JSON output
            if output:
                try:
                    parsed_output = json.loads(output)
                    logger.info(f"Successfully parsed output as JSON: {type(parsed_output)}")
                    output = parsed_output
                except json.JSONDecodeError:
                    logger.info("Output is not valid JSON, keeping as string")
            
            # Final result compilation
            result = {
                "success": success and return_code == 0,
                "error": error_details["error"] if error_details else None,
                "output": output,
                "stdout": captured_stdout,
                "stderr": captured_stderr,
                "return_code": return_code
            }
            
            logger.info(f"Final result: success={result['success']}, error={repr(result['error'])}, output_type={type(result['output'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in sandbox execution: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "output": None,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
    
    def _collect_artifacts(self, temp_dir: str) -> Dict[str, Any]:
        """Collect any artifacts (plots, files) generated during execution."""
        
        artifacts = {
            "plots": [],
            "files": []
        }
        
        # Look for common plot file extensions
        plot_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
        
        for file_path in Path(temp_dir).iterdir():
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                # Skip system files
                if file_path.name in ['main.py', 'stdout.txt', 'stderr.txt', 'success.txt', 'error.json']:
                    continue
                
                if file_ext in plot_extensions:
                    # Read plot file
                    try:
                        with open(file_path, 'rb') as f:
                            artifacts["plots"].append({
                                "filename": file_path.name,
                                "size": file_path.stat().st_size,
                                "type": file_ext[1:]  # Remove the dot
                            })
                        logger.info(f"Collected plot artifact: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to read plot file {file_path}: {e}")
                
                else:
                    # Other generated files
                    artifacts["files"].append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "type": file_ext[1:] if file_ext else "unknown"
                    })
        
        return artifacts
    
    def _create_requirements_file(self, temp_dir: str, code: str):
        """Create requirements.txt based on detected imports and auto-install if needed."""
        
        # Map import names to package names
        package_mapping = {
            'pandas': 'pandas',
            'numpy': 'numpy', 
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'requests': 'requests',
            'bs4': 'beautifulsoup4',
            'beautifulsoup4': 'beautifulsoup4',
            'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'scipy': 'scipy',
            'lxml': 'lxml',
            'html5lib': 'html5lib',
            'openpyxl': 'openpyxl',
            'xlrd': 'xlrd',
            'sklearn': 'scikit-learn',
            'plotly': 'plotly',
            'dash': 'dash',
            'streamlit': 'streamlit',
            'flask': 'flask',
            'django': 'django',
            'fastapi': 'fastapi',
            'sqlalchemy': 'sqlalchemy',
            'pymongo': 'pymongo',
            'psycopg2': 'psycopg2-binary',
            'mysql': 'mysql-connector-python',
            'redis': 'redis',
            'celery': 'celery',
            'boto3': 'boto3',
            'azure': 'azure-storage-blob',
            'google': 'google-cloud-storage',
            'tweepy': 'tweepy',
            'selenium': 'selenium',
            'scrapy': 'scrapy'
        }
        
        # Extract imports from code
        imports = set()
        for line in code.split('\n'):
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('import '):
                # Handle "import module" and "import module as alias"
                import_part = line.replace('import ', '').split('#')[0].strip()  # Remove comments
                modules = import_part.split(',')
                for module in modules:
                    module = module.split('.')[0].split(' as ')[0].strip()
                    if module:
                        imports.add(module)
            elif line.startswith('from '):
                # Handle "from module import something"
                import_part = line.split('#')[0].strip()  # Remove comments
                if ' import ' in import_part:
                    module = import_part.split(' ')[1].split('.')[0].strip()
                    if module:
                        imports.add(module)
        
        # Create requirements.txt with auto-install
        requirements = []
        for imp in imports:
            # Skip built-in modules
            if imp in self.BUILTIN_MODULES:
                logger.debug(f"Skipping built-in module: {imp}")
                continue
                
            if imp in package_mapping:
                req = package_mapping[imp]
                if req not in requirements:
                    requirements.append(req)
                    logger.debug(f"Mapped import '{imp}' to package '{req}'")
            elif len(imp) > 1:  # Avoid single-letter imports
                # For unknown packages, try to install them directly
                if imp not in requirements:
                    requirements.append(imp)
                    logger.debug(f"Adding unknown package for installation: {imp}")
        
        if requirements:
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, "w") as f:
                f.write('\n'.join(requirements))
            
            # Auto-install packages
            self._install_packages(requirements, temp_dir)
    
    def _install_packages(self, requirements: List[str], temp_dir: str):
        """Dynamically install packages in the sandbox."""
        import subprocess
        
        logger.info(f"Installing {len(requirements)} packages: {requirements}")
        
        for package in requirements:
            # Skip built-in modules that can't be pip installed
            if package in self.BUILTIN_MODULES:
                logger.debug(f"Skipping built-in module: {package}")
                continue
                
            try:
                logger.info(f"Installing package: {package}")
                start_time = time.time()
                
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package, '--quiet', '--no-warn-script-location'
                ], capture_output=True, text=True, timeout=60, cwd=temp_dir)
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed: {package} (took {duration:.1f}s)")
                    
                    # Verify installation by trying to import
                    try:
                        # For certain packages, we need to import with different names
                        import_names = {
                            'beautifulsoup4': 'bs4',
                            'pillow': 'PIL', 
                            'opencv-python': 'cv2',
                            'scikit-learn': 'sklearn',
                            'pyyaml': 'yaml',
                            'pyserial': 'serial',
                            'psycopg2-binary': 'psycopg2',
                            'mysql-connector-python': 'mysql.connector',
                            'python-dateutil': 'dateutil',
                            'python-magic': 'magic',
                            'pycryptodome': 'Crypto',
                            'pyjwt': 'jwt',
                            'python-dotenv': 'dotenv'
                        }
                        
                        # Use the correct import name for testing
                        test_import = import_names.get(package, package)
                        
                        test_result = subprocess.run([
                            sys.executable, '-c', f'import {test_import}; print(f"{test_import} imported successfully")'
                        ], capture_output=True, text=True, timeout=10)
                        
                        if test_result.returncode == 0:
                            logger.debug(f"Verified {package} import as '{test_import}': {test_result.stdout.strip()}")
                        else:
                            logger.debug(f"Package {package} installed but import test for '{test_import}' had issues (this may be normal): {test_result.stderr}")
                            
                    except Exception as verify_error:
                        logger.debug(f"Could not verify {package} installation: {verify_error}")
                        
                else:
                    logger.warning(f"Failed to install {package} (took {duration:.1f}s)")
                    logger.warning(f"Return code: {result.returncode}")
                    logger.warning(f"Stderr: {result.stderr.strip()}")
                    if result.stdout.strip():
                        logger.warning(f"Stdout: {result.stdout.strip()}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout installing {package} (60s limit exceeded)")
            except Exception as e:
                logger.error(f"Error installing {package}: {e}", exc_info=True)
    
    def _create_sandbox_template(self) -> str:
        """Create a template for the sandbox environment."""
        
        template = """
# Sandbox Environment Template
# This template provides a secure execution environment for user code

import sys
import os

# Restrict certain dangerous operations
import builtins

# Override dangerous built-ins
original_open = builtins.open
original_exec = builtins.exec
original_eval = builtins.eval

def safe_open(file, mode='r', **kwargs):
    # Restrict file operations to current directory
    if os.path.isabs(file):
        raise PermissionError("Absolute paths not allowed")
    if '..' in file:
        raise PermissionError("Path traversal not allowed")
    return original_open(file, mode, **kwargs)

# Replace built-ins
builtins.open = safe_open

print("Sandbox environment initialized")
"""
        return template

# Example usage for testing
if __name__ == "__main__":
    executor = SandboxExecutor()
    
    test_code = '''
import pandas as pd
import json

# Create sample data
data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
df = pd.DataFrame(data)

# Analyze data
result = {
    "mean_age": df["age"].mean(),
    "count": len(df),
    "summary": df.describe().to_dict()
}

print(json.dumps(result, indent=2))
'''
    
    result = executor.execute_code(test_code, {})
    print("Execution result:", result)
