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
        Use LLM to fix code based on error messages.
        
        Args:
            code: The failing code
            error_msg: Error message
            stderr: Standard error output
            error_history: History of previous errors and attempts
            
        Returns:
            Fixed code or None if LLM couldn't fix it
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
            
            logger.info("Requesting code fix from LLM...")
            
            # Get fixed code from LLM
            fixed_code_response = llm_client.generate(
                prompt=fix_prompt,
                max_tokens=2000
            )
            
            if fixed_code_response and isinstance(fixed_code_response, str):
                fixed_code = fixed_code_response.strip()
                
                # Clean up the response (remove markdown if present)
                if fixed_code.startswith('```python'):
                    fixed_code = fixed_code.replace('```python\n', '').replace('```', '')
                elif fixed_code.startswith('```'):
                    fixed_code = fixed_code.replace('```\n', '').replace('```', '')
                
                # Validate that we got actual code
                if len(fixed_code) > 10 and ('import ' in fixed_code or 'def ' in fixed_code or '=' in fixed_code):
                    logger.info("LLM provided code fix")
                    return fixed_code
                else:
                    logger.warning("LLM response doesn't appear to be valid code")
                    return None
            else:
                logger.warning("No valid response from LLM for code fix")
                return None
                
        except Exception as e:
            logger.error(f"Error getting code fix from LLM: {e}")
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
                # Map to installable names
                mapped_pkg = self.PACKAGE_MAPPING.get(pkg.lower(), pkg)
                if mapped_pkg not in clean_packages:
                    clean_packages.append(mapped_pkg)
        
        return clean_packages

    def _setup_sandbox_environment(self, temp_dir: str, files: Dict[str, Any], code: str):
        
        # Copy uploaded files to sandbox
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
                    logger.info(f"Copied file {filename} to sandbox")
                except Exception as e:
                    logger.warning(f"Failed to copy file {filename}: {e}")
                    continue
        
        # Create the main execution script
        main_script = self._wrap_code_with_safety(code)
        
        with open(os.path.join(temp_dir, "main.py"), "w", encoding="utf-8") as f:
            f.write(main_script)
        
        # Create requirements.txt if needed
        self._create_requirements_file(temp_dir, code)
    
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
        response_obj = type('Response', (), {{'content': content, 'text': content.decode('utf-8', errors='ignore')}})()
        return response_obj
    requests.get = patched_get
except Exception:
    pass

# Override import to auto-install missing packages
original_import = __builtins__['__import__']

def auto_install_import(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return original_import(name, globals, locals, fromlist, level)
    except ImportError as e:
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

__builtins__['__import__'] = auto_install_import

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
        
        try:
            # Change to sandbox directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            # Set up environment variables for security
            env = os.environ.copy()
            env['PYTHONPATH'] = temp_dir
            env['HOME'] = temp_dir
            
            # Run the code
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
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                # Kill the process group (Unix) or terminate process (Windows)
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
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
                    "stderr": stderr or "Execution timed out"
                }
            
            finally:
                os.chdir(original_cwd)
            
            # Check if execution was successful
            success_file = os.path.join(temp_dir, "success.txt")
            success = False
            
            if os.path.exists(success_file):
                with open(success_file, "r", encoding="utf-8", errors="replace") as f:
                    success = f.read().strip() == "true"
            
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
            
            if os.path.exists(stderr_file):
                with open(stderr_file, "r", encoding="utf-8", errors="replace") as f:
                    captured_stderr = f.read()
            
            if os.path.exists(error_file):
                with open(error_file, "r", encoding="utf-8", errors="replace") as f:
                    error_details = json.load(f)
            
            # Determine final output
            output = captured_stdout.strip() if captured_stdout.strip() else None
            
            # Try to parse JSON output
            if output:
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    pass  # Keep as string
            
            return {
                "success": success and return_code == 0,
                "error": error_details["error"] if error_details else None,
                "output": output,
                "stdout": captured_stdout,
                "stderr": captured_stderr,
                "return_code": return_code
            }
            
        except Exception as e:
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
            if line.startswith('import '):
                module = line.replace('import ', '').split('.')[0].split(' as ')[0].split(',')[0].strip()
                imports.add(module)
            elif line.startswith('from '):
                module = line.split(' ')[1].split('.')[0]
                imports.add(module)
        
        # Create requirements.txt with auto-install
        requirements = []
        for imp in imports:
            if imp in package_mapping:
                requirements.append(package_mapping[imp])
            elif imp not in ['sys', 'os', 'json', 'csv', 're', 'math', 'datetime', 'time', 'io', 'base64', 'collections', 'traceback', 'signal', 'platform']:
                # For unknown packages, try to install them directly
                requirements.append(imp)
        
        if requirements:
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, "w") as f:
                f.write('\n'.join(requirements))
            
            # Auto-install packages
            self._install_packages(requirements, temp_dir)
    
    def _install_packages(self, requirements: List[str], temp_dir: str):
        """Dynamically install packages in the sandbox."""
        import subprocess
        
        for package in requirements:
            try:
                logger.info(f"Installing package: {package}")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package, '--quiet', '--no-warn-script-location'
                ], capture_output=True, text=True, timeout=60, cwd=temp_dir)
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed: {package}")
                else:
                    logger.warning(f"Failed to install {package}: {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout installing {package}")
            except Exception as e:
                logger.error(f"Error installing {package}: {e}")
    
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
