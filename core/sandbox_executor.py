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

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy and pandas data types"""
    def default(self, obj):
        # Handle NumPy types
        if hasattr(obj, 'dtype'):  # NumPy arrays and scalars
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
        
        # Handle pandas types
        if hasattr(obj, 'to_dict'):  # DataFrames, Series
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # pandas scalars
            return obj.item()
            
        # Handle other common types
        if str(type(obj)).startswith('<class \'numpy.'):
            # Fallback for any numpy type we missed
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
                
        return super().default(obj)

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
        Execute Python code in a sandboxed environment.
        
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
        
        # Create temporary directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Setup sandbox environment
                self._setup_sandbox_environment(temp_dir, files, code)
                
                # No import validation - allow all imports
                
                # Execute code
                result = self._run_code_in_sandbox(temp_dir, timeout)
                
                # Collect artifacts (plots, files generated)
                artifacts = self._collect_artifacts(temp_dir)
                result["artifacts"] = artifacts
                
                return result
                
            except Exception as e:
                logger.error(f"Sandbox execution failed: {e}")
                logger.error(f"Exception type: {type(e)}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error(f"Type of files: {type(files)}, value: {files}")
                logger.error(f"Type of allowed_libraries: {type(allowed_libraries)}, value: {allowed_libraries}")
                logger.error(f"Code to execute:\n{code}")
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "code": code,
                    "files_type": str(type(files)),
                    "files_value": str(files),
                    "allowed_libraries_type": str(type(allowed_libraries)),
                    "allowed_libraries_value": str(allowed_libraries),
                    "output": None,
                    "stdout": "",
                    "stderr": str(e),
                    "artifacts": {}
                }
    
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
    
    def _setup_sandbox_environment(self, temp_dir: str, files: Dict[str, Any], code: str):
        """Setup the sandbox environment with files and code."""
        
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
        
        with open(os.path.join(temp_dir, "main.py"), "w") as f:
            f.write(main_script)
        
        # Create requirements.txt if needed
        self._create_requirements_file(temp_dir, code)
        
        # Install required packages
        self._install_requirements(temp_dir)
        
        # Verify environment consistency to prevent mismatches
        self._verify_environment_consistency(temp_dir)
    
    def _wrap_code_with_safety(self, code: str) -> str:
        """Wrap user code with safety measures and output capture."""
        
        # Auto-convert json.dumps calls to use safe encoder
        safe_code = code.replace('json.dumps(', 'safe_json_dumps(')
        
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

# Custom JSON encoder for NumPy/pandas compatibility
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NumPy types
        if hasattr(obj, 'dtype'):  # NumPy arrays and scalars
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
        
        # Handle pandas types
        if hasattr(obj, 'to_dict'):  # DataFrames, Series
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # pandas scalars
            return obj.item()
            
        # Handle other common types
        if str(type(obj)).startswith('<class \\'numpy.'):
            # Fallback for any numpy type we missed
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
                
        return super().default(obj)

# Helper function for safe JSON serialization
def safe_json_dumps(obj, **kwargs):
    \"\"\"JSON dumps that handles NumPy/pandas types automatically\"\"\"
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)

# Set resource limits
def set_limits():
    if resource:
        # Memory limit (in bytes)
        resource.setrlimit(resource.RLIMIT_AS, ({self.max_memory_mb * 1024 * 1024}, {self.max_memory_mb * 1024 * 1024}))
        # CPU time limit (in seconds)
        resource.setrlimit(resource.RLIMIT_CPU, ({self.max_cpu_time_seconds}, {self.max_cpu_time_seconds}))

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

def main():
    try:
        # Set limits if supported
        set_limits()

        # Set timeout alarm (Unix only)
        if platform.system() != "Windows":
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm({self.max_cpu_time_seconds})

        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        # Redirect output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # User code starts here (with safe JSON handling)
            exec('''{safe_code}''')

        # Get captured output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()

        # Clear alarm (if set)
        if platform.system() != "Windows":
            signal.alarm(0)

        # Write results to files
        with open("stdout.txt", "w") as f:
            f.write(stdout_content)

        with open("stderr.txt", "w") as f:
            f.write(stderr_content)

        with open("success.txt", "w") as f:
            f.write("true")

        # Print final stdout for capture
        print(stdout_content, end="")

    except Exception as e:
        error_info = {{
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }}

        with open("error.json", "w") as f:
            json.dump(error_info, f, indent=2)

        with open("success.txt", "w") as f:
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
            
            # Set up environment variables for security AND package discovery
            env = os.environ.copy()
            # CRITICAL: Put sandbox directory FIRST in Python path so locally installed packages are found first
            # Use appropriate path separator for the OS
            path_separator = ";" if platform.system() == "Windows" else ":"
            current_pythonpath = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = temp_dir + (path_separator + current_pythonpath if current_pythonpath else '')
            env['HOME'] = temp_dir
            
            # Log environment details to ensure consistency
            logger.debug(f"Execution environment: Python={sys.executable}, CWD={temp_dir}")
            logger.debug(f"PYTHONPATH={env.get('PYTHONPATH', 'Not set')}")
            
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
                with open(success_file, "r") as f:
                    success = f.read().strip() == "true"
            
            # Read captured output files
            stdout_file = os.path.join(temp_dir, "stdout.txt")
            stderr_file = os.path.join(temp_dir, "stderr.txt")
            error_file = os.path.join(temp_dir, "error.json")
            
            captured_stdout = ""
            captured_stderr = ""
            error_details = None
            
            if os.path.exists(stdout_file):
                with open(stdout_file, "r") as f:
                    captured_stdout = f.read()
            
            if os.path.exists(stderr_file):
                with open(stderr_file, "r") as f:
                    captured_stderr = f.read()
            
            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    error_details = json.load(f)
            
            # Add debugging output
            logger.debug(f"🔍 DEBUGGING SANDBOX OUTPUT:")
            logger.debug(f"🔍 Success file exists: {os.path.exists(success_file)}")
            logger.debug(f"🔍 Success value: {success}")
            logger.debug(f"🔍 Return code: {return_code}")
            logger.debug(f"🔍 Raw captured_stdout: {repr(captured_stdout)}")
            logger.debug(f"🔍 Raw captured_stderr: {repr(captured_stderr)}")
            if error_details:
                logger.debug(f"🔍 Error details: {error_details}")
            
            # Determine final output
            output = captured_stdout.strip() if captured_stdout.strip() else None
            
            # Try to parse JSON output
            if output:
                try:
                    parsed_output = json.loads(output)
                    logger.debug(f"🔍 JSON parsing successful: {type(parsed_output)}")
                    logger.debug(f"🔍 Parsed JSON keys: {list(parsed_output.keys()) if isinstance(parsed_output, dict) else 'Not a dict'}")
                    output = parsed_output
                except json.JSONDecodeError as json_err:
                    logger.debug(f"🔍 JSON parsing failed: {json_err}")
                    logger.debug(f"🔍 Keeping output as string")
                    pass  # Keep as string
            else:
                logger.debug(f"🔍 No output to parse")
            
            logger.debug(f"🔍 Final processed output: {repr(output)}")
            logger.debug(f"🔍 Final output type: {type(output)}")
            
            final_result = {
                "success": success and return_code == 0,
                "error": error_details["error"] if error_details else None,
                "output": output,
                "stdout": captured_stdout,
                "stderr": captured_stderr,
                "return_code": return_code
            }
            
            logger.debug(f"🔍 Final sandbox result: success={final_result['success']}, has_output={bool(final_result['output'])}")
            
            return final_result
            
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
        """Create requirements.txt based on detected imports."""
        
        # Map import names to package names
        package_mapping = {
            # Core data science libraries
            'pandas': 'pandas',
            'numpy': 'numpy', 
            'scipy': 'scipy',
            'statsmodels': 'statsmodels',
            'sklearn': 'scikit-learn',
            'scikit-learn': 'scikit-learn',
            
            # Visualization libraries
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'folium': 'folium',
            
            # Web scraping and HTTP
            'requests': 'requests',
            'bs4': 'beautifulsoup4',
            'beautifulsoup4': 'beautifulsoup4',
            'lxml': 'lxml',
            'html5lib': 'html5lib',
            
            # Image processing
            'PIL': 'Pillow',
            'cv2': 'opencv-python',
            'skimage': 'scikit-image',
            
            # Network analysis
            'networkx': 'networkx',
            
            # Specialized data tools
            'yfinance': 'yfinance',
            'tweepy': 'tweepy',
            'praw': 'praw',
            
            # Excel and file handling
            'openpyxl': 'openpyxl',
            'xlrd': 'xlrd',
            'xlsxwriter': 'xlsxwriter',
            
            # Web frameworks and APIs
            'dash': 'dash',
            'streamlit': 'streamlit',
            'flask': 'flask',
            'fastapi': 'fastapi',
            
            # Database
            'sqlalchemy': 'sqlalchemy',
            'psycopg2': 'psycopg2-binary'
        }
        
        # Extract imports from code
        imports = set()
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                module = line.replace('import ', '').split('.')[0].split(' as ')[0]
                imports.add(module)
            elif line.startswith('from '):
                module = line.split(' ')[1].split('.')[0]
                imports.add(module)
        
        # Create requirements.txt
        requirements = []
        for imp in imports:
            if imp in package_mapping:
                requirements.append(package_mapping[imp])
        
        if requirements:
            requirements_path = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_path, "w") as f:
                f.write('\n'.join(requirements))
    
    def _install_requirements(self, temp_dir: str):
        """Install packages from requirements.txt if it exists."""
        requirements_path = os.path.join(temp_dir, "requirements.txt")
        
        if os.path.exists(requirements_path):
            try:
                logger.info("Installing required packages...")
                
                # CRITICAL: Use the same Python executable that will run the code
                # AND install packages locally in the sandbox directory to avoid environment mismatches
                python_executable = sys.executable
                
                # Set up the SAME environment variables as code execution
                env = os.environ.copy()
                # Use correct path separator for Linux/Docker vs Windows
                path_separator = ":" if platform.system() != "Windows" else ";"
                current_pythonpath = env.get('PYTHONPATH', '')
                env['PYTHONPATH'] = temp_dir + (path_separator + current_pythonpath if current_pythonpath else '')
                env['HOME'] = temp_dir
                
                # Install packages with --target to install locally in sandbox
                # This ensures packages are installed exactly where the code will look for them
                install_command = [
                    python_executable, "-m", "pip", "install", 
                    "-r", "requirements.txt",
                    "--target", temp_dir,  # Install packages directly in sandbox directory
                    "--no-cache-dir",      # Don't use cache to avoid conflicts (important for Docker)
                    "--disable-pip-version-check",  # Reduce noise
                ]
                
                # Additional Docker/Linux optimizations
                if platform.system() != "Windows":
                    install_command.extend([
                        "--upgrade",           # Ensure latest versions in Docker
                        "--force-reinstall"    # Force reinstall to local directory
                    ])
                
                logger.debug(f"🐧 Install command: {' '.join(install_command)}")
                logger.debug(f"🐧 Install environment: PYTHONPATH={env.get('PYTHONPATH')}")
                
                # Install packages using pip with same environment as execution
                process = subprocess.Popen(
                    install_command,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env  # Use SAME environment as code execution
                )
                
                stdout, stderr = process.communicate(timeout=120)  # 2 minute timeout for installs
                
                if process.returncode == 0:
                    logger.info("Successfully installed packages locally in sandbox")
                    if stdout:
                        logger.debug(f"Install stdout: {stdout}")
                else:
                    logger.warning(f"Package installation failed: {stderr}")
                    if stdout:
                        logger.debug(f"Install stdout: {stdout}")
                        
            except subprocess.TimeoutExpired:
                logger.error("Package installation timed out")
            except Exception as e:
                logger.error(f"Failed to install packages: {e}")
                
    def _verify_environment_consistency(self, temp_dir: str):
        """Verify that installed packages are accessible in the execution environment."""
        requirements_path = os.path.join(temp_dir, "requirements.txt")
        
        if os.path.exists(requirements_path):
            try:
                # Read requirements
                with open(requirements_path, 'r') as f:
                    packages = [line.strip() for line in f.readlines() if line.strip()]
                
                if packages:
                    # Test import in the same environment that will execute the code
                    packages_repr = repr(packages)
                    test_script = f"""
import sys
import json

results = {{}}
packages = {packages_repr}

for package in packages:
    try:
        if package == 'beautifulsoup4':
            import bs4
            results[package] = True
        elif package == 'scikit-learn':
            import sklearn
            results[package] = True  
        elif package == 'Pillow':
            import PIL
            results[package] = True
        else:
            __import__(package)
            results[package] = True
    except ImportError:
        results[package] = False

print(json.dumps(results))
"""
                    
                    # Use SAME environment as code execution AND installation
                    env = os.environ.copy()
                    # Use correct path separator for Linux/Docker vs Windows
                    path_separator = ":" if platform.system() != "Windows" else ";"
                    current_pythonpath = env.get('PYTHONPATH', '')
                    env['PYTHONPATH'] = temp_dir + (path_separator + current_pythonpath if current_pythonpath else '')
                    env['HOME'] = temp_dir
                    
                    logger.debug(f"🔍 Verification environment: PYTHONPATH={env.get('PYTHONPATH')}")
                    logger.debug(f"🔍 Platform: {platform.system()}, Path separator: '{path_separator}'")
                    
                    # Run verification with same Python executable and environment
                    process = subprocess.Popen(
                        [sys.executable, "-c", test_script],
                        cwd=temp_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env
                    )
                    
                    stdout, stderr = process.communicate(timeout=30)
                    
                    if process.returncode == 0 and stdout:
                        import json
                        results = json.loads(stdout.strip())
                        failed_packages = [pkg for pkg, success in results.items() if not success]
                        
                        if failed_packages:
                            logger.error(f"Environment mismatch detected! Failed packages: {failed_packages}")
                        else:
                            logger.info("✅ Environment consistency verified - all packages accessible")
                    else:
                        logger.warning(f"Could not verify environment consistency: {stderr}")
                        
            except Exception as e:
                logger.warning(f"Environment verification failed: {e}")
    
    def _create_sandbox_template(self) -> str:
        """Create a template for the sandbox environment."""
        
        template = """
# Sandbox Environment Template
# This template provides a secure execution environment for user code

import sys
import os
import json
import numpy as np

# Restrict certain dangerous operations
import builtins

# Custom JSON encoder for NumPy/pandas compatibility
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle NumPy types
        if hasattr(obj, 'dtype'):  # NumPy arrays and scalars
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
        
        # Handle pandas types
        if hasattr(obj, 'to_dict'):  # DataFrames, Series
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # pandas scalars
            return obj.item()
            
        # Handle other common types
        if str(type(obj)).startswith('<class \\'numpy.'):
            # Fallback for any numpy type we missed
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return str(obj)
                
        return super().default(obj)

# Helper function for safe JSON serialization
def safe_json_dumps(obj, **kwargs):
    \"\"\"JSON dumps that handles NumPy/pandas types automatically\"\"\"
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)

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

# Make safe_json_dumps available globally
builtins.safe_json_dumps = safe_json_dumps

print("Sandbox environment initialized with JSON compatibility")
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
