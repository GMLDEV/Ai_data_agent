import subprocess
import tempfile
import os
import sys
import psutil
import signal
import time
import json
from typing import Dict, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SandboxExecutor:
    def __init__(self, memory_limit: int, cpu_limit: float, time_limit: int):
        self.memory_limit = memory_limit  # bytes
        self.cpu_limit = cpu_limit  # CPU cores
        self.time_limit = time_limit  # seconds
        
        # Allowed libraries for sandbox
        self.allowed_imports = [
            'pandas', 'numpy', 'json', 'csv', 're', 'os', 'sys',
            'io', 'base64', 'math', 'datetime', 'collections'
        ]
        
        # Libraries that will be available in Phase 2
        self.future_imports = [
            'matplotlib', 'seaborn', 'requests', 'beautifulsoup4', 
            'duckdb', 'PIL'
        ]
    
    def execute_simple(self, code: str) -> Dict[str, Any]:
        """Simple execution for Phase 1 testing"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create script file
                script_path = os.path.join(temp_dir, 'test_script.py')
                
                # Wrap code with basic error handling
                wrapped_code = self._create_simple_wrapper(code)
                
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(wrapped_code)
                
                # Execute script
                result = self._run_script(script_path, temp_dir)
                
                return result
                
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return {
                "success": False,
                "error": f"Sandbox execution failed: {str(e)}",
                "stdout": "",
                "stderr": str(e)
            }

    def _create_simple_wrapper(self, user_code: str) -> str:
        """Create a simple wrapper for user code"""
        wrapper = f'''
import sys
import json
import traceback
from io import StringIO

# Capture stdout and stderr
stdout_buffer = StringIO()
stderr_buffer = StringIO()

try:
    # Redirect output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    
    # Execute user code
{self._indent_code(user_code, 4)}
    
    # Get the captured output
    captured_stdout = stdout_buffer.getvalue()
    captured_stderr = stderr_buffer.getvalue()
    
    # Restore original stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Print to console so we can see it in process output
    if captured_stdout:
        print("=== STDOUT ===")
        print(captured_stdout)
    if captured_stderr:
        print("=== STDERR ===") 
        print(captured_stderr)
    
    # Capture results
    result = {{
        "success": True,
        "stdout": captured_stdout,
        "stderr": captured_stderr,
        "error": None
    }}
    
except Exception as e:
    captured_stdout = stdout_buffer.getvalue()
    captured_stderr = stderr_buffer.getvalue()
    
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    error_info = str(e)
    traceback_info = traceback.format_exc()
    
    print("=== ERROR ===")
    print(f"Error: {{error_info}}")
    print(f"Traceback: {{traceback_info}}")
    
    result = {{
        "success": False,
        "stdout": captured_stdout,
        "stderr": captured_stderr,
        "error": error_info,
        "traceback": traceback_info
    }}

finally:
    # Ensure stdout/stderr are restored
    sys.stdout = old_stdout
    sys.stderr = old_stderr

# Write result to file
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, default=str)

print("=== EXECUTION COMPLETED ===")
'''
        return wrapper

    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code for wrapping"""
        return '\n'.join(' ' * spaces + line for line in code.split('\n'))
    
    def _run_script(self, script_path: str, temp_dir: str) -> Dict[str, Any]:
        """Run script with basic monitoring"""
        try:
            # Start process
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.time_limit)
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "error": "Execution timeout",
                    "timeout": True,
                    "stdout": "",
                    "stderr": "Process killed due to timeout"
                }
            
            # Try to read result file
            result_path = os.path.join(temp_dir, "result.json")
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    result['process_stdout'] = stdout
                    result['process_stderr'] = stderr
                    result['exit_code'] = process.returncode
                    return result
            else:
                return {
                    "success": False,
                    "error": "No result file generated",
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": process.returncode
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Script execution failed: {str(e)}",
                "stdout": "",
                "stderr": str(e)
            }
    
    def test_sandbox(self) -> Dict[str, Any]:
        """Test sandbox functionality"""
        test_code = '''
import sys
print("Python version:", sys.version)
print("Available modules test:")

# Test basic operations
result = 2 + 2
print(f"2 + 2 = {result}")

# Test list operations
numbers = [1, 2, 3, 4, 5]
print(f"Sum of {numbers} = {sum(numbers)}")

print("Sandbox test completed successfully!")
'''
        return self.execute_simple(test_code)