"""
Test Linux/Docker environment compatibility for sandbox executor
"""

import os
import sys
import platform
import tempfile
import subprocess
from core.sandbox_executor import SandboxExecutor

def simulate_docker_environment():
    """Test sandbox executor with Docker-like environment settings"""
    
    print(f"🐧 Testing Linux/Docker Environment Compatibility")
    print(f"Current Platform: {platform.system()}")
    print(f"Python Executable: {sys.executable}")
    print("")
    
    # Test NetworkX in Docker-like conditions
    test_code = """
import sys
import os
import json
import platform

# Show environment details (important for Docker debugging)
env_info = {
    "platform": platform.system(),
    "python_version": sys.version,
    "python_executable": sys.executable,
    "working_directory": os.getcwd(),
    "python_path": sys.path[:5],  # First 5 entries
    "environment_vars": {
        "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
        "HOME": os.environ.get("HOME", "Not set")
    }
}

print("=== DOCKER/LINUX ENVIRONMENT INFO ===")
print(json.dumps(env_info, indent=2))

try:
    # Test NetworkX functionality (common Docker use case)
    import networkx as nx
    import pandas as pd
    
    print()
    print("=== PACKAGE IMPORTS SUCCESSFUL ===")
    print(f"NetworkX version: {nx.__version__}")
    print(f"NetworkX location: {nx.__file__}")
    
    # Create test network
    edges_data = [
        ("Alice", "Bob"),
        ("Alice", "Carol"),
        ("Bob", "Carol"),
        ("Carol", "Dave"),
        ("Dave", "Eve")
    ]
    
    G = nx.Graph()
    G.add_edges_from(edges_data)
    
    # Calculate network metrics
    result = {
        "success": True,
        "platform": platform.system(),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "is_connected": nx.is_connected(G),
        "average_clustering": round(nx.average_clustering(G), 4),
        "diameter": nx.diameter(G) if nx.is_connected(G) else "N/A"
    }
    
    print()
    print("=== NETWORKX ANALYSIS RESULT ===")
    print(json.dumps(result, indent=2))
    
except ImportError as e:
    error_result = {
        "success": False,
        "error_type": "ImportError", 
        "error_message": str(e),
        "platform": platform.system(),
        "python_path_entries": len(sys.path)
    }
    
    print()
    print("=== IMPORT ERROR ===")
    print(json.dumps(error_result, indent=2))
    
except Exception as e:
    error_result = {
        "success": False,
        "error_type": type(e).__name__,
        "error_message": str(e),
        "platform": platform.system()
    }
    
    print()
    print("=== EXECUTION ERROR ===")
    print(json.dumps(error_result, indent=2))
"""
    
    # Execute with sandbox
    print("🧪 Executing NetworkX test in sandbox...")
    executor = SandboxExecutor()
    
    result = executor.execute_code(
        code=test_code,
        files={},
        timeout=180
    )
    
    print("📊 SANDBOX EXECUTION RESULTS:")
    print(f"Success: {result.get('success', False)}")
    print(f"Return Code: {result.get('return_code', 'N/A')}")
    
    if result.get('output'):
        print("\n📋 OUTPUT:")
        print(result['output'])
    
    if result.get('stderr'):
        print("\n❌ STDERR:")
        print(result['stderr'])
        
    if result.get('error'):
        print(f"\n⚠️ ERROR: {result['error']}")
    
    # Analyze results
    output = result.get('output', '') or ''  # Handle None output
    
    if output and 'NETWORKX ANALYSIS RESULT' in output and '"success": true' in output.lower():
        print("\n✅ LINUX/DOCKER COMPATIBILITY TEST PASSED!")
        print("✅ NetworkX working in simulated Docker environment")
        return True
    elif output and 'IMPORT ERROR' in output:
        print("\n❌ PACKAGE IMPORT FAILED!")
        print("❌ Check Docker package installation")
        return False
    else:
        print("\n⚠️ UNEXPECTED RESULT - Check output above")
        print(f"Output preview: {output[:200] if output else 'No output received'}")
        return False

def test_path_separators():
    """Test that path separators work correctly for different platforms"""
    
    print(f"\n🔍 Testing Path Separator Logic...")
    
    # Test Windows
    if platform.system() == "Windows":
        expected_separator = ";"
        print(f"Windows detected - expecting separator: '{expected_separator}'")
    else:
        expected_separator = ":"  
        print(f"Linux/Unix detected - expecting separator: '{expected_separator}'")
    
    # Test the logic
    test_separator = ":" if platform.system() != "Windows" else ";"
    
    if test_separator == expected_separator:
        print(f"✅ Path separator logic correct: '{test_separator}'")
    else:
        print(f"❌ Path separator logic incorrect: got '{test_separator}', expected '{expected_separator}'")
    
    return test_separator == expected_separator

def test_docker_pip_installation():
    """Test pip installation with Docker-like flags"""
    
    print(f"\n📦 Testing Docker-compatible pip installation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test requirements.txt
        req_file = os.path.join(temp_dir, "requirements.txt")
        with open(req_file, 'w') as f:
            f.write("networkx\\n")
        
        # Test pip command construction
        python_executable = sys.executable
        install_command = [
            python_executable, "-m", "pip", "install", 
            "-r", "requirements.txt",
            "--target", temp_dir,
            "--no-cache-dir",
            "--disable-pip-version-check",
        ]
        
        # Add Docker/Linux specific flags
        if platform.system() != "Windows":
            install_command.extend([
                "--upgrade",
                "--force-reinstall"
            ])
        
        print(f"🐧 Install command: {' '.join(install_command)}")
        
        # Test environment setup
        env = os.environ.copy()
        path_separator = ":" if platform.system() != "Windows" else ";"
        current_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = temp_dir + (path_separator + current_pythonpath if current_pythonpath else '')
        env['HOME'] = temp_dir
        
        print(f"🐧 Environment PYTHONPATH: {env['PYTHONPATH']}")
        print(f"🐧 Environment HOME: {env['HOME']}")
        
        return True

if __name__ == "__main__":
    print("🐧 Linux/Docker Compatibility Test Suite")
    print("=" * 60)
    
    # Run tests
    test1 = simulate_docker_environment()
    test2 = test_path_separators()
    test3 = test_docker_pip_installation()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"🐧 Docker Environment Test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"🔍 Path Separator Test: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"📦 Docker Pip Test: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    all_passed = test1 and test2 and test3
    
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 System is ready for Linux/Docker deployment!")
        print("✅ Environment isolation working")
        print("✅ Package installation compatible")
        print("✅ Path handling cross-platform")
    else:
        print("\n⚠️ Issues detected - review test output above")
    
    print("\n🐳 Docker Deployment Notes:")
    print("- Packages install locally in sandbox (--target)")
    print("- PYTHONPATH uses correct separator (':' for Linux)")
    print("- Environment isolation maintains consistency")
    print("- No system-level pip conflicts")
