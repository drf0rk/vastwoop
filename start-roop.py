#!/usr/bin/env python3
# VASTWOOP/ROOP Starter Script with error handling

import os
import sys
import platform
import importlib
import subprocess
import time

def print_banner():
    """Print a welcome banner"""
    banner = """
    ____   ____   ____  ____     ______    __    ____  ______  
   |    \ /    \ /    \|    \   |      |  /  ]  /    ||      | 
   |  D  )  o  |  o  ||  _  |  |      | /  /  |  o  ||      | 
   |    /|     |     ||  |  |  |_|  |_|/  /   |     ||_|  |_| 
   |    \|  _  |  _  ||  |  |    |  | /   \_  |  _  |  |  |   
   |  .  \  |  |  |  ||  |  |    |  | \     | |  |  |  |  |   
   |__|\_|__|__|__|__||__|__|    |__|  \____| |__|__|  |__|   
    """
    print(banner)

def setup_environment():
    """Set up environment variables"""
    # Set crucial environment variables
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
    
    # Look for CUDA installation
    cuda_paths = ["/usr/local/cuda-11.8", "/usr/local/cuda", "/usr/local/cuda-11"]
    for path in cuda_paths:
        if os.path.exists(path):
            os.environ["CUDA_HOME"] = path
            os.environ["LD_LIBRARY_PATH"] = f"{path}/lib64:/usr/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"
            os.environ["PATH"] = f"{path}/bin:{os.environ.get('PATH', '')}" 
            print(f"Found CUDA at {path}")
            break

def check_dependencies():
    """Check that all dependencies are available"""
    try:
        # First import torch to ensure proper order
        import torch
        print(f"• PyTorch {torch.__version__} ✓")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
    except ImportError:
        print("✗ PyTorch not found - please run setup.sh first")
        return False
    
    try:
        # Then import onnxruntime
        import onnxruntime as ort
        print(f"• ONNX Runtime {ort.__version__} ✓")
        providers = ort.get_available_providers()
        print(f"  - Available providers: {providers}")
        if 'CUDAExecutionProvider' not in providers:
            print("  ⚠️ CUDA provider not available for ONNX Runtime")
    except ImportError:
        print("✗ ONNX Runtime not found - please run setup.sh first")
        return False
    
    # Check other dependencies
    dependencies = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("gradio", "Gradio"),
        ("insightface", "InsightFace")
    ]
    
    for module_name, display_name in dependencies:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "Unknown")
            print(f"• {display_name} {version} ✓")
        except ImportError:
            print(f"✗ {display_name} not found - please run setup.sh first")
            return False
    
    return True

def find_repository():
    """Find the repository directory"""
    # List of possible locations
    repo_paths = [
        "/vastwoop",
        "/roop-floyd",
        os.path.join(os.path.expanduser("~"), "vastwoop"),
        os.path.join(os.path.expanduser("~"), "roop-floyd")
    ]
    
    # Check current directory first
    if os.path.exists("run.py"):
        return os.getcwd()
    
    # Check all potential paths
    for path in repo_paths:
        if os.path.exists(os.path.join(path, "run.py")):
            return path
    
    return None

def create_import_helper(repo_dir):
    """Create import helper if it doesn't exist"""
    import_helper_path = os.path.join(repo_dir, "import_helper.py")
    if not os.path.exists(import_helper_path):
        print("Creating import helper module...")
        with open(import_helper_path, "w") as f:
            f.write("""# Ensure torch is imported before onnxruntime
import os
import sys

# Set environment variables
os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'

# Import torch first
import torch
print(f"PyTorch {torch.__version__} loaded successfully")

# Then import onnxruntime
import onnxruntime as ort
print(f"ONNX Runtime {ort.__version__} loaded successfully")

# Export function to get optimal providers
def get_providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

__all__ = ['torch', 'ort', 'get_providers']
""")
        print("Import helper created.")

def fix_run_py(repo_dir):
    """Fix run.py to ensure proper import order"""
    run_py_path = os.path.join(repo_dir, "run.py")
    if os.path.exists(run_py_path):
        # Create backup
        backup_path = os.path.join(repo_dir, "run.py.bak")
        if not os.path.exists(backup_path):
            with open(run_py_path, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
            print("Created backup of run.py")
        
        # Read run.py
        with open(run_py_path, "r") as f:
            content = f.read()
        
        # Only modify if it doesn't already have the import
        if "from import_helper import" not in content:
            new_content = f"# Import torch before onnxruntime\nfrom import_helper import torch, ort, get_providers\n\n{content}"
            with open(run_py_path, "w") as f:
                f.write(new_content)
            print("Updated run.py to import torch before onnxruntime")

def start_application(repo_dir):
    """Start the application"""
    run_py = os.path.join(repo_dir, "run.py")
    
    if not os.path.exists(run_py):
        print(f"Error: Could not find run.py in {repo_dir}")
        return False
    
    print("\nStarting application...")
    
    # Run the application
    try:
        process = subprocess.Popen([sys.executable, run_py])
        return process
    except Exception as e:
        print(f"Error starting application: {e}")
        return None

def main():
    """Main entry point"""
    print_banner()
    print("Initializing VASTWOOP/ROOP launcher...\n")
    
    # Set up environment
    setup_environment()
    
    # Find repository
    repo_dir = find_repository()
    if not repo_dir:
        print("Error: Could not find repository with run.py")
        print("Please run this script from the repository directory or install using setup.sh")
        return 1
    
    print(f"Found repository at: {repo_dir}")
    os.chdir(repo_dir)
    
    # Create import helper
    create_import_helper(repo_dir)
    
    # Fix run.py
    fix_run_py(repo_dir)
    
    # Check dependencies
    if not check_dependencies():
        print("\nSome dependencies are missing. Please run setup.sh first.")
        return 1
    
    # Start application
    print("\nAll checks passed! Starting application...")
    app_process = start_application(repo_dir)
    
    if app_process:
        print("Application started successfully!")
        try:
            app_process.wait()
        except KeyboardInterrupt:
            print("Shutting down...")
            app_process.terminate()
    else:
        print("Failed to start application.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
