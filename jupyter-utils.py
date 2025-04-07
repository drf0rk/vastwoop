# JupyterLab utilities for VASTWOOP/ROOP
import os
import sys
import subprocess
import importlib.util

def goto_vastwoop():
    """Navigate to the VASTWOOP directory"""
    vastwoop_dir = "/vastwoop"
    if os.path.exists(vastwoop_dir):
        os.chdir(vastwoop_dir)
        print(f"Changed directory to: {vastwoop_dir}")
        
        # Make sure import_helper is loaded first
        if vastwoop_dir not in sys.path:
            sys.path.insert(0, vastwoop_dir)
            
        # Force import of torch before onnxruntime
        try:
            from import_helper import torch, ort, get_providers
            print("Environment verified: torch loaded before onnxruntime")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    else:
        print(f"Directory not found: {vastwoop_dir}")
        return False

def goto_root():
    """Navigate to the root directory"""
    os.chdir("/")
    print("Changed directory to root: /")
    return True

def check_gpu_status():
    """Check GPU status including memory usage"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        print(result.stdout)
        return True
    except:
        print("nvidia-smi not available - cannot check GPU status")
        return False
    
def restart_runtime():
    """Provide instructions to restart the Jupyter runtime"""
    print("To restart the runtime:")
    print("1. Click on the 'Runtime' menu")
    print("2. Select 'Restart runtime...'")
    print("3. Confirm the restart")
    print("\nThis will clear all variables and restart the Python kernel.")
    return True

def clear_cuda_cache():
    """Clear CUDA cache to free GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_allocated()
            print(f"CUDA cache cleared. Memory before: {before/1e6:.2f} MB, after: {after/1e6:.2f} MB")
            return True
        else:
            print("CUDA not available")
            return False
    except ImportError:
        print("PyTorch not installed - cannot clear CUDA cache")
        return False

def fix_run_py():
    """Fix run.py to ensure proper import order"""
    repo_paths = ["/vastwoop", "/roop-floyd", "/content/vastwoop", "/content/roop-floyd"]
    
    for repo_dir in repo_paths:
        run_py_path = f"{repo_dir}/run.py"
        if os.path.exists(run_py_path):
            with open(run_py_path, "r") as f:
                content = f.read()
            
            # Only modify if it doesn't already have the import
            if "from import_helper import" not in content:
                new_content = f"# Import torch before onnxruntime\nfrom import_helper import torch, ort, get_providers\n\n{content}"
                with open(run_py_path, "w") as f:
                    f.write(new_content)
                print(f"Modified {run_py_path} to import torch before onnxruntime")
                return True
    
    print("Could not find run.py in any of the expected locations")
    return False
