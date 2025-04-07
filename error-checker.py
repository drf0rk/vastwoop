# Error checking and diagnostics for VASTWOOP/ROOP
import sys
import os
import importlib.util
import subprocess
import platform
import time
import gc

def check_dependency(package):
    """Check if a package is installed and return its version"""
    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            return False, None
        
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "Unknown")
        return True, version
    except ImportError:
        return False, None

def check_gpu():
    """Check CUDA GPU availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "None"
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        
        # Get CUDA memory info
        if cuda_available:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            free_mem = total_mem - allocated_mem
            
            mem_info = {
                "total": total_mem,
                "reserved": reserved_mem,
                "allocated": allocated_mem,
                "free": free_mem
            }
        else:
            mem_info = {"total": 0, "reserved": 0, "allocated": 0, "free": 0}
            
        return cuda_available, device_name, cuda_version, mem_info
    except ImportError:
        return False, "None", "Not installed", {"total": 0, "reserved": 0, "allocated": 0, "free": 0}

def run_system_check():
    """Run a comprehensive system check and report status"""
    print("=== VASTWOOP/ROOP System Check ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check critical dependencies
    dependencies = [
        "torch", "torchvision", "onnxruntime", "numpy", 
        "gradio", "opencv-python-headless", "insightface"
    ]
    
    for dep in dependencies:
        installed, version = check_dependency(dep.split('-')[0])  # Handle opencv-python-headless
        status = f"v{version}" if installed else "NOT INSTALLED"
        print(f"{dep}: {status}")
    
    # Check GPU
    cuda_available, device_name, cuda_version, mem_info = check_gpu()
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA version: {cuda_version}")
    print(f"GPU device: {device_name}")
    if cuda_available:
        print(f"GPU memory: Total {mem_info['total']:.2f} GB, Free {mem_info['free']:.2f} GB")
    
    # Check ONNX providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        if 'CUDAExecutionProvider' not in providers:
            print("WARNING: CUDA execution provider not available for ONNX Runtime")
    except ImportError:
        print("ONNX Runtime: NOT INSTALLED")
    
    # Check environment variables
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    print(f"CUDA_HOME: {cuda_home}")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")
    
    return cuda_available

def fix_common_issues():
    """Attempt to fix common issues with the setup"""
    fixed_issues = []
    
    # Issue 1: CUDA_HOME not set
    if not os.environ.get('CUDA_HOME'):
        cuda_paths = ["/usr/local/cuda-11.8", "/usr/local/cuda", "/usr/local/cuda-11"]
        for path in cuda_paths:
            if os.path.exists(path):
                os.environ["CUDA_HOME"] = path
                fixed_issues.append(f"Set CUDA_HOME to {path}")
                break
    
    # Issue 2: Missing LD_LIBRARY_PATH
    if os.environ.get('CUDA_HOME') and 'LD_LIBRARY_PATH' not in os.environ:
        cuda_home = os.environ.get('CUDA_HOME')
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:/usr/lib/x86_64-linux-gnu"
        fixed_issues.append("Set LD_LIBRARY_PATH to include CUDA libraries")
    
    # Issue 3: ORT_TENSORRT_ENGINE_CACHE_ENABLE not set
    if not os.environ.get('ORT_TENSORRT_ENGINE_CACHE_ENABLE'):
        os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
        fixed_issues.append("Enabled TensorRT engine caching for better performance")
    
    # Issue 4: Clear CUDA cache if memory is low
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            if allocated > 0.8 * total:  # If more than 80% used
                torch.cuda.empty_cache()
                gc.collect()
                fixed_issues.append("Cleared CUDA cache due to high memory usage")
    except:
        pass
    
    if fixed_issues:
        print("Fixed issues:")
        for issue in fixed_issues:
            print(f"- {issue}")
    else:
        print("No common issues were detected")

def check_import_order():
    """Verify that torch is imported before onnxruntime"""
    # Use a subprocess to check import order without affecting current process
    check_code = """
import sys
imports = []
real_import = __import__

def tracking_import(name, *args, **kwargs):
    if name in ('torch', 'onnxruntime'):
        imports.append(name)
    return real_import(name, *args, **kwargs)

sys.modules['__builtin__.__import__' if sys.version_info[0] < 3 else 'builtins.__import__'] = tracking_import

try:
    import torch
    import onnxruntime
    correct_order = imports == ['torch', 'onnxruntime']
    print(f"Import order: {imports}")
    print(f"Correct order: {correct_order}")
except Exception as e:
    print(f"Error: {e}")
"""
    result = subprocess.run([sys.executable, '-c', check_code], capture_output=True, text=True)
    print(result.stdout)
    return 'Correct order: True' in result.stdout

def monitor_gpu_for(seconds=10):
    """Monitor GPU usage for a specified number of seconds"""
    if not check_dependency('torch')[0]:
        print("PyTorch not installed - cannot monitor GPU")
        return
    
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available - cannot monitor GPU")
        return
    
    print(f"Monitoring GPU usage for {seconds} seconds...")
    start_time = time.time()
    while time.time() - start_time < seconds:
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        print(f"Time: {time.time() - start_time:.1f}s | Allocated: {allocated:.3f} GB | Reserved: {reserved:.3f} GB")
        time.sleep(1)
    print("Monitoring complete")

def get_providers():
    """Get optimal ONNX Runtime providers"""
    try:
        import torch
        import onnxruntime as ort
        return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    except:
        return ['CPUExecutionProvider']

def troubleshoot():
    """Run full troubleshooting routine"""
    print("Running full troubleshooting...")
    
    # Step 1: System check
    print("\n=== System Check ===")
    run_system_check()
    
    # Step 2: Fix common issues
    print("\n=== Fixing Common Issues ===")
    fix_common_issues()
    
    # Step 3: Check import order
    print("\n=== Verifying Import Order ===")
    check_import_order()
    
    # Step 4: Short GPU monitoring
    print("\n=== GPU Monitoring (5s) ===")
    monitor_gpu_for(5)
    
    print("\nTroubleshooting complete. If issues persist, please check requirements.txt and reinstall dependencies.")
