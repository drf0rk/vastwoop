# VASTWOOP Setup Cell - Run this in JupyterLab
# Basic system dependencies
!apt update -y && apt install -y git python3-pip ffmpeg --no-install-recommends

# Clone repository
import os
repo_dir = "/vastwoop"
if not os.path.exists(repo_dir):
    !git clone https://github.com/drf0rk/vastwoop.git {repo_dir}

# Set CUDA environment variables - critical for compatibility
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/compat/lib.real:{os.environ['CUDA_HOME']}/lib64:/usr/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['PATH']}"
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"

# Clean installation of PyTorch and onnxruntime to avoid conflicts
!pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
!pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
!pip install --no-cache-dir onnxruntime-gpu==1.16.3

# Create import_helper.py for proper import order
%%writefile {repo_dir}/import_helper.py
# Ensure torch is imported before onnxruntime
import os
import sys

# Set environment variables
os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.8"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/compat/lib.real:{os.environ['CUDA_HOME']}/lib64:/usr/lib/x86_64-linux-gnu:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Import torch first
import torch
print(f"PyTorch {torch.__version__} loaded successfully")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Then import onnxruntime
import onnxruntime as ort
print(f"ONNX Runtime {ort.__version__} loaded successfully")
print(f"Available providers: {ort.get_available_providers()}")
print(f"Device: {ort.get_device()}")

# Export function to get optimal providers
def get_providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

# Go to project directory and install remaining dependencies
%cd {repo_dir}

# Create requirements.txt with properly ordered dependencies
%%writefile requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu118
numpy==1.26.4
torch==2.1.0+cu118
torchvision==0.16.0+cu118
onnxruntime-gpu==1.16.3
gradio==5.9.1
opencv-python-headless==4.10.0.84
onnx==1.16.1
insightface==0.7.3
albucore==0.0.16
psutil==5.9.6
tqdm==4.66.4
ftfy
regex
pyvirtualcam

# Install remaining dependencies
!pip install --no-cache-dir -r requirements.txt

# Additional specific dependency fixes
!pip install --upgrade gradio --force
!pip install --upgrade fastapi pydantic
!pip install "numpy<2.0"

# Create JupyterLab navigation helper
%%writefile {repo_dir}/jupyter_utils.py
import os
import sys

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

# Fix run.py if it exists
if os.path.exists(f"{repo_dir}/run.py"):
    with open(f"{repo_dir}/run.py", "r") as f:
        content = f.read()
    
    # Only modify if it doesn't already have the import
    if "from import_helper import" not in content:
        new_content = f"# Import torch before onnxruntime\nfrom import_helper import torch, ort, get_providers\n\n{content}"
        with open(f"{repo_dir}/run.py", "w") as f:
            f.write(new_content)
        print("Modified run.py to import torch before onnxruntime")

# Verify final installation
from import_helper import torch, ort, get_providers
print("\n=== Final Environment Check ===")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}")
print(f"onnxruntime: {ort.__version__}, Device: {ort.get_device()}")
print(f"Recommended providers: {get_providers()}")
print("Setup complete! You can now use 'from import_helper import torch, ort, get_providers' in your code")
