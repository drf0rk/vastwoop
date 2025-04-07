#!/bin/bash
# VASTWOOP/ROOP-FLOYD Setup Script

echo "=== Starting VASTWOOP/ROOP Setup ==="

# Basic system setup
apt update -y
apt install -y git python3-pip ffmpeg --no-install-recommends

# Set CUDA environment variables
export CUDA_HOME="/usr/local/cuda-11.8"
export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib.real:${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export ORT_TENSORRT_ENGINE_CACHE_ENABLE="1"

# Clone repository
REPO_DIR="/vastwoop"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/drf0rk/vastwoop.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Clean installation of PyTorch and onnxruntime
pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir onnxruntime-gpu==1.16.3

# Create requirements.txt with properly ordered dependencies
cat > requirements.txt << 'EOL'
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
EOL

# Install remaining dependencies
pip install --no-cache-dir -r requirements.txt

# Additional specific dependency fixes
pip install --upgrade gradio --force
pip install --upgrade fastapi pydantic
pip install "numpy<2.0"

# Verify installation
python -c "import torch; import onnxruntime as ort; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); print(f'onnxruntime: {ort.__version__}, Device: {ort.get_device()}')"

echo "=== Setup complete ==="
