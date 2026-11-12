#!/bin/bash
set -e

echo "📦 Upgrading pip and setting up build tools..."
# Gym 0.21.0 and old PyTorch versions need older setuptools and wheel
pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40.0"

echo "🏋️ Installing Gym 0.21.0 with --no-build-isolation..."
# This bypasses the pip build isolation that pulls in modern setuptools and breaks the gym 0.21.0 setup.py
pip install gym==0.21.0 --no-build-isolation

echo "🔥 Installing PyTorch (1.13.1 for CUDA 11.7)..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117

echo "🧊 Installing PyTorch3D..."
pip install pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html

echo "📚 Installing the rest of the requirements..."
pip install -r working_requirements.txt

echo "🎯 Installing graspnetAPI with --no-deps..."
pip install graspnetAPI --no-deps

echo "⚡ Building & installing pointnet2_ops (PointNet++ CUDA ops for LocalGrasp)..."
pip install ninja
if ! python -c "import pointnet2_ops" 2>/dev/null; then
    if [ ! -d "/tmp/pointMLP-pytorch" ]; then
        git clone https://github.com/ma-xu/pointMLP-pytorch.git /tmp/pointMLP-pytorch
    fi
    (
        cd /tmp/pointMLP-pytorch/pointnet2_ops_lib
        export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"
        export FORCE_CUDA=1
        pip install --no-build-isolation .
    )
fi

echo "🔒 Enforcing numpy 1.23.5..."
pip install "numpy==1.23.5"

echo "✅ Python dependencies installed successfully!"
