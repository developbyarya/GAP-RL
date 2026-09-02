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

echo "✅ Python dependencies installed successfully!"
