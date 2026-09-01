#!/bin/bash
set -e

# Use sudo if script is not run as root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

echo "🔄 Updating package lists..."
$SUDO apt-get update

echo "📦 Installing system utilities (Vulkan, Vim, Curl, Wget, etc.)..."
$SUDO apt-get install -y \
    curl \
    wget \
    vim \
    git \
    build-essential \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    tmux \
    htop \
    unzip

echo "🐍 Installing Miniconda..."
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    
    echo "⚙️ Initializing conda for bash..."
    $HOME/miniconda3/bin/conda init bash
    
    echo "✅ Miniconda installed successfully!"
    echo "⚠️  Please restart your shell or run 'source ~/.bashrc' to activate conda."
else
    echo "✅ Miniconda is already installed at $HOME/miniconda3."
fi

echo "🎉 All installations completed!"
