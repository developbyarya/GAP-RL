FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies for Vulkan, SSH, and Build Tools
RUN apt-get update && apt-get install -y \
    vulkan-utils \
    libvulkan1 \
    libegl1-mesa \
    libgl1-mesa-glx \
    wget \
    git \
    ninja-build \
    build-essential \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Export Vulkan ICD to use NVIDIA's implementation
ENV VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"

# 2. Install Miniconda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 3. Create 'gaprl' conda environment with Python 3.9
RUN conda create -n gaprl python=3.9 -y

# Use the 'gaprl' environment for all subsequent commands
SHELL ["conda", "run", "-n", "gaprl", "/bin/bash", "-c"]

WORKDIR /workspace

# 4. Clone GAP-RL repository
RUN git clone https://github.com/developbyarya/GAP-RL.git

WORKDIR /workspace/GAP-RL

# Set CUDA vars for compiling PointNetOps
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"
ENV FORCE_CUDA=1

# 5. Install all base Python dependencies using the robust shell script
# This correctly handles pip downgrading and gym isolation
RUN ./scripts/install_python_deps.sh

# 6. Install evaluation dependencies (graspnetAPI & grasp_nms)
# graspnetAPI forces a numpy downgrade to 1.20.3, so we restore numpy==1.23.5 immediately after
RUN pip install graspnetAPI grasp_nms && \
    pip install numpy==1.23.5

# 7. Install the GAP-RL package in editable mode
RUN pip install -e .

# Configure SSH for remote connection
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22

# Start SSH daemon
CMD ["/usr/sbin/sshd", "-D"]
