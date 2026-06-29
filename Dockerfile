FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# ^ critical: must include "graphics" or "all" — default is "compute,utility"
#   which is why people get llvmpipe fallback even with the GPU passed through

# Hapus repositori/kunci NVIDIA lama yang bermasalah agar apt-get update tidak mogok
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list

# Jalankan update dengan mengabaikan error validasi sertifikat sementara untuk memperbarui sistem
RUN apt-get update -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true \
    || true

# Jalankan instalasi ca-certificates untuk memperbaiki validasi SSL/TLS
RUN apt-get install -y --allow-unauthenticated ca-certificates

RUN apt-get update && apt-get install -y \
    libvulkan1 vulkan-tools \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libxext6 libx11-6 libxrender1 libsm6 \
    wget git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA's EGL/GLVND vendor json + ICD — this is the part nvidia/cudagl used to
# bake in for you. Pulling them straight from NVIDIA's container-toolkit repo:
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d
RUN echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
    > /etc/vulkan/icd.d/nvidia_icd.json
RUN echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Miniconda
# RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/conda.sh \
#    && bash /tmp/conda.sh -b -p /opt/conda && rm /tmp/conda.sh
# ENV PATH=/opt/conda/bin:$PATH

# RUN conda create -n gaprl python=3.9 -y
# SHELL ["conda", "run", "-n", "gaprl", "/bin/bash", "-c"]

# RUN pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# WORKDIR /workspace
# RUN git clone https://github.com/THU-VCLab/GAP-RL.git
# WORKDIR /workspace/GAP-RL
# RUN pip install -r requirements.txt && pip install -e .

ENV DISPLAY=
CMD ["/bin/bash"]
