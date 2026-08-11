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
    wget git curl ca-certificates vim \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA EGL/GLVND + Vulkan ICDs (needed because cuda images no longer ship cudagl).
# Put ICD in both loader search paths; force NVIDIA so Mesa lavapipe (llvmpipe) is ignored.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /usr/share/vulkan/icd.d \
    && printf '%s\n' \
        '{' \
        '    "file_format_version": "1.0.0",' \
        '    "ICD": {' \
        '        "library_path": "libGLX_nvidia.so.0",' \
        '        "api_version": "1.3.0"' \
        '    }' \
        '}' \
        > /etc/vulkan/icd.d/nvidia_icd.json \
    && cp /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json \
    && printf '%s\n' \
        '{' \
        '    "file_format_version": "1.0.0",' \
        '    "ICD": {' \
        '        "library_path": "libEGL_nvidia.so.0"' \
        '    }' \
        '}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    # Drop Mesa ICDs if a package pulled them in (causes vulkaninfo -> llvmpipe)
    && rm -f /usr/share/vulkan/icd.d/*lvp* \
            /usr/share/vulkan/icd.d/*radeon* \
            /usr/share/vulkan/icd.d/*intel* \
            /etc/vulkan/icd.d/*lvp* \
            /etc/vulkan/icd.d/*radeon* \
            /etc/vulkan/icd.d/*intel* \
            /usr/share/glvnd/egl_vendor.d/50_mesa.json

# Force NVIDIA Vulkan/EGL; without this the loader often picks lavapipe.
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV DISPLAY=
CMD ["/bin/bash"]
