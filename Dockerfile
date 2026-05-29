# syntax=docker/dockerfile:1.7
# ComfyUI texture/img2img container — built from scratch on the CUDA runtime.
# PyTorch wheels bundle their own CUDA, but we base on nvidia/cuda so cudnn/driver
# capabilities are present and GPU passthrough (--gpus all) works cleanly.

ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

# Which torch wheel index to pull (cu124 matches the 12.4.x base above).
ARG TORCH_INDEX=cu124
# Pin ComfyUI to a ref for reproducible builds; override at build time if needed.
ARG COMFYUI_REF=master

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFYUI_HOME=/opt/ComfyUI \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---- System dependencies -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git git-lfs wget curl ca-certificates \
        libgl1 libglib2.0-0 libgomp1 \
        ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# ---- PyTorch (CUDA build) ------------------------------------------------
# torchaudio is required: ComfyUI core imports it unconditionally (audio VAE).
RUN python -m pip install \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

# Pin the whole torch stack so no custom-node requirements.txt can pull a
# mismatched build (e.g. a CUDA 13 torchaudio that needs libcudart.so.13).
# PIP_CONSTRAINT applies to every subsequent pip install in this build.
# transformers<5: 5.x drops the CLIP `text_model.` key prefix, which breaks
# loading models saved under 4.x (e.g. Ubisoft CHORD's chord_v1.safetensors).
RUN printf 'torch==2.5.1+%s\ntorchvision==0.20.1+%s\ntorchaudio==2.5.1+%s\ntransformers>=4.50.3,<5\n' \
        "${TORCH_INDEX}" "${TORCH_INDEX}" "${TORCH_INDEX}" \
        > /etc/pip-torch-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-torch-constraints.txt \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${TORCH_INDEX}

# ---- ComfyUI core --------------------------------------------------------
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_HOME} \
    && cd ${COMFYUI_HOME} \
    && git checkout ${COMFYUI_REF}
RUN python -m pip install -r ${COMFYUI_HOME}/requirements.txt

# ---- Custom nodes (texture / tiling / PBR stack) -------------------------
COPY scripts/install_custom_nodes.sh /tmp/install_custom_nodes.sh
RUN bash /tmp/install_custom_nodes.sh

# Local custom nodes shipped with this project (e.g. tiling-aware upscale).
COPY custom_nodes/ ${COMFYUI_HOME}/custom_nodes/

# Snapshot baked nodes so the entrypoint can seed them into a named volume
# (lets users add nodes via ComfyUI-Manager without losing the baked set).
RUN cp -a ${COMFYUI_HOME}/custom_nodes /opt/default_custom_nodes

# ---- Config + entrypoint -------------------------------------------------
COPY extra_model_paths.yaml ${COMFYUI_HOME}/extra_model_paths.yaml
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR ${COMFYUI_HOME}
EXPOSE 8188

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--listen", "0.0.0.0", "--port", "8188"]
