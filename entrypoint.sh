#!/usr/bin/env bash
# Container entrypoint: seed baked custom nodes into the (possibly persistent)
# custom_nodes volume, ensure model directories exist, then launch ComfyUI.
set -e

COMFYUI_HOME="${COMFYUI_HOME:-/opt/ComfyUI}"

# --- Seed baked custom nodes ---------------------------------------------
# A named volume mounted at custom_nodes starts empty (or stale across rebuilds).
# Copy any baked node that isn't already present so the texture stack is always
# available, while still persisting nodes the user installs via ComfyUI-Manager.
if [ -d /opt/default_custom_nodes ]; then
  for d in /opt/default_custom_nodes/*/; do
    name="$(basename "$d")"
    if [ ! -d "${COMFYUI_HOME}/custom_nodes/${name}" ]; then
      echo "[entrypoint] seeding custom node: ${name}"
      cp -a "$d" "${COMFYUI_HOME}/custom_nodes/${name}"
    fi
  done
fi

# --- Ensure model directory layout exists --------------------------------
mkdir -p \
  /data/models/checkpoints \
  /data/models/vae \
  /data/models/loras \
  /data/models/controlnet \
  /data/models/upscale_models \
  /data/models/clip \
  /data/models/clip_vision \
  /data/models/unet \
  /data/models/diffusion_models \
  /data/models/embeddings \
  /data/models/style_models \
  /data/models/ipadapter \
  /data/models/text_encoders \
  /data/models/LLM

# --- Persist captioner (Florence-2) downloads on the bind mount -----------
# kijai's DownloadAndLoadFlorence2Model writes to $COMFYUI_HOME/models/LLM (the
# primary models dir, NOT a volume), so the ~1.8 GB model would re-download on
# every container recreate. Symlink that path to the bind-mounted /data/models/LLM.
mkdir -p "${COMFYUI_HOME}/models" /data/models/LLM
if [ ! -L "${COMFYUI_HOME}/models/LLM" ]; then
  if [ -d "${COMFYUI_HOME}/models/LLM" ]; then
    cp -an "${COMFYUI_HOME}/models/LLM/." /data/models/LLM/ 2>/dev/null || true
    rm -rf "${COMFYUI_HOME}/models/LLM"
  fi
  ln -s /data/models/LLM "${COMFYUI_HOME}/models/LLM"
fi

echo "[entrypoint] launching ComfyUI: main.py $*"
exec python -u "${COMFYUI_HOME}/main.py" "$@"
