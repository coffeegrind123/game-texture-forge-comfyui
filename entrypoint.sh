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
  /data/models/ipadapter

echo "[entrypoint] launching ComfyUI: main.py $*"
exec python -u "${COMFYUI_HOME}/main.py" "$@"
