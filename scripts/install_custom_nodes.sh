#!/usr/bin/env bash
# Installs the texture/img2img/PBR custom-node stack into ComfyUI.
# Tolerant by design: a single failed repo logs a WARN and continues so the
# image still builds. Re-run is idempotent (existing dirs are skipped).
set -uo pipefail

COMFYUI_HOME="${COMFYUI_HOME:-/opt/ComfyUI}"
cd "${COMFYUI_HOME}/custom_nodes"

# Verified repository URLs (checked against GitHub before baking).
REPOS=(
  # Node/package manager — install/update other nodes from the UI.
  "https://github.com/ltdrdata/ComfyUI-Manager.git"
  # Tiling: patches model conv layers to circular padding (works on img2img too).
  "https://github.com/spinagon/ComfyUI-seamless-tiling.git"
  # Post-process an already-rendered image into a seamless tile (offset/radial).
  "https://github.com/SparknightLLC/ComfyUI-MakeSeamlessTexture.git"
  # Non-square / rotated tile shapes.
  "https://github.com/JosefKuchar/ComfyUI-AdvancedTiling.git"
  # PBR map extraction + channel packing + utilities (Marigold/Lotus backed).
  "https://github.com/amtarr/ComfyUI-TextureAlchemy.git"
  # ControlNet preprocessors (Tile, Canny, Depth, Normal, LineArt ...).
  "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
  # Marigold depth/normal estimation (used by TextureAlchemy PBR extractor).
  "https://github.com/kijai/ComfyUI-Marigold.git"
  # Ubisoft CHORD: single-image -> full PBR (BaseColor/Normal/Height/Rough/Metal).
  "https://github.com/ubisoft/ComfyUI-Chord.git"
  # Florence-2 captioning -> content-aware restyle prompts (PromptGen models).
  "https://github.com/kijai/ComfyUI-Florence2.git"
  # IP-Adapter: inject a style reference while a ControlNet holds layout.
  "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"
  # Unsampling / noise injection: invert a source to its own noise for tight restyle.
  "https://github.com/BlenderNeko/ComfyUI_Noise.git"
)

for url in "${REPOS[@]}"; do
  name="$(basename "$url" .git)"
  echo "=================================================================="
  echo "=== ${name}"
  echo "=================================================================="
  if [ -d "$name" ]; then
    echo "[skip] $name already present"
    continue
  fi
  if ! git clone --depth 1 "$url" "$name"; then
    echo "[WARN] clone failed: $name — continuing"
    continue
  fi
  if [ -f "$name/requirements.txt" ]; then
    echo "[pip] installing requirements for $name"
    python -m pip install -r "$name/requirements.txt" \
      || echo "[WARN] requirements failed for $name — node may be partially functional"
  fi
done

# Extra Python deps for the texture-forge local nodes + captioning:
#   diffusers/accelerate  -> StableMaterials/MatForger PBR pipeline (MatForgerMaterialEstimation)
#   timm/einops/peft      -> Florence-2 model + LoRA support
# PIP_CONSTRAINT keeps transformers<5 (CHORD compatibility) regardless.
echo "[pip] installing shared deps for texture-forge + captioning"
python -m pip install \
    "diffusers>=0.27" accelerate "timm>=0.9" einops peft \
  || echo "[WARN] shared deps install failed — some nodes may be partially functional"

echo "Custom node installation complete."
ls -1 "${COMFYUI_HOME}/custom_nodes"
