#!/usr/bin/env bash
# Downloads the starter model set for the texture img2img + tiling + PBR workflow.
#
# IMPORTANT: this runs INSIDE the container (writing to /data/models), because the
# Docker daemon runs on the host and does not share this shell's filesystem — files
# written here on the host side would never reach the container. Invoke via:
#     make models
# which pipes this script into `docker compose exec -T comfyui bash -s`.
set -uo pipefail

M="/data/models"
mkdir -p "$M"/{checkpoints,vae,controlnet,upscale_models}

dl() {  # dl <dest_dir> <url> <filename> [gated]
  local dir="$1" url="$2" name="$3" gated="${4:-}"
  local out="${dir}/${name}"
  if [ -f "$out" ]; then echo "[skip] ${name} (exists)"; return; fi
  local auth=()
  if [ -n "${HF_TOKEN:-}" ]; then auth=(--header="Authorization: Bearer ${HF_TOKEN}"); fi
  echo "[get ] ${name}"
  if wget -q --show-progress "${auth[@]}" -O "$out" "$url"; then
    echo "[ ok ] ${name}"
  else
    rm -f "$out"
    if [ -n "$gated" ] && [ -z "${HF_TOKEN:-}" ]; then
      echo "[GATE] ${name}: this is a GATED Hugging Face repo."
      echo "       1) open ${url%/resolve/*} and click 'Agree and access'"
      echo "       2) create a token at https://huggingface.co/settings/tokens"
      echo "       3) re-run with the token, e.g.:"
      echo "          docker compose exec -e HF_TOKEN=hf_xxx -T comfyui bash -s < scripts/download_models.sh"
    else
      echo "[WARN] failed: ${name}"
    fi
  fi
}

echo "Downloading into container path: ${M}"

# Photoreal SDXL checkpoint (img2img "more realistic" base)
dl "$M/checkpoints" \
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
  "sd_xl_base_1.0.safetensors"

# SDXL VAE (fp16 fix)
dl "$M/vae" \
  "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors" \
  "sdxl_vae.safetensors"

# SDXL Tile ControlNet (locks structure during realistic restyle)
dl "$M/controlnet" \
  "https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" \
  "controlnet-tile-sdxl-1.0.safetensors"

# 4x upscale model (detail pass; stays tileable under the circular model)
dl "$M/upscale_models" \
  "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth" \
  "4x-UltraSharp.pth"

# Ubisoft CHORD: single-image -> full PBR maps (GATED — needs HF_TOKEN + license accept)
dl "$M/checkpoints" \
  "https://huggingface.co/Ubisoft/ubisoft-laforge-chord/resolve/main/chord_v1.safetensors" \
  "chord_v1.safetensors" gated

echo "Done. Files now visible to ComfyUI under ${M}."
ls -lhR "$M" | grep -E '\.(safetensors|pth)$' || true
