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
mkdir -p "$M"/{checkpoints,vae,controlnet,upscale_models,ipadapter,clip_vision,text_encoders,diffusion_models,LLM}

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

# Juggernaut XL v9 — the DEFAULT restyle checkpoint (much stronger material realism
# than SDXL base; the tiling pipeline is unchanged). Saved under the name the gateway
# default expects.
dl "$M/checkpoints" \
  "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" \
  "Juggernaut-XL_v9.safetensors"

# SDXL base 1.0 (fallback / comparison base)
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

# SDXL Union ControlNet (one model = depth/canny/lineart/scribble/tile). Set
# `controlnet_union: true` + a `control_type` to use it for restyle.
dl "$M/controlnet" \
  "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors" \
  "controlnet-union-sdxl-1.0.safetensors"

# IP-Adapter (SDXL plus) + its CLIP-ViT-H image encoder — for style-transfer restyle.
dl "$M/ipadapter" \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" \
  "ip-adapter-plus_sdxl_vit-h.safetensors"
dl "$M/clip_vision" \
  "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
  "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# Ubisoft CHORD: single-image -> full PBR maps (GATED — needs HF_TOKEN + license accept)
dl "$M/checkpoints" \
  "https://huggingface.co/Ubisoft/ubisoft-laforge-chord/resolve/main/chord_v1.safetensors" \
  "chord_v1.safetensors" gated

# ----------------------------------------------------------------------------- #
# OPTIONAL — FLUX.1 Kontext instruction-restyle (~24 GB). Large; only needed for
# the /restyle-flux endpoint (NOT tileable). Comment out if you don't need it.
# The diffusion model repo is gated (accept the license at
# huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), the encoders/vae are not.
# ----------------------------------------------------------------------------- #
if [ "${TEXFORGE_FLUX:-0}" = "1" ]; then
  dl "$M/diffusion_models" \
    "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors" \
    "flux1-dev-kontext_fp8_scaled.safetensors" gated
  dl "$M/text_encoders" \
    "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/text_encoders/clip_l.safetensors" \
    "clip_l.safetensors"
  dl "$M/text_encoders" \
    "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/text_encoders/t5xxl_fp16.safetensors" \
    "t5xxl_fp16.safetensors"
  dl "$M/vae" \
    "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/vae/ae.safetensors" \
    "ae.safetensors"
else
  echo "[note] FLUX Kontext skipped (set TEXFORGE_FLUX=1 to download ~24 GB for /restyle-flux)."
fi

# NOTE: StableMaterials / MatForger (the 'stablematerials' PBR backend) are pulled
# automatically by the diffusers pipeline (trust_remote_code) on first /pbr call;
# no manual download here. Florence-2 PromptGen (captioning) likewise auto-downloads
# into /data/models/LLM on first restyle with auto_caption=true.

echo "Done. Files now visible to ComfyUI under ${M}."
ls -lhR "$M" | grep -E '\.(safetensors|pth)$' || true
