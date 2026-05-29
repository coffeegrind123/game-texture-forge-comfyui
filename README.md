# ComfyUI — Texture / img2img Container

A from-scratch CUDA container image for ComfyUI, pre-loaded with the custom-node
stack for **turning existing textures into similar-but-different ones** (e.g. more
realistic) while **creating or preserving seamless tiling**, plus full PBR map
generation for gamedev.

## What's baked in

| Node pack | Role in the texture pipeline |
|---|---|
| `ComfyUI-Manager` | Install/update more nodes from the UI |
| `ComfyUI-seamless-tiling` (spinagon) | Circular-pad the model → tiling on **img2img**; `Make Circular VAE`, `Offset Image` to verify seams |
| `ComfyUI-MakeSeamlessTexture` | Post-process a non-tiling source into a tile (offset/radial/half-shift) |
| `ComfyUI-AdvancedTiling` | Rotated / non-square tile shapes |
| `comfyui_controlnet_aux` | Tile/Canny/Depth/Normal preprocessors → lock structure during restyle |
| `ComfyUI-TextureAlchemy` | Extract & pack PBR maps (BaseColor/Normal/Rough/Height/AO, ORM/RMA) |
| `ComfyUI-Marigold` | Depth/normal estimation backing the PBR extractor |
| `ComfyUI-Chord` (Ubisoft) | Single image → complete PBR material set |

## REST API for agents / engine integration

A high-level gateway (`gateway/`, service `gateway`) fronts ComfyUI with simple
async operations — `/restyle`, `/make-seamless`, `/pbr` — so AI agents and the
s&box engine can drive the pipeline without building node graphs. Swagger UI at
`http://<host>:8080/docs`, OpenAPI at `/openapi.json`. See `gateway/README.md`
(includes the s&box C# client). Started by compose alongside ComfyUI.

## Requirements

- Docker + Docker Compose (v2)
- NVIDIA GPU on the Docker host + NVIDIA Container Toolkit (`--gpus all` support)

## Quick start

```bash
make build          # build comfyui-texture:latest  (large: CUDA base + torch)
make gpu-test       # confirm the container sees the GPU
make models         # download starter checkpoints/controlnet/upscaler/CHORD
make up             # start → http://localhost:8188
make logs           # follow logs
make down           # stop
```

Plain Docker equivalent (no compose):

```bash
docker build -t comfyui-texture .
docker run --gpus all -p 8188:8188 \
  -v "$PWD/data/models:/data/models" \
  -v "$PWD/data/input:/opt/ComfyUI/input" \
  -v "$PWD/data/output:/opt/ComfyUI/output" \
  -v "$PWD/data/user:/opt/ComfyUI/user" \
  comfyui-texture
```

## Layout

```
comfyui/
├── Dockerfile                 # nvidia/cuda:12.4.1 + torch cu124 + ComfyUI + nodes
├── docker-compose.yml         # GPU reservation, volume mounts, port 8188
├── entrypoint.sh              # seeds baked nodes, ensures model dirs, launches
├── extra_model_paths.yaml     # points model categories at /data/models
├── scripts/
│   ├── install_custom_nodes.sh   # baked-node installer (build time)
│   ├── download_models.sh        # host-side starter model downloader
│   └── gen_workflow.py           # builds + validates the workflow JSON
├── workflows/                 # generated UI + API workflow JSON
└── data/                      # host-mounted (models / input / output / user)
```

Models are **not** baked into the image — they live in `./data/models` and are
mapped via `extra_model_paths.yaml`. User-installed nodes persist in the
`comfyui_custom_nodes` volume; the baked stack is re-seeded on every start.

## The texture workflow

Ready-to-use graph: **`workflows/texture_img2img_tiling.json`** (also auto-listed in
the ComfyUI workflow sidebar — it's mounted at `data/user/default/workflows/`).
An API-format copy (`workflows/texture_img2img_tiling_api.json`) is included for
headless `/prompt` use. Both are generated and type-validated by
`scripts/gen_workflow.py` against the live `/object_info` schema.

Existing texture → more-realistic variant, tiling guaranteed:

1. `LoadImage` (your texture) → `VAEEncode`
2. `CheckpointLoaderSimple` (photoreal SDXL) → **`SeamlessTile` (tiling=enable)** patches the model
3. `MakeCircularVAE` (tiling=enable) → circular decode VAE
4. `TilePreprocessor` + `ControlNetApplyAdvanced` (Tile CN, strength 0.6) → locks structure
5. `KSampler` at **denoise 0.45**, dpmpp_2m / karras → adds realism without drifting
6. `VAEDecode` (circular) → `ImageUpscaleWithModel` (4x-UltraSharp) → still tileable
7. `OffsetImage` (50/50) → `SaveImage` seam check, plus `SaveImage` of the final tile
8. PBR branch: `ChordLoadModel` → `ChordMaterialEstimation` → BaseColor / Normal / Roughness / Metalness

**If your source does NOT already tile:** insert `SeamlessTextureRadialMask` (or
`SeamlessTextureHalfShift`) between `LoadImage` and `VAEEncode` to seam it first.
Tune realism with the `KSampler` **denoise** (lower = closer to original) and the
ControlNet **strength** (higher = stricter structure lock).

### Run it

> **Mounts:** the data dir (`./data` by default) is bind-mounted for
> models/input/output/user. If your Docker daemon runs on a separate host (Docker
> Desktop / WSL), the daemon resolves the bind path on **that** host, so set
> `TEXFORGE_DATA` in a local `.env` to the host path — see `.env.example`. With that
> set, `./data` round-trips: drop a texture in `./data/input` and the container sees
> it; renders land in `./data/output`; the workflow lives in
> `./data/user/default/workflows/`.

```bash
make up                          # start ComfyUI
make models                      # download models into ./data/models (runs in-container)
# CHORD's weights are GATED on Hugging Face. Accept the license at
# https://huggingface.co/Ubisoft/ubisoft-laforge-chord then:
#   make models HF_TOKEN=hf_xxxxx     # fetches the one remaining file (chord_v1.safetensors)
# The core img2img+tiling+realism path works WITHOUT CHORD; it only gates the PBR branch.
# copy your source texture into ./data/input/your_texture.png  (direct, no cp needed)
# open http://localhost:8188 → Workflows tab → refresh → texture_img2img_tiling → Queue
# results appear in ./data/output
```

In the workflow, set the `LoadImage` node to the filename you placed in
`./data/input`, then Queue. The `make workflow` / `put-input` / `get-output`
targets remain as conveniences but are no longer required now that `./data` is shared.

### Regenerate the workflow (after adding/swapping nodes)

```bash
docker compose exec -T comfyui python -c "import urllib.request;open('/tmp/oi.json','wb').write(urllib.request.urlopen('http://127.0.0.1:8188/object_info').read())"
docker compose cp comfyui:/tmp/oi.json /tmp/oi.json
python3 scripts/gen_workflow.py /tmp/oi.json
```
