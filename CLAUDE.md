# Game Texture Forge (ComfyUI) — Project Guide

Single source of truth for this repo. A local **ComfyUI** texture pipeline plus a
**REST gateway** that lets AI agents / the s&box engine restyle game textures
(realistic, tiling preserved) and generate PBR maps. Read this top-to-bottom once.

> For the agent-facing API contract, see **`AGENTS.md`** (self-contained: endpoints,
> the submit→poll→download model, params). For pipeline internals see **`README.md`**;
> for the gateway + s&box C# client see **`gateway/README.md`**.

## 1. What it is

- **ComfyUI** (`comfyui-texture:latest`, CUDA) with a baked texture node stack:
  seamless-tiling (spinagon), MakeSeamlessTexture, AdvancedTiling, TextureAlchemy,
  comfyui_controlnet_aux, Marigold, **Ubisoft CHORD** (gated), ComfyUI-Manager,
  **ComfyUI-Florence2** (caption→prompt), **ComfyUI_IPAdapter_plus** (style transfer),
  **ComfyUI_Noise** (unsampling), and custom nodes in `custom_nodes/`:
  `comfyui-tiling-upscale` (`TilingAwareUpscale`) and `comfyui-texture-forge-nodes`
  (`TextureForgePromptCompose`, `MatForgerMaterialEstimation`).
- **Gateway** (`comfyui-gateway:latest`, FastAPI) on host port **8080** fronting
  ComfyUI (`:8188`) with `/restyle`, `/make-seamless`, `/pbr`, async jobs, OpenAPI.
- Two compose services: `comfyui` (GPU) and `gateway`. `make` targets wrap everything.

## 2. Build & run

```bash
make build              # build comfyui image (long: CUDA base + torch + nodes)
docker compose build gateway
make up                 # start both
make models             # download starter models INTO the container (/data/models)
make gpu-test           # confirm torch sees the GPU
```
The pipeline was verified end-to-end on an RTX 4090: img2img restyle → 4096² output,
**seam ratio 0.96** (tiles cleanly); PBR emits BaseColor/Normal/Roughness/Metalness.

## 3. Critical gotchas (these cost real time — read before debugging)

### The Docker daemon runs on the HOST (Docker Desktop / WSL), not in this shell
- **Bind mounts resolve on the daemon's host filesystem.** A WSL-relative `./data`
  mount silently resolves to a throwaway dir inside the Docker VM — files written here
  never reach the container and vice-versa. Fix: `TEXFORGE_DATA` in a **gitignored `.env`**
  points at the HOST path (e.g. `//c/users/<you>/.../comfyui/data`); compose uses
  `${TEXFORGE_DATA:-./data}`. With it set, `./data` round-trips both ways.
- **Published ports bind the host's localhost, not this shell.** `curl localhost:8080`
  from here returns nothing — introspect via `docker compose exec <svc> ...` instead.
- Build context streams fine regardless.

### transformers must be `<5`
transformers 5.x dropped the CLIP `text_model.` key prefix, which breaks loading models
saved under 4.x — notably CHORD's `chord_v1.safetensors` (`Missing key(s) … text_encoder…`).
Pinned `transformers>=4.50.3,<5` in the Dockerfile constraints (`/etc/pip-torch-constraints.txt`).
The image ships 4.57.6.

### The torch stack is pinned via `PIP_CONSTRAINT`
ComfyUI core imports `torchaudio` unconditionally; an unpinned node `requirements.txt`
once pulled a CUDA-13 torchaudio (`libcudart.so.13`) and crash-looped startup. All of
torch/torchvision/torchaudio (+ transformers) are pinned in the constraints file so no
node install can swap in a mismatched build.

### Custom nodes live in a named volume, seeded by the entrypoint
`comfyui_custom_nodes` is a named volume; `entrypoint.sh` copies any baked node that
isn't present into it on start. So baked nodes always load, and Manager-installed extras
persist. **`docker compose down -v` wipes this volume** (baked nodes re-seed; extras lost).

### Models are gitignored bind-mounts; CHORD is gated
`data/` (12 GB of weights + ComfyUI's user db/logs) is **never committed**. `make models`
downloads into `/data/models`. **CHORD is a gated HF repo** — accept the license at
huggingface.co/Ubisoft/ubisoft-laforge-chord, then `make models HF_TOKEN=hf_xxx`. A read
token is enough; the license-accept (not the token) is what unblocks the 403.

### `object_info` has two combo encodings
Some nodes expose combo options as a list at `[0]`; others use `["COMBO", {"options":[…]}]`
at `[1]`. The gateway's `combo_options` handles both — don't assume one.

### Cold load dominates latency
The first job after `up` reloads ~9 GB into VRAM (~7 min from the Windows mount); warm
jobs are tens of seconds. A forge runs **restyle → pbr = two GPU jobs**; loading CHORD can
evict SDXL, so back-to-back restyles may each reload. Keep ComfyUI warm for fast iteration.

## 4. The pipeline (how tiling/realism/PBR are wired)

- **Tiling:** `SeamlessTile` patches the model's conv layers to circular padding (works on
  img2img too), `MakeCircularVAE` does the same for decode → tiling preserved/created.
- **Realism without drift:** Tile-ControlNet + img2img at `denoise ~0.45` adds detail while
  locking structure.
- **Tiling-aware upscale:** stock ESRGAN upscale isn't tiling-aware (seam ratio ~2.3 at 4K).
  `TilingAwareUpscale` circular-pads → upscales → crops, dropping the seam ratio to ~0.96.
- **PBR:** CHORD `ChordMaterialEstimation` (single image → 4 maps). The minimal
  image-to-material graph needs only `ChordLoadModel + ChordMaterialEstimation` — no base model.
  Alternative backend: `MatForgerMaterialEstimation` (StableMaterials/MatForger, 5 maps incl. height).
- Workflows live in `workflows/`; regenerate with `scripts/gen_workflow.py <object_info.json>`
  (validates every link against the live schema).

### Restyle: "different but similar" (don't ship a near-copy)
The old defaults (denoise 0.45 + Tile-ControlNet full range + a content-free prompt) made the
restyle a *detail-preserving upscaler* — it just rebuilt the input. The fix, now the default:
- **Content-aware prompt:** `auto_caption=true` runs Florence-2 PromptGen on the input, then
  `TextureForgePromptCompose` joins `caption + style + quality_suffix` into the positive prompt.
  The caller's **`style`** (e.g. `"(covered in moss:1.3)"`) is what diverges the output. A prompt
  with no material/target gives img2img nothing to move toward, so it reconstructs the source.
- **`variation` is the primary divergence knob (default 0.8):** the goal is a NEW texture that only
  shallowly references the original (same material family via the caption, new everything else). It maps
  0→1 onto denoise (0.45→0.92) + controlnet_strength (0.50→0.12) + controlnet_end_percent (0.60→0.20),
  overriding those three. Lower it for a subtle restyle; null it to drive denoise/controlnet manually.
  Verified: variation 0.8 on a flat-blue tile texture → a genuinely new tiled surface, mean Δ 69/channel
  (vs ~15 for the old subtle default) while still reading as the same material kind.
- **Default steering is subtle realism + neutral colour:** `quality_suffix` pushes
  `(realistic natural color palette:1.2), (muted neutral tones:1.1), photorealistic, PBR` and the
  negative pushes away from oversaturation / blue cast / stylisation — so strong colours drift
  toward neutral and materials read more realistic, without changing the texture's identity. Bumping
  denoise/controlnet too far back toward 0.45/full-range is what caused the original "same texture" bug.
- **control_type** chooses the structural signal: `tile` preserves colour too (least change);
  `depth`/`canny`/`lineart`/`scribble` lock geometry only (more restyle). Use the xinsir Union
  model + `controlnet_union=true` to switch types without a new download.
- **IP-Adapter** (`ip_adapter=true`, `weight_type="style transfer"`) injects a reference look;
  **`method="unsample"`** (ComfyUI_Noise) inverts the source to its own noise for a tighter
  layout lock with a bigger appearance change than plain img2img.
- **/restyle-flux** (FLUX.1 Kontext) is instruction-based editing but **NOT tileable** (DiT —
  circular padding doesn't apply); separate endpoint, download with `TEXFORGE_FLUX=1`.

## 5. The gateway

- `gateway/app/`: `main.py` (routes + job manager), `comfy.py` (ComfyUI client + ws),
  `prompts.py` (graph builders), `models.py` (pydantic params).
- **Job completion is detected from the ws AND reconciled against ComfyUI `/history` +
  `/queue` on every `/jobs/{id}` poll** (plus a background sweeper). This is deliberate:
  ws-only detection could hang a job forever if an event was missed (reconnect, container
  restart) — reconciliation self-heals on the client's next poll. A ComfyUI-side failure is
  reported as `failed` with the real error, not `completed` with no outputs.
- Jobs are **in-memory** — a gateway restart clears job history (outputs persist in
  ComfyUI's `output/`). No in-flight job survives a restart.
- Pure HTTP to ComfyUI (`/upload/image`, `/prompt`, `/view`) — no shared volume needed.

## 6. Persistence / down-up safety

Plain `docker compose down` + `up` is safe: `data/` (bind mount) and `comfyui_custom_nodes`
(named volume) persist, and the transformers fix + node stack are baked into the image.
**Avoid `down -v`** (wipes the custom-nodes volume). First job after `up` is a cold load.

## 7. Secrets / git hygiene

- `.gitignore` excludes `data/`, `out/`, `.env`, `__pycache__`, logs.
- HF tokens are passed via env (`make models HF_TOKEN=…`), never written to a file.
- Machine-specific host paths live only in `.env` (gitignored); `.env.example` documents it.
- Repo verified clean for public: no secrets, no real author email, no host paths committed.

## 8. Related

The s&box consumer (`TextureGateway.cs`, `MgeTextureForge.cs`) lives in a **separate repo**
(`~/sbox-public/projects/mge/`), not here. This repo is the ComfyUI pipeline + gateway only.
