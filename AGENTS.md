# Texture Pipeline — Agent Guide

You (an AI agent) drive a texture-generation pipeline through a REST gateway.
This file is the complete contract. The machine-readable schema is at
`GET {BASE}/openapi.json`; this prose explains how to actually use it.

## Base URL

```
BASE = http://<host>:8080        # default when running via docker compose
```
Discover what's installed first: `GET {BASE}/capabilities` returns the available
checkpoints, controlnets, upscale models, samplers, schedulers, control_types, ipadapter
and clip_vision models, and feature flags: `chord_available`, `stablematerials_available`,
`ipadapter_available`, `florence2_available`, `flux_kontext_available`.
`GET {BASE}/health` reports gateway + ComfyUI reachability.

## The contract: every operation is ASYNC

1. `POST` an operation → get `{ "job_id": "...", "status": "queued" }`.
2. Poll `GET {BASE}/jobs/{job_id}` until `status` is `completed` or `failed`.
   - `progress` is 0.0–1.0. Operations are GPU-bound: `restyle` can take minutes
     (longer on a cold model load). Poll every ~2s; do NOT assume instant results.
3. On `completed`, download each output: `GET {BASE}/jobs/{job_id}/outputs/{label}`
   returns a PNG. On `failed`, read the `error` field.

Job response shape:
```json
{ "job_id": "abc123", "status": "completed", "operation": "restyle",
  "progress": 1.0, "error": null,
  "outputs": [ { "label": "result", "filename": "...", "subfolder": "...",
                 "type": "output", "url": "/jobs/abc123/outputs/result" } ] }
```

## Supplying the input image (all operations)

Pick ONE:
- multipart form: field `file` (the image) + field `params` (a JSON string)
- JSON body with `"image_base64": "<...>"` (data-URI prefix optional)
- JSON body with `"image_url": "<...>"` (gateway fetches it)

## Operations

### POST /restyle — existing texture → restyled variant, tiling preserved/created
Output label: `result` (PNG; 4× the working size when `upscale=true`).

The defaults are tuned to produce a **recognisably different** texture (not a near-copy):
content-aware captioning + a higher denoise + a ControlNet that releases mid-sampling.
To actually change the look, set **`style`** (e.g. `"(covered in green moss:1.3), damp"`).

**Prompt / steering**

| param | type | default | notes |
|---|---|---|---|
| style | string | "" | **the transformation** — appended to the prompt; ComfyUI `(phrase:weight)` honoured. This is what makes the output different. |
| auto_caption | bool | true | caption the input so the prompt describes what it IS, then append `style` + `quality_suffix`. Turn off to use `prompt` verbatim. |
| caption_task | string | more_detailed_caption | or `prompt_gen_mixed_caption` (PromptGen: tags+desc) |
| prompt | string | photoreal default | base/content prompt used only when `auto_caption=false` |
| quality_suffix | string | tiling/PBR tags | appended last |
| negative_prompt | string | shadow/perspective-aware default | |

**Diffusion**

| param | type | default | notes |
|---|---|---|---|
| method | enum | img2img | `img2img` or `unsample` (invert→resample; tighter layout lock, bigger look change) |
| denoise | float 0–1 | 0.6 | 0.55–0.70 = restyle band; <0.45 only refines (img2img only) |
| steps / cfg | int / float | 28 / 6.5 | |
| sampler_name / scheduler | string | dpmpp_2m / karras | must be in `/capabilities` |
| seed | int | -1 | -1 = random |
| tiling | enum | enable | `enable`/`x_only`/`y_only`/`disable` |

**Structure control (ControlNet)**

| param | type | default | notes |
|---|---|---|---|
| use_controlnet | bool | true | structure lock |
| control_type | enum | tile | `tile`/`depth`/`canny`/`lineart`/`scribble`. `depth`/`canny` restyle far more than `tile`. |
| controlnet_strength | float 0–2 | 0.4 | lower = more restyle freedom |
| controlnet_start_percent | float 0–1 | 0.0 | |
| controlnet_end_percent | float 0–1 | 0.5 | **release point** — <1.0 lets the back half repaint material. Key restyle lever. |
| controlnet_union | bool | false | set true when `controlnet` is the xinsir Union model |
| controlnet | string | tile sdxl | use `controlnet-union-sdxl-1.0.safetensors` + `controlnet_union:true` to switch types |

**IP-Adapter (style injection)**

| param | type | default | notes |
|---|---|---|---|
| ip_adapter | bool | false | inject a reference look (style transfer) while ControlNet holds layout |
| ip_adapter_weight | float | 0.8 | |
| ip_adapter_weight_type | string | style transfer | e.g. `style transfer`, `style transfer precise` |
| ip_adapter_image_base64 / ip_adapter_image_url | string | null | optional separate style reference; omit = self-variation from the input |

**Output / models**

| param | type | default | notes |
|---|---|---|---|
| upscale / upscale_pad | bool / int | true / 64 | tiling-aware model upscale |
| input_size | int or null | 1024 | resize longest side (aspect kept); null = as-is |
| checkpoint | string | Juggernaut-XL_v9 | must exist in `/capabilities` |
| controlnet / upscale_model / caption_model | string | defaults | must exist |

### POST /restyle-flux — instruction-based restyle via FLUX.1 Kontext
**NOT seamlessly tileable** (DiT model — the circular-padding trick doesn't apply). Use for
one-off restyles where seamlessness isn't required. Requires the FLUX models (download with
`TEXFORGE_FLUX=1`); check `/capabilities.flux_kontext_available`. Output label: `result`.

| param | type | default | notes |
|---|---|---|---|
| prompt | string | (required) | edit instruction, e.g. "make this brick wall mossy, keep the layout" |
| guidance | float | 2.5 | FluxGuidance |
| steps | int | 20 | |
| sampler_name / scheduler | string | euler / simple | |
| width / height | int | 1024 | output canvas |
| seed | int | -1 | |
| input_size | int or null | 1024 | resize longest side before encoding |
| unet_name / clip_name1 / clip_name2 / vae_name | string | FLUX Kontext defaults | must exist |

### POST /make-seamless — make a non-tiling image tile (no diffusion, fast)
Output label: `result`.

| param | type | default | notes |
|---|---|---|---|
| method | enum | radial | `radial`/`halfshift`/`mirrored` |
| inner_radius | float | 0.85 | radial/halfshift |
| outer_radius | float | 1.0 | radial/halfshift |
| scatter_strength | float | 1.0 | radial only |
| blend_curve | string | cubic | cosine/linear/smoothstep/smootherstep/quadratic/cubic |
| orientation | string | both | halfshift only: both/horizontal/vertical |

### POST /pbr — PBR material maps
Two backends. **chord** (default): labels `basecolor`, `normal`, `roughness`, `metalness` (4 PNGs).
**stablematerials** (gvecchio StableMaterials/MatForger): adds a `height` map → labels
`basecolor`, `normal`, `height`, `roughness`, `metalness` (5 PNGs); auto-downloads on first use.

| param | type | default | notes |
|---|---|---|---|
| backend | enum | chord | `chord` or `stablematerials` (check `/capabilities.chord_available` / `.stablematerials_available`) |
| chord_model | string | chord_v1.safetensors | chord backend; must exist |
| input_size | int or null | 1024 | chord works best at 1024; stablematerials native 512 |
| sm_repo | string | gvecchio/StableMaterials | stablematerials backend repo id (MatForger also works) |
| material_prompt | string | "" | stablematerials: empty = condition on the input image |
| steps / cfg / tileable | int / float / bool | 25 / 6.0 / true | stablematerials only |

## Worked example (curl)

```bash
BASE=http://localhost:8080
B64=$(base64 -w0 my_texture.png)

# `style` is what makes the output DIFFERENT from the input (the rest is auto-captioned).
JOB=$(curl -s $BASE/restyle -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$B64\",\"style\":\"(covered in green moss:1.3), damp weathered stone\",\"denoise\":0.6,\"tiling\":\"enable\",\"upscale\":true}" \
  | jq -r .job_id)

# poll until done
while :; do
  S=$(curl -s $BASE/jobs/$JOB | jq -r .status)
  [ "$S" = completed ] && break
  [ "$S" = failed ] && { curl -s $BASE/jobs/$JOB | jq .error; exit 1; }
  sleep 2
done

curl -s $BASE/jobs/$JOB/outputs/result -o restyled.png
```

## Errors

- Bad params / unknown model name → `400` with `{message, detail}` (detail carries
  ComfyUI's node validation errors — fix the offending param, often a model name not
  in `/capabilities`).
- Runtime failure → job `status: "failed"`, human-readable `error`.

## Notes

- Jobs are in-memory; a gateway restart clears job history (output files persist in
  ComfyUI's `output/`). Download results before assuming long-term availability.
- For a typed client, generate one from `{BASE}/openapi.json`.
- Pipeline internals (how tiling/realism/PBR are wired) are in `README.md` and
  `gateway/README.md`; you do NOT need them to use the API — this file is enough.
```
