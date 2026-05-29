# Texture Pipeline — Agent Guide

You (an AI agent) drive a texture-generation pipeline through a REST gateway.
This file is the complete contract. The machine-readable schema is at
`GET {BASE}/openapi.json`; this prose explains how to actually use it.

## Base URL

```
BASE = http://<host>:8080        # default when running via docker compose
```
Discover what's installed first: `GET {BASE}/capabilities` returns the available
checkpoints, controlnets, upscale models, samplers, schedulers, and whether CHORD
(PBR) is available. `GET {BASE}/health` reports gateway + ComfyUI reachability.

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

### POST /restyle — existing texture → realistic variant, tiling preserved/created
Output label: `result` (PNG; 4× the working size when `upscale=true`).

| param | type | default | notes |
|---|---|---|---|
| prompt | string | photoreal default | what to push the texture toward |
| negative_prompt | string | default | |
| denoise | float 0–1 | 0.45 | lower = closer to source, higher = more new detail |
| steps | int | 28 | |
| cfg | float | 6.5 | |
| sampler_name | string | dpmpp_2m | must be in `/capabilities.samplers` |
| scheduler | string | karras | must be in `/capabilities.schedulers` |
| seed | int | -1 | -1 = random |
| tiling | enum | enable | `enable`/`x_only`/`y_only`/`disable` |
| use_controlnet | bool | true | Tile-ControlNet structure lock |
| controlnet_strength | float 0–2 | 0.6 | |
| upscale | bool | true | tiling-aware model upscale |
| upscale_pad | int | 64 | circular pad for seamless upscale |
| input_size | int or null | 1024 | resize longest side (aspect kept); null = as-is |
| checkpoint / controlnet / upscale_model | string | SDXL defaults | must exist in `/capabilities` |

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

### POST /pbr — PBR material maps via CHORD
Outputs labels: `basecolor`, `normal`, `roughness`, `metalness` (4 PNGs).

| param | type | default | notes |
|---|---|---|---|
| backend | enum | chord | only `chord` currently |
| chord_model | string | chord_v1.safetensors | must exist; check `/capabilities.chord_available` |
| input_size | int or null | 1024 | CHORD works best at 1024 |

## Worked example (curl)

```bash
BASE=http://localhost:8080
B64=$(base64 -w0 my_texture.png)

JOB=$(curl -s $BASE/restyle -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$B64\",\"denoise\":0.45,\"tiling\":\"enable\",\"upscale\":true}" \
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
