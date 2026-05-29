"""Texture Pipeline Gateway — a high-level REST API over ComfyUI for AI agents.

Operations: /restyle (img2img realism + tiling), /make-seamless, /pbr.
Async job model with live progress via ComfyUI's websocket. Inputs accepted as
multipart file upload OR JSON (image_base64 / image_url). OpenAPI at /openapi.json,
Swagger UI at /docs.
"""
import asyncio
import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse
from PIL import Image

from .comfy import ComfyClient, PromptError
from . import prompts
from .models import (RestyleParams, SeamlessParams, PBRParams, FluxKontextParams,
                     SeamlessMethod, ControlType, JobResponse, JobStatus, OutputItem, Capabilities)

COMFY_URL = os.environ.get("COMFY_URL", "http://comfyui:8188")

JOBS: dict = {}          # job_id -> dict
PID_INDEX: dict = {}     # comfy prompt_id -> job_id
_current_pid: Optional[str] = None
_oi_cache = {"t": 0.0, "data": None}


async def _sweeper():
    """Backstop reconciliation for jobs nobody is actively polling."""
    while True:
        await asyncio.sleep(10)
        for job in list(JOBS.values()):
            if job["status"] in (JobStatus.queued, JobStatus.running):
                try:
                    await _reconcile(job)
                except Exception:
                    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.comfy = ComfyClient(COMFY_URL)
    ws_task = asyncio.create_task(app.state.comfy.listen(_on_event))
    sweep_task = asyncio.create_task(_sweeper())
    yield
    ws_task.cancel()
    sweep_task.cancel()
    await app.state.comfy.aclose()


app = FastAPI(
    title="Texture Pipeline Gateway",
    version="1.0.0",
    description="High-level REST API to drive the ComfyUI texture pipeline "
                "(realistic img2img restyle with preserved tiling, seamless conversion, PBR maps).",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Websocket event handling -> job state
# --------------------------------------------------------------------------- #
async def _on_event(msg: dict):
    global _current_pid
    t = msg.get("type")
    d = msg.get("data", {}) or {}
    pid = d.get("prompt_id")
    if t == "execution_start" and pid in PID_INDEX:
        _set(pid, status=JobStatus.running)
    elif t == "executing":
        node = d.get("node")
        if node is None and pid in PID_INDEX:
            await _finalize(pid)
        elif pid in PID_INDEX:
            _current_pid = pid
            _set(pid, status=JobStatus.running)
    elif t == "progress":
        target = pid if pid in PID_INDEX else _current_pid
        if target in PID_INDEX:
            mx = d.get("max") or 1
            _set(target, progress=min(1.0, (d.get("value", 0) or 0) / mx))
    elif t == "execution_error" and pid in PID_INDEX:
        _set(pid, status=JobStatus.failed,
             error=d.get("exception_message") or "execution error")


def _set(pid: str, **kw):
    job = JOBS.get(PID_INDEX.get(pid))
    if job:
        job.update(kw)


def _history_error(entry: dict) -> Optional[str]:
    """Extract a human error from a ComfyUI history entry, or None if it succeeded."""
    status = entry.get("status", {}) or {}
    if status.get("status_str") == "error" or status.get("completed") is False:
        for mtype, mdata in status.get("messages", []):
            if mtype == "execution_error":
                return (mdata or {}).get("exception_message") or "execution error"
        if status.get("status_str") == "error":
            return "execution error"
    return None


async def _finalize(pid: str):
    job = JOBS.get(PID_INDEX.get(pid))
    if not job or job["status"] in (JobStatus.completed, JobStatus.failed):
        return  # terminal already (e.g. an execution_error arrived first)
    try:
        hist = await app.state.comfy.history(pid)
    except Exception as e:
        job.update(status=JobStatus.failed, error=f"history fetch failed: {e}")
        return
    entry = hist.get(pid, {})
    err = _history_error(entry)
    if err:
        job.update(status=JobStatus.failed, error=err)
        return
    outputs = []
    for node_id, label in job["labels"].items():
        for im in entry.get("outputs", {}).get(node_id, {}).get("images", []):
            outputs.append({"label": label, "filename": im["filename"],
                            "subfolder": im.get("subfolder", ""), "type": im.get("type", "output"),
                            "url": f"/jobs/{job['id']}/outputs/{label}"})
    job.update(status=JobStatus.completed, progress=1.0, outputs=outputs)


async def _reconcile(job: dict):
    """Self-heal a non-terminal job by polling ComfyUI directly — covers any websocket
    event the gateway missed (reconnect, dropped frame, container restart). Called on
    every /jobs poll and by the background sweeper, so a stuck job can't wedge a client."""
    if job["status"] in (JobStatus.completed, JobStatus.failed):
        return
    pid = job.get("prompt_id")
    if not pid:
        return
    try:
        hist = await app.state.comfy.history(pid)
    except Exception:
        return
    if pid in hist:                     # ComfyUI finished it (success or error)
        await _finalize(pid)
        return
    # Not in history yet — still queued/running? If it has vanished from both, it was
    # lost; fail it after a short grace so the client stops waiting.
    if not await app.state.comfy.in_queue(pid):
        if time.time() - job.get("created", 0) > 30:
            job.update(status=JobStatus.failed,
                       error="job vanished from ComfyUI (no history, not queued) — likely a "
                             "ComfyUI restart or a rejected graph")


# --------------------------------------------------------------------------- #
# Input resolution (multipart file OR JSON base64/url)
# --------------------------------------------------------------------------- #
async def _read_request(request: Request, Model):
    ct = request.headers.get("content-type", "")
    if ct.startswith("multipart/form-data"):
        form = await request.form()
        raw = form.get("params") or "{}"
        params = Model.model_validate_json(raw if isinstance(raw, str) else "{}")
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            return params, await upload.read(), getattr(upload, "filename", "upload.png")
    else:
        params = Model.model_validate(await request.json())

    if params.image_base64:
        b = params.image_base64.split(",", 1)[-1]
        return params, base64.b64decode(b), "upload.png"
    if params.image_url:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(params.image_url)
            r.raise_for_status()
            return params, r.content, os.path.basename(params.image_url) or "upload.png"
    raise HTTPException(400, "No image supplied (use multipart 'file', or JSON image_base64 / image_url).")


def _compute_scaled(data: bytes, input_size: Optional[int]):
    if not input_size:
        return None
    try:
        w, h = Image.open(io.BytesIO(data)).size
    except Exception:
        return None
    m = max(w, h)
    if m == input_size:
        return None
    s = input_size / m
    nw = max(8, round(w * s / 8) * 8)
    nh = max(8, round(h * s / 8) * 8)
    return None if (nw, nh) == (w, h) else (nw, nh)


async def _fetch_optional_image(b64: Optional[str], url: Optional[str]) -> Optional[bytes]:
    """Resolve a secondary image (e.g. IP-Adapter style reference) from base64/url."""
    if b64:
        return base64.b64decode(b64.split(",", 1)[-1])
    if url:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.content
    return None


async def _start(operation: str, build, params, data, fname) -> dict:
    comfy: ComfyClient = app.state.comfy
    job_id = uuid.uuid4().hex[:12]
    # Unique per-job input filename so overlapping requests can't clobber each other's
    # upload (a fixed name + overwrite would make two jobs read the same image).
    name = await comfy.upload_image(data, f"gw_{job_id}.png")
    prefix = f"api/{job_id}"

    if operation == "make-seamless":
        prompt, labels = build(name, params, prefix)
    elif operation == "restyle":
        scaled = _compute_scaled(data, getattr(params, "input_size", None))
        ip_name = None
        if getattr(params, "ip_adapter", False):
            ip_bytes = await _fetch_optional_image(
                getattr(params, "ip_adapter_image_base64", None),
                getattr(params, "ip_adapter_image_url", None))
            if ip_bytes is not None:
                ip_name = await comfy.upload_image(ip_bytes, f"gw_{job_id}_ip.png")
        prompt, labels = build(name, params, prefix, scaled, ip_name)
    else:
        scaled = _compute_scaled(data, getattr(params, "input_size", None))
        prompt, labels = build(name, params, prefix, scaled)

    job = {"id": job_id, "operation": operation, "status": JobStatus.queued,
           "progress": 0.0, "error": None, "outputs": [], "labels": labels,
           "prompt_id": None, "created": time.time()}
    JOBS[job_id] = job
    try:
        pid = await comfy.submit(prompt)
    except PromptError as e:
        job.update(status=JobStatus.failed, error="prompt rejected")
        raise HTTPException(400, {"message": "ComfyUI rejected the graph", "detail": e.body})
    job["prompt_id"] = pid
    PID_INDEX[pid] = job_id
    return job


def _job_response(job: dict) -> JobResponse:
    return JobResponse(job_id=job["id"], status=job["status"], operation=job["operation"],
                       progress=round(job["progress"], 3), error=job["error"],
                       outputs=[OutputItem(**o) for o in job["outputs"]])


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    ok = True
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            ok = (await c.get(f"{COMFY_URL}/system_stats")).status_code == 200
    except Exception:
        ok = False
    return {"gateway": "ok", "comfyui_reachable": ok}


@app.get("/capabilities", response_model=Capabilities)
async def capabilities():
    comfy: ComfyClient = app.state.comfy
    if time.time() - _oi_cache["t"] > 30 or _oi_cache["data"] is None:
        _oi_cache["data"] = await comfy.object_info()
        _oi_cache["t"] = time.time()
    oi = _oi_cache["data"]
    co = comfy.combo_options
    chord = "chord_v1.safetensors" in co(oi, "ChordLoadModel", "ckpt_name")
    unets = co(oi, "UNETLoader", "unet_name")
    flux_kontext = ("FluxKontextImageScale" in oi and "ReferenceLatent" in oi
                    and any("kontext" in str(n).lower() for n in unets))
    return Capabilities(
        comfyui_reachable=True,
        checkpoints=co(oi, "CheckpointLoaderSimple", "ckpt_name"),
        controlnets=co(oi, "ControlNetLoader", "control_net_name"),
        upscale_models=co(oi, "UpscaleModelLoader", "model_name"),
        vaes=co(oi, "VAELoader", "vae_name"),
        samplers=co(oi, "KSampler", "sampler_name"),
        schedulers=co(oi, "KSampler", "scheduler"),
        chord_available=chord,
        stablematerials_available="MatForgerMaterialEstimation" in oi,
        ipadapter_available="IPAdapterAdvanced" in oi,
        florence2_available="Florence2Run" in oi,
        flux_kontext_available=flux_kontext,
        ipadapter_models=co(oi, "IPAdapterModelLoader", "ipadapter_file"),
        clip_vision_models=co(oi, "CLIPVisionLoader", "clip_name"),
        control_types=[c.value for c in ControlType],
        seamless_methods=[m.value for m in SeamlessMethod],
        operations=["restyle", "restyle-flux", "make-seamless", "pbr"],
    )


@app.post("/restyle", response_model=JobResponse,
          summary="Restyle an existing texture (img2img) with preserved/created tiling")
async def restyle(request: Request):
    params, data, fname = await _read_request(request, RestyleParams)
    return _job_response(await _start("restyle", prompts.build_restyle, params, data, fname))


@app.post("/restyle-flux", response_model=JobResponse,
          summary="Instruction-based restyle via FLUX.1 Kontext (NOT seamlessly tileable)")
async def restyle_flux(request: Request):
    params, data, fname = await _read_request(request, FluxKontextParams)
    return _job_response(await _start("restyle-flux", prompts.build_restyle_flux, params, data, fname))


@app.post("/make-seamless", response_model=JobResponse,
          summary="Make a non-tiling image seamless (no diffusion)")
async def make_seamless(request: Request):
    params, data, fname = await _read_request(request, SeamlessParams)
    return _job_response(await _start("make-seamless", prompts.build_seamless, params, data, fname))


@app.post("/pbr", response_model=JobResponse,
          summary="Estimate PBR maps (BaseColor/Normal/Roughness/Metalness) via CHORD")
async def pbr(request: Request):
    params, data, fname = await _read_request(request, PBRParams)
    return _job_response(await _start("pbr", prompts.build_pbr, params, data, fname))


@app.get("/jobs/{job_id}", response_model=JobResponse, summary="Poll job status / progress / outputs")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    # Self-heal on every poll: if the ws missed this job's completion/error, catch it here
    # so the client's next poll reflects ComfyUI's true state instead of hanging on 'running'.
    if job["status"] in (JobStatus.queued, JobStatus.running):
        await _reconcile(job)
    return _job_response(job)


@app.get("/jobs/{job_id}/outputs/{label}", summary="Download a result image (PNG)")
async def get_output(job_id: str, label: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    match = next((o for o in job["outputs"] if o["label"] == label), None)
    if not match:
        raise HTTPException(404, f"no output '{label}' (status={job['status']})")
    data = await app.state.comfy.fetch_view(match["filename"], match["subfolder"], match["type"])
    return Response(content=data, media_type="image/png")
