"""Request/response models for the texture pipeline gateway."""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class Tiling(str, Enum):
    enable = "enable"
    x_only = "x_only"
    y_only = "y_only"
    disable = "disable"


class SeamlessMethod(str, Enum):
    radial = "radial"
    halfshift = "halfshift"
    mirrored = "mirrored"


class PBRBackend(str, Enum):
    chord = "chord"


class ImageRef(BaseModel):
    """Ways to supply the input image when not using multipart file upload."""
    image_base64: Optional[str] = Field(None, description="Base64-encoded image bytes (data-URI prefix optional).")
    image_url: Optional[str] = Field(None, description="URL the gateway will fetch the image from.")


class RestyleParams(ImageRef):
    prompt: str = Field(
        "photorealistic surface, sharp fine detail, natural lighting, physically based, high detail, 4k",
        description="Positive prompt steering the realistic restyle.")
    negative_prompt: str = "blurry, cartoon, flat, painted, low detail, visible seam, watermark, text"
    denoise: float = Field(0.45, ge=0.0, le=1.0, description="Lower = closer to source; higher = more new detail.")
    steps: int = Field(28, ge=1, le=150)
    cfg: float = Field(6.5, ge=1.0, le=30.0)
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = Field(-1, description="-1 = random.")
    tiling: Tiling = Tiling.enable
    use_controlnet: bool = Field(True, description="Tile-ControlNet structure lock.")
    controlnet_strength: float = Field(0.6, ge=0.0, le=2.0)
    upscale: bool = Field(True, description="Tiling-aware model upscale pass after decode.")
    upscale_pad: int = Field(64, ge=0, le=512, description="Circular pad for seamless upscale.")
    input_size: Optional[int] = Field(1024, description="Resize longest side to this (aspect-preserved, /8). null = as-is.")
    checkpoint: str = "sd_xl_base_1.0.safetensors"
    controlnet: str = "controlnet-tile-sdxl-1.0.safetensors"
    upscale_model: str = "4x-UltraSharp.pth"


class SeamlessParams(ImageRef):
    method: SeamlessMethod = SeamlessMethod.radial
    inner_radius: float = Field(0.85, ge=0.0, le=1.5)
    outer_radius: float = Field(1.0, ge=0.1, le=2.0)
    scatter_strength: float = Field(1.0, ge=0.0, le=2.0, description="radial only")
    blend_curve: str = "cubic"
    orientation: str = Field("both", description="halfshift only: both|horizontal|vertical")


class PBRParams(ImageRef):
    backend: PBRBackend = PBRBackend.chord
    chord_model: str = "chord_v1.safetensors"
    input_size: Optional[int] = Field(1024, description="CHORD operates best at 1024.")


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class OutputItem(BaseModel):
    label: str
    filename: str
    subfolder: str
    type: str
    url: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    operation: str
    progress: float = 0.0
    error: Optional[str] = None
    outputs: List[OutputItem] = []


class Capabilities(BaseModel):
    comfyui_reachable: bool
    checkpoints: List[str]
    controlnets: List[str]
    upscale_models: List[str]
    vaes: List[str]
    samplers: List[str]
    schedulers: List[str]
    chord_available: bool
    seamless_methods: List[str]
    operations: List[str]
