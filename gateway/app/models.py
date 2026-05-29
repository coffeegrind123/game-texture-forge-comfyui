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
    stablematerials = "stablematerials"


class RestyleMethod(str, Enum):
    img2img = "img2img"      # add-noise-to-source then denoise (classic)
    unsample = "unsample"    # invert source to its own noise, then resample (tighter layout lock)


class ControlType(str, Enum):
    tile = "tile"            # preserves structure AND colour (least restyle freedom)
    depth = "depth"          # surface relief only — best "same layout, new material"
    canny = "canny"          # hard edges (grout/mortar/panel lines)
    lineart = "lineart"      # softer edges, good for organic surfaces
    scribble = "scribble"    # loosest structural hint, most restyle freedom


class ImageRef(BaseModel):
    """Ways to supply the input image when not using multipart file upload."""
    image_base64: Optional[str] = Field(None, description="Base64-encoded image bytes (data-URI prefix optional).")
    image_url: Optional[str] = Field(None, description="URL the gateway will fetch the image from.")


class RestyleParams(ImageRef):
    # --- prompt / steering ------------------------------------------------- #
    prompt: str = Field(
        "photorealistic surface, sharp fine detail, physically based, high detail",
        description="Base/fallback positive prompt. When auto_caption is on this is unused "
                    "(the caption replaces it); when off it is the content portion of the prompt.")
    style: str = Field(
        "",
        description="Transformation modifier appended to the prompt, e.g. "
                    "'(covered in thick green moss:1.3), damp'. This is what makes the output "
                    "DIFFERENT from the source. ComfyUI weight syntax (phrase:weight) is honoured.")
    quality_suffix: str = Field(
        "diffuse even lighting, orthographic, seamless texture, PBR albedo, highly detailed, 8k",
        description="Quality/tiling tags appended last.")
    auto_caption: bool = Field(
        True,
        description="Caption the input (Florence-2 PromptGen) so the prompt describes what the "
                    "texture actually IS, then append `style` + `quality_suffix`. The single biggest "
                    "fix for 'it just outputs the same texture'.")
    caption_task: str = Field(
        "more_detailed_caption",
        description="Florence2Run task. 'more_detailed_caption' (universal) or "
                    "'prompt_gen_mixed_caption' (PromptGen models: tags+description).")
    negative_prompt: str = (
        "perspective, shadows, depth of field, vignette, 3d render, tilted, border, frame, "
        "cartoon, blurry, low detail, worst quality, lowres, jpeg artifacts, watermark, signature, "
        "text, visible seam")

    # --- diffusion --------------------------------------------------------- #
    method: RestyleMethod = Field(RestyleMethod.img2img, description="img2img or unsample.")
    denoise: float = Field(0.6, ge=0.0, le=1.0,
                           description="Lower = closer to source; higher = more new detail. "
                                       "0.55-0.70 is the restyle band; <0.45 just refines.")
    steps: int = Field(28, ge=1, le=150)
    cfg: float = Field(6.5, ge=1.0, le=30.0)
    sampler_name: str = "dpmpp_2m"
    scheduler: str = "karras"
    seed: int = Field(-1, description="-1 = random.")
    tiling: Tiling = Tiling.enable

    # --- structure control (ControlNet) ----------------------------------- #
    use_controlnet: bool = Field(True, description="Structure lock via ControlNet.")
    control_type: ControlType = Field(
        ControlType.tile,
        description="Which structural signal to lock. 'depth'/'canny' restyle far more than 'tile'.")
    controlnet_strength: float = Field(0.4, ge=0.0, le=2.0,
                                       description="Lower = more restyle freedom. 0.4 is a good restyle default.")
    controlnet_start_percent: float = Field(0.0, ge=0.0, le=1.0)
    controlnet_end_percent: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Release ControlNet at this fraction of sampling. <1.0 lets the back half "
                    "repaint the material — the key restyle lever. 0.4-0.6 recommended.")
    controlnet_union: bool = Field(
        False,
        description="Set true when `controlnet` is the xinsir Union model — wires "
                    "SetUnionControlNetType from control_type.")

    # --- IP-Adapter (style injection) -------------------------------------- #
    ip_adapter: bool = Field(False, description="Inject a reference look via IP-Adapter (style transfer).")
    ip_adapter_weight: float = Field(0.8, ge=-1.0, le=5.0)
    ip_adapter_weight_type: str = Field(
        "style transfer",
        description="IPAdapterAdvanced weight_type, e.g. 'style transfer', 'style transfer precise'.")
    ip_adapter_model: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    clip_vision_model: str = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    ip_adapter_image_base64: Optional[str] = Field(
        None, description="Optional separate style-reference image. If omitted, the input texture "
                          "is used as its own reference (self-variation).")
    ip_adapter_image_url: Optional[str] = None

    # --- output / scale ---------------------------------------------------- #
    upscale: bool = Field(True, description="Tiling-aware model upscale pass after decode.")
    upscale_pad: int = Field(64, ge=0, le=512, description="Circular pad for seamless upscale.")
    input_size: Optional[int] = Field(1024, description="Resize longest side to this (aspect-preserved, /8). null = as-is.")

    # --- models ------------------------------------------------------------ #
    checkpoint: str = "Juggernaut-XL_v9.safetensors"
    controlnet: str = "controlnet-tile-sdxl-1.0.safetensors"
    upscale_model: str = "4x-UltraSharp.pth"
    caption_model: str = "MiaoshouAI/Florence-2-large-PromptGen-v2.0"


class FluxKontextParams(ImageRef):
    """Instruction-based restyle via FLUX.1 Kontext [dev]. NOT seamlessly tileable
    (DiT model — the circular-padding tiling trick does not apply); use for one-off
    restyles where seamlessness is not required, or fix seams afterwards."""
    prompt: str = Field(..., description="Edit instruction, e.g. 'make this brick wall mossy and "
                                         "weathered, keep the layout'.")
    guidance: float = Field(2.5, ge=0.0, le=10.0, description="FluxGuidance value (2.5 recommended for Kontext).")
    steps: int = Field(20, ge=1, le=60)
    seed: int = Field(-1, description="-1 = random.")
    sampler_name: str = "euler"
    scheduler: str = "simple"
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    input_size: Optional[int] = Field(1024, description="Resize longest side before encoding.")
    unet_name: str = "flux1-dev-kontext_fp8_scaled.safetensors"
    clip_name1: str = "t5xxl_fp16.safetensors"
    clip_name2: str = "clip_l.safetensors"
    vae_name: str = "ae.safetensors"


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
    # StableMaterials backend params (ignored by chord):
    sm_repo: str = Field("gvecchio/StableMaterials",
                         description="StableMaterials/MatForger repo id (stablematerials backend).")
    material_prompt: str = Field("", description="Text prompt; empty = condition on the input image.")
    steps: int = Field(25, ge=1, le=200, description="stablematerials only")
    cfg: float = Field(6.0, ge=1.0, le=30.0, description="stablematerials only")
    tileable: bool = Field(True, description="stablematerials only")


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
    stablematerials_available: bool
    ipadapter_available: bool
    florence2_available: bool
    flux_kontext_available: bool
    ipadapter_models: List[str]
    clip_vision_models: List[str]
    control_types: List[str]
    seamless_methods: List[str]
    operations: List[str]
