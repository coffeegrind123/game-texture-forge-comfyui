"""ComfyUI API-format graph builders for each high-level operation.

Each builder returns (prompt, labels):
  prompt  - the ComfyUI /prompt graph (node-id -> {class_type, inputs})
  labels  - {save_node_id: output_label} so the gateway can name results.

Output slot indices are the stable ComfyUI conventions:
  CheckpointLoaderSimple -> MODEL(0) CLIP(1) VAE(2)
  ControlNetApplyAdvanced -> positive(0) negative(1)
  ChordMaterialEstimation -> basecolor(0) normal(1) roughness(2) metalness(3)
"""
import random
from .models import RestyleParams, SeamlessParams, PBRParams, SeamlessMethod


def _seed(s: int) -> int:
    return random.randint(0, 2**32 - 1) if s is None or s < 0 else s


def build_restyle(image_name: str, p: RestyleParams, prefix: str, scaled_wh=None):
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p.checkpoint}},
        "2": {"class_type": "SeamlessTile", "inputs": {"model": ["1", 0], "tiling": p.tiling.value, "copy_model": "Make a copy"}},
        "3": {"class_type": "MakeCircularVAE", "inputs": {"vae": ["1", 2], "tiling": p.tiling.value, "copy_vae": "Make a copy"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
    }
    img_src = ["4", 0]
    if scaled_wh is not None:
        w, h = scaled_wh
        g["40"] = {"class_type": "ImageScale",
                   "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                              "width": w, "height": h, "crop": "disabled"}}
        img_src = ["40", 0]

    g["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": img_src, "vae": ["1", 2]}}
    g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": p.prompt}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": p.negative_prompt}}

    pos, neg = ["6", 0], ["7", 0]
    if p.use_controlnet and p.controlnet_strength > 0:
        g["8"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": p.controlnet}}
        g["9"] = {"class_type": "TilePreprocessor", "inputs": {"image": img_src, "pyrUp_iters": 3, "resolution": 1024}}
        g["10"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": ["6", 0], "negative": ["7", 0], "control_net": ["8", 0],
                              "image": ["9", 0], "strength": p.controlnet_strength,
                              "start_percent": 0.0, "end_percent": 1.0}}
        pos, neg = ["10", 0], ["10", 1]

    g["11"] = {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "seed": _seed(p.seed), "steps": p.steps, "cfg": p.cfg,
                          "sampler_name": p.sampler_name, "scheduler": p.scheduler,
                          "positive": pos, "negative": neg, "latent_image": ["5", 0], "denoise": p.denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}

    final = ["12", 0]
    if p.upscale:
        g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": p.upscale_model}}
        g["14"] = {"class_type": "TilingAwareUpscale",
                   "inputs": {"upscale_model": ["13", 0], "image": ["12", 0], "pad": p.upscale_pad}}
        final = ["14", 0]

    g["15"] = {"class_type": "SaveImage", "inputs": {"images": final, "filename_prefix": f"{prefix}/result"}}
    return g, {"15": "result"}


def build_seamless(image_name: str, p: SeamlessParams, prefix: str):
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}}}
    if p.method == SeamlessMethod.radial:
        g["2"] = {"class_type": "SeamlessTextureRadialMask",
                  "inputs": {"image": ["1", 0], "inner_radius": p.inner_radius, "outer_radius": p.outer_radius,
                             "scatter_strength": p.scatter_strength, "blend_curve": p.blend_curve}}
    elif p.method == SeamlessMethod.halfshift:
        g["2"] = {"class_type": "SeamlessTextureHalfShift",
                  "inputs": {"image": ["1", 0], "inner_radius": p.inner_radius, "outer_radius": p.outer_radius,
                             "blend_curve": p.blend_curve, "orientation": p.orientation}}
    else:  # mirrored
        g["2"] = {"class_type": "SeamlessTextureMirroredCollage",
                  "inputs": {"image": ["1", 0], "blend_curve": p.blend_curve}}
    g["3"] = {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": f"{prefix}/seamless"}}
    return g, {"3": "result"}


def build_pbr(image_name: str, p: PBRParams, prefix: str, scaled_wh=None):
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}}}
    img_src = ["1", 0]
    if scaled_wh is not None:
        w, h = scaled_wh
        g["1b"] = {"class_type": "ImageScale",
                   "inputs": {"image": ["1", 0], "upscale_method": "lanczos",
                              "width": w, "height": h, "crop": "disabled"}}
        img_src = ["1b", 0]
    g["2"] = {"class_type": "ChordLoadModel", "inputs": {"ckpt_name": p.chord_model}}
    g["3"] = {"class_type": "ChordMaterialEstimation", "inputs": {"chord_model": ["2", 0], "image": img_src}}
    g["4"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": f"{prefix}/basecolor"}}
    g["5"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 1], "filename_prefix": f"{prefix}/normal"}}
    g["6"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 2], "filename_prefix": f"{prefix}/roughness"}}
    g["7"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 3], "filename_prefix": f"{prefix}/metalness"}}
    return g, {"4": "basecolor", "5": "normal", "6": "roughness", "7": "metalness"}
