"""ComfyUI API-format graph builders for each high-level operation.

Each builder returns (prompt, labels):
  prompt  - the ComfyUI /prompt graph (node-id -> {class_type, inputs})
  labels  - {save_node_id: output_label} so the gateway can name results.

Output slot indices are the stable ComfyUI conventions:
  CheckpointLoaderSimple   -> MODEL(0) CLIP(1) VAE(2)
  ControlNetApplyAdvanced  -> positive(0) negative(1)
  ChordMaterialEstimation  -> basecolor(0) normal(1) roughness(2) metalness(3)
  Florence2Run             -> image(0) mask(1) caption(2) data(3)
  MatForgerMaterialEstimation -> basecolor(0) normal(1) height(2) roughness(3) metalness(4)
"""
import random
from .models import (RestyleParams, SeamlessParams, PBRParams, FluxKontextParams,
                     SeamlessMethod, PBRBackend, RestyleMethod, ControlType)


def _seed(s: int) -> int:
    return random.randint(0, 2**32 - 1) if s is None or s < 0 else s


# control_type -> SetUnionControlNetType `type` string (xinsir Union model).
_UNION_TYPE = {
    ControlType.tile: "tile",
    ControlType.depth: "depth",
    ControlType.canny: "canny/lineart/anime_lineart/mlsd",
    ControlType.lineart: "canny/lineart/anime_lineart/mlsd",
    ControlType.scribble: "hed/pidi/scribble/ted",
}


def _preprocessor(ct: ControlType, image_src, res: int) -> dict:
    """ControlNet preprocessor node (comfyui_controlnet_aux) for a control type.
    All auto-download their weights on first use; tile/canny need no model."""
    if ct == ControlType.depth:
        return {"class_type": "DepthAnythingV2Preprocessor",
                "inputs": {"image": image_src, "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": res}}
    if ct == ControlType.canny:
        return {"class_type": "CannyEdgePreprocessor",
                "inputs": {"image": image_src, "low_threshold": 100, "high_threshold": 200, "resolution": res}}
    if ct == ControlType.lineart:
        return {"class_type": "LineArtPreprocessor",
                "inputs": {"image": image_src, "coarse": "disable", "resolution": res}}
    if ct == ControlType.scribble:
        return {"class_type": "ScribblePreprocessor",
                "inputs": {"image": image_src, "resolution": res}}
    # default: tile (structure + colour; the most preservation, least restyle)
    return {"class_type": "TilePreprocessor",
            "inputs": {"image": image_src, "pyrUp_iters": 3, "resolution": res}}


def _effective_diffusion(p: RestyleParams):
    """Resolve the divergence knobs. `variation` (when set) is a single 0..1 dial that maps to
    denoise + controlnet strength/end — high variation = a new texture, structure barely held, so
    the caption (material family) is what carries identity. Otherwise use the explicit params."""
    if p.variation is None:
        return p.denoise, p.controlnet_strength, p.controlnet_end_percent
    v = max(0.0, min(1.0, p.variation))
    denoise = 0.45 + v * 0.47          # 0.45 (subtle) .. 0.92 (mostly new)
    cn_strength = 0.50 - v * 0.38      # 0.50 (locked) .. 0.12 (faint echo)
    cn_end = 0.60 - v * 0.40           # 0.60 .. 0.20 (release control early)
    return denoise, cn_strength, cn_end


def build_restyle(image_name: str, p: RestyleParams, prefix: str, scaled_wh=None, ip_image_name=None):
    eff_denoise, eff_cn_strength, eff_cn_end = _effective_diffusion(p)
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

    res = p.input_size or 1024
    g["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": img_src, "vae": ["1", 2]}}

    # --- model chain: tiling-patched model, optionally IP-Adapter style-injected ---
    model_ref = ["2", 0]
    if p.ip_adapter:
        ip_img = img_src
        if ip_image_name:
            g["41"] = {"class_type": "LoadImage", "inputs": {"image": ip_image_name}}
            ip_img = ["41", 0]
        g["30"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": p.ip_adapter_model}}
        g["31"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": p.clip_vision_model}}
        g["32"] = {"class_type": "IPAdapterAdvanced",
                   "inputs": {"model": model_ref, "ipadapter": ["30", 0], "image": ip_img,
                              "weight": p.ip_adapter_weight, "weight_type": p.ip_adapter_weight_type,
                              "combine_embeds": "concat", "start_at": 0.0, "end_at": 1.0,
                              "embeds_scaling": "V only", "clip_vision": ["31", 0]}}
        model_ref = ["32", 0]

    # --- positive prompt: content-aware caption (+ style + quality) or literal ---
    if p.auto_caption:
        g["60"] = {"class_type": "DownloadAndLoadFlorence2Model",
                   "inputs": {"model": p.caption_model, "precision": "fp16"}}
        g["61"] = {"class_type": "Florence2Run",
                   "inputs": {"image": img_src, "florence2_model": ["60", 0],
                              "text_input": "", "task": p.caption_task, "fill_mask": True,
                              "keep_model_loaded": True, "num_beams": 3, "max_new_tokens": 256,
                              "do_sample": False, "seed": 1}}
        g["62"] = {"class_type": "TextureForgePromptCompose",
                   "inputs": {"caption": ["61", 2], "style": p.style,
                              "suffix": p.quality_suffix, "prefix": "", "separator": ", "}}
        g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ["62", 0]}}
    else:
        literal = ", ".join(x.strip() for x in (p.prompt, p.style, p.quality_suffix) if x and x.strip())
        g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": literal}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": p.negative_prompt}}

    pos, neg = ["6", 0], ["7", 0]
    if p.use_controlnet and eff_cn_strength > 0:
        g["8"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": p.controlnet}}
        cn_ref = ["8", 0]
        if p.controlnet_union:
            g["80"] = {"class_type": "SetUnionControlNetType",
                       "inputs": {"control_net": ["8", 0], "type": _UNION_TYPE.get(p.control_type, "auto")}}
            cn_ref = ["80", 0]
        g["9"] = _preprocessor(p.control_type, img_src, res)
        g["10"] = {"class_type": "ControlNetApplyAdvanced",
                   "inputs": {"positive": pos, "negative": neg, "control_net": cn_ref,
                              "image": ["9", 0], "strength": eff_cn_strength,
                              "start_percent": p.controlnet_start_percent,
                              "end_percent": eff_cn_end}}
        pos, neg = ["10", 0], ["10", 1]

    # --- sampler: img2img (add noise) or unsample (invert then resample) ---
    if p.method == RestyleMethod.unsample:
        g["11u"] = {"class_type": "BNK_Unsampler",
                    "inputs": {"model": model_ref, "steps": p.steps, "end_at_step": 0, "cfg": p.cfg,
                               "sampler_name": p.sampler_name, "scheduler": p.scheduler,
                               "normalize": "disable", "positive": pos, "negative": neg,
                               "latent_image": ["5", 0]}}
        g["11"] = {"class_type": "KSamplerAdvanced",
                   "inputs": {"model": model_ref, "add_noise": "disable", "noise_seed": _seed(p.seed),
                              "steps": p.steps, "cfg": p.cfg, "sampler_name": p.sampler_name,
                              "scheduler": p.scheduler, "positive": pos, "negative": neg,
                              "latent_image": ["11u", 0], "start_at_step": 0, "end_at_step": p.steps,
                              "return_with_leftover_noise": "disable"}}
    else:
        g["11"] = {"class_type": "KSampler",
                   "inputs": {"model": model_ref, "seed": _seed(p.seed), "steps": p.steps, "cfg": p.cfg,
                              "sampler_name": p.sampler_name, "scheduler": p.scheduler,
                              "positive": pos, "negative": neg, "latent_image": ["5", 0], "denoise": eff_denoise}}
    g["12"] = {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}

    final = ["12", 0]
    if p.upscale:
        g["13"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": p.upscale_model}}
        g["14"] = {"class_type": "TilingAwareUpscale",
                   "inputs": {"upscale_model": ["13", 0], "image": ["12", 0], "pad": p.upscale_pad}}
        final = ["14", 0]

    g["15"] = {"class_type": "SaveImage", "inputs": {"images": final, "filename_prefix": f"{prefix}/result"}}
    return g, {"15": "result"}


def build_restyle_flux(image_name: str, p: FluxKontextParams, prefix: str, scaled_wh=None):
    """FLUX.1 Kontext instruction edit. Not tileable (DiT) — for non-seamless restyles."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": p.unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": p.clip_name1, "clip_name2": p.clip_name2, "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae_name}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
    }
    img_src = ["4", 0]
    if scaled_wh is not None:
        w, h = scaled_wh
        g["40"] = {"class_type": "ImageScale",
                   "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                              "width": w, "height": h, "crop": "disabled"}}
        img_src = ["40", 0]

    g["5"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": img_src}}
    g["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}}
    g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.prompt}}
    g["8"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["7", 0], "guidance": p.guidance}}
    g["9"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["6", 0]}}
    g["10"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}}
    g["11"] = {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": p.width, "height": p.height, "batch_size": 1}}
    g["12"] = {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "seed": _seed(p.seed), "steps": p.steps, "cfg": 1.0,
                          "sampler_name": p.sampler_name, "scheduler": p.scheduler,
                          "positive": ["9", 0], "negative": ["10", 0], "latent_image": ["11", 0],
                          "denoise": 1.0}}
    g["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}}
    g["14"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"{prefix}/result"}}
    return g, {"14": "result"}


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

    if p.backend == PBRBackend.stablematerials:
        res = p.input_size or 512
        g["2"] = {"class_type": "MatForgerMaterialEstimation",
                  "inputs": {"repo_id": p.sm_repo, "prompt": p.material_prompt, "steps": p.steps,
                             "cfg": p.cfg, "resolution": res, "tileable": p.tileable, "image": img_src}}
        labels = {"basecolor": 0, "normal": 1, "height": 2, "roughness": 3, "metalness": 4}
        out_labels = {}
        nid = 3
        for label, slot in labels.items():
            g[str(nid)] = {"class_type": "SaveImage",
                           "inputs": {"images": ["2", slot], "filename_prefix": f"{prefix}/{label}"}}
            out_labels[str(nid)] = label
            nid += 1
        return g, out_labels

    # default: Ubisoft CHORD
    g["2"] = {"class_type": "ChordLoadModel", "inputs": {"ckpt_name": p.chord_model}}
    g["3"] = {"class_type": "ChordMaterialEstimation", "inputs": {"chord_model": ["2", 0], "image": img_src}}
    g["4"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": f"{prefix}/basecolor"}}
    g["5"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 1], "filename_prefix": f"{prefix}/normal"}}
    g["6"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 2], "filename_prefix": f"{prefix}/roughness"}}
    g["7"] = {"class_type": "SaveImage", "inputs": {"images": ["3", 3], "filename_prefix": f"{prefix}/metalness"}}
    return g, {"4": "basecolor", "5": "normal", "6": "roughness", "7": "metalness"}
