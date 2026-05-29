"""Texture Forge support nodes.

Two nodes the gateway pipeline relies on:

  * TextureForgePromptCompose - joins an (optional) auto-caption with a caller
    style modifier and a quality/PBR suffix into a single positive-prompt STRING
    that can be wired straight into CLIPTextEncode's `text` input. This is what
    makes the restyle prompt *content-aware per input* (caption) while still
    carrying the requested transformation (style) and the seamless/PBR tags.

  * MatForgerMaterialEstimation - single-image (or text) -> PBR maps using the
    gvecchio StableMaterials / MatForger diffusers pipelines, exposing the five
    maps (basecolor / normal / height / roughness / metalness) as DISCRETE IMAGE
    outputs - mirroring how Ubisoft CHORD's node presents its maps. This exists
    because the only third-party wrapper (smthemex/ComfyUI_PBR_Maker) packs all
    maps into two batched tensors and ships a broken save path; we want clean,
    individually-wireable outputs the gateway can save under stable labels.
"""
import torch


# --------------------------------------------------------------------------- #
# Prompt composition
# --------------------------------------------------------------------------- #
class TextureForgePromptCompose:
    """Join prefix + caption + style + suffix into one prompt string.

    Empty parts are skipped so a missing caption (auto_caption off upstream, or
    an empty Florence-2 result) collapses gracefully. `caption` is an input so it
    can be linked from a captioner; everything else is a widget with a default.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": ("STRING", {"default": "", "multiline": True}),
                "suffix": ("STRING", {"default": "", "multiline": True}),
                "prefix": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {
                "caption": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    CATEGORY = "texture_forge"

    def compose(self, style, suffix, prefix, separator, caption=""):
        parts = [p.strip() for p in (prefix, caption, style, suffix) if p and p.strip()]
        return (separator.join(parts),)


# --------------------------------------------------------------------------- #
# StableMaterials / MatForger PBR estimation
# --------------------------------------------------------------------------- #
_SM_CACHE: dict = {}  # repo_id -> pipeline (avoid reloading every call)

_MAP_NAMES = ("basecolor", "normal", "height", "roughness", "metalness")
# StableMaterials/MatForger expose the metalness map as `.metallic`.
_MAP_ATTRS = {
    "basecolor": ("basecolor", "base_color", "albedo"),
    "normal": ("normal", "normals"),
    "height": ("height", "displacement", "depth"),
    "roughness": ("roughness", "rough"),
    "metalness": ("metallic", "metalness", "metal"),
}


def _to_image_tensor(obj):
    """Coerce a PIL.Image / numpy array / tensor map into a ComfyUI IMAGE tensor
    of shape [1, H, W, 3], float32 in 0..1. Single-channel maps are broadcast to 3."""
    import numpy as np
    from PIL import Image

    if isinstance(obj, Image.Image):
        arr = np.array(obj).astype(np.float32) / 255.0
    elif isinstance(obj, np.ndarray):
        arr = obj.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    elif torch.is_tensor(obj):
        arr = obj.detach().cpu().float().numpy()
        if arr.max() > 1.5:
            arr = arr / 255.0
    else:
        raise TypeError(f"unsupported map type: {type(obj)}")

    if arr.ndim == 2:                       # H,W -> H,W,1
        arr = arr[..., None]
    if arr.shape[0] in (1, 3) and arr.ndim == 3 and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # C,H,W -> H,W,C
    if arr.shape[-1] == 1:                  # grey -> rgb
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:                  # rgba -> rgb
        arr = arr[..., :3]

    t = torch.from_numpy(np.ascontiguousarray(arr)).float().clamp(0.0, 1.0)
    return t.unsqueeze(0)                    # 1,H,W,3


class MatForgerMaterialEstimation:
    """Generate PBR maps from a text prompt and/or an input image.

    Loads a gvecchio material-diffusion pipeline (StableMaterials by default,
    MatForger also works) via diffusers `trust_remote_code`. When `image` is
    connected it conditions on the image (restyle-PBR); otherwise it uses the
    text prompt. Outputs the five maps as separate IMAGE tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "gvecchio/StableMaterials"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "tileable": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = _MAP_NAMES
    FUNCTION = "estimate"
    CATEGORY = "texture_forge"

    def _load(self, repo_id):
        if repo_id in _SM_CACHE:
            return _SM_CACHE[repo_id]
        import comfy.model_management as mm
        from diffusers import DiffusionPipeline

        device = mm.get_torch_device()
        try:
            use_fp16 = bool(mm.should_use_fp16(device))
        except Exception:
            use_fp16 = torch.cuda.is_available()
        dtype = torch.float16 if use_fp16 else torch.float32
        pipe = DiffusionPipeline.from_pretrained(
            repo_id, trust_remote_code=True, torch_dtype=dtype)
        try:
            pipe = pipe.to(device)
        except Exception:
            pass
        for opt in ("enable_vae_tiling", "enable_attention_slicing"):
            fn = getattr(pipe, opt, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        _SM_CACHE[repo_id] = pipe
        return pipe

    def estimate(self, repo_id, prompt, steps, cfg, resolution, tileable, image=None):
        from PIL import Image
        import numpy as np

        pipe = self._load(repo_id)

        kwargs = dict(num_inference_steps=int(steps), guidance_scale=float(cfg),
                      height=int(resolution), width=int(resolution))
        # Pass tileable only if the pipeline accepts it (StableMaterials/MatForger do).
        try:
            import inspect
            if "tileable" in inspect.signature(pipe.__call__).parameters:
                kwargs["tileable"] = bool(tileable)
        except (TypeError, ValueError):
            kwargs["tileable"] = bool(tileable)

        if image is not None:
            arr = (image[0].detach().cpu().numpy() * 255.0).round().astype("uint8")
            cond = Image.fromarray(arr)
            result = pipe(prompt=cond, **kwargs)
        else:
            result = pipe(prompt=prompt or "material texture", **kwargs)

        material = getattr(result, "images", None)
        material = material[0] if material else result
        if isinstance(material, (list, tuple)):
            material = material[0]

        outputs = []
        for name in _MAP_NAMES:
            obj = None
            for attr in _MAP_ATTRS[name]:
                obj = getattr(material, attr, None)
                if obj is not None:
                    break
            if obj is None:
                # Missing map (e.g. a pipeline that omits height): emit mid-grey
                # so downstream save/labels stay consistent rather than crashing.
                h = w = int(resolution)
                grey = np.full((h, w, 3), 0.5, dtype="float32")
                outputs.append(torch.from_numpy(grey).unsqueeze(0))
            else:
                outputs.append(_to_image_tensor(obj))
        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "TextureForgePromptCompose": TextureForgePromptCompose,
    "MatForgerMaterialEstimation": MatForgerMaterialEstimation,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextureForgePromptCompose": "Texture Forge Prompt Compose",
    "MatForgerMaterialEstimation": "StableMaterials/MatForger PBR",
}
