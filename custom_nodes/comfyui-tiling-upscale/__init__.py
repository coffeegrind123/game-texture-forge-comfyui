"""Tiling-aware upscale node.

Stock `ImageUpscaleWithModel` (ESRGAN etc.) uses non-wrapping conv padding at the
image border, which breaks seamless tiling at the upscaled resolution. This node
circular-pads the image before the upscale so the border convolutions see the
correct wrapped neighbours, then crops the padding (scaled) back off. The retained
region tiles cleanly.

It reuses ComfyUI's stock model-upscale routine (OOM-safe tiled inference) so
behaviour matches the normal node apart from the seam handling.
"""
import torch

try:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
except Exception:  # pragma: no cover - import path guard
    ImageUpscaleWithModel = None


class TilingAwareUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "pad": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image, pad):
        if ImageUpscaleWithModel is None:
            raise RuntimeError("comfy_extras.nodes_upscale_model.ImageUpscaleWithModel not found")

        # image: [B, H, W, C], values 0..1
        b, h, w, c = image.shape
        p = min(pad, max(0, min(h, w) - 1))  # circular pad must be < dim size

        if p > 0:
            x = image.permute(0, 3, 1, 2)                       # B,C,H,W
            x = torch.nn.functional.pad(x, (p, p, p, p), mode="circular")
            padded = x.permute(0, 2, 3, 1).contiguous()
        else:
            padded = image

        up = ImageUpscaleWithModel().upscale(upscale_model, padded)[0]  # B,H',W',C

        if p > 0:
            scale = up.shape[1] / padded.shape[1]
            m = int(round(p * scale))
            if m > 0:
                up = up[:, m:up.shape[1] - m, m:up.shape[2] - m, :].contiguous()
        return (up,)


NODE_CLASS_MAPPINGS = {"TilingAwareUpscale": TilingAwareUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"TilingAwareUpscale": "Tiling-Aware Upscale (Model)"}
