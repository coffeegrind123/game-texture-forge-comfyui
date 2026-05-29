#!/usr/bin/env python3
"""Generate the texture img2img + tiling + PBR workflow for ComfyUI.

Defines the graph once, then emits BOTH:
  - workflows/texture_img2img_tiling.json      (UI / litegraph - drag into ComfyUI)
  - workflows/texture_img2img_tiling_api.json  (API /prompt format)

Slot indices and widget ordering are derived from the live /object_info schema,
and every connection is type-checked, so the output imports/runs without guesswork.

Usage (object_info fetched from the running container's loopback):
    docker compose exec -T comfyui python /workspace/gen_workflow.py
or pass a path to a saved object_info.json as argv[1].
"""
import json, sys, urllib.request, os

# --------------------------------------------------------------------------
# Graph definition: each node is (key, class_type, widgets{}, inputs{name:(src_key, out_name)})
# --------------------------------------------------------------------------
NODES = [
    ("ckpt",  "CheckpointLoaderSimple", {"ckpt_name": "sd_xl_base_1.0.safetensors"}, {}),

    # Tiling: patch model + VAE to circular padding (applies on img2img too).
    ("tile",  "SeamlessTile",   {"tiling": "enable", "copy_model": "Make a copy"},
        {"model": ("ckpt", "MODEL")}),
    ("cvae",  "MakeCircularVAE", {"tiling": "enable", "copy_vae": "Make a copy"},
        {"vae": ("ckpt", "VAE")}),

    # Source texture -> latent.
    ("load",  "LoadImage", {"image": "test_texture.png"}, {}),
    ("enc",   "VAEEncode", {}, {"pixels": ("load", "IMAGE"), "vae": ("ckpt", "VAE")}),

    # Prompts.
    ("pos",   "CLIPTextEncode",
        {"text": "photorealistic surface, sharp fine detail, natural lighting, physically based, high detail, 4k"},
        {"clip": ("ckpt", "CLIP")}),
    ("neg",   "CLIPTextEncode",
        {"text": "blurry, cartoon, flat, painted, low detail, visible seam, watermark, text"},
        {"clip": ("ckpt", "CLIP")}),

    # Tile ControlNet locks structure so denoise adds realism without inventing geometry.
    ("cnl",   "ControlNetLoader", {"control_net_name": "controlnet-tile-sdxl-1.0.safetensors"}, {}),
    ("tilep", "TilePreprocessor", {"pyrUp_iters": 3, "resolution": 1024},
        {"image": ("load", "IMAGE")}),
    ("cna",   "ControlNetApplyAdvanced",
        {"strength": 0.6, "start_percent": 0.0, "end_percent": 1.0},
        {"positive": ("pos", "CONDITIONING"), "negative": ("neg", "CONDITIONING"),
         "control_net": ("cnl", "CONTROL_NET"), "image": ("tilep", "IMAGE")}),

    # img2img sample on the circular-padded model.
    ("ks",    "KSampler",
        {"seed": 0, "steps": 28, "cfg": 6.5, "sampler_name": "dpmpp_2m",
         "scheduler": "karras", "denoise": 0.45},
        {"model": ("tile", "MODEL"), "positive": ("cna", "positive"),
         "negative": ("cna", "negative"), "latent_image": ("enc", "LATENT")}),

    # Decode with the circular VAE so the tile survives decode.
    ("dec",   "VAEDecode", {}, {"samples": ("ks", "LATENT"), "vae": ("cvae", "VAE")}),

    # Detail upscale (stays tileable - operates on an already-circular image).
    ("up",    "UpscaleModelLoader", {"model_name": "4x-UltraSharp.pth"}, {}),
    ("upimg", "TilingAwareUpscale", {"pad": 64},
        {"upscale_model": ("up", "UPSCALE_MODEL"), "image": ("dec", "IMAGE")}),

    # Outputs: final tile + half-offset seam check.
    ("save",  "SaveImage", {"filename_prefix": "texture/realistic_tiled"},
        {"images": ("upimg", "IMAGE")}),
    ("off",   "OffsetImage", {"x_percent": 50.0, "y_percent": 50.0},
        {"pixels": ("upimg", "IMAGE")}),
    ("savechk", "SaveImage", {"filename_prefix": "texture/seam_check"},
        {"images": ("off", "IMAGE")}),

    # PBR branch (Ubisoft CHORD): single image -> full material set.
    ("chordm", "ChordLoadModel", {"ckpt_name": "chord_v1.safetensors"}, {}),
    ("chord",  "ChordMaterialEstimation", {},
        {"chord_model": ("chordm", "chord_model"), "image": ("upimg", "IMAGE")}),
    ("sbc", "SaveImage", {"filename_prefix": "texture/pbr_basecolor"}, {"images": ("chord", "basecolor")}),
    ("sn",  "SaveImage", {"filename_prefix": "texture/pbr_normal"},    {"images": ("chord", "normal")}),
    ("sr",  "SaveImage", {"filename_prefix": "texture/pbr_roughness"}, {"images": ("chord", "roughness")}),
    ("sm",  "SaveImage", {"filename_prefix": "texture/pbr_metalness"}, {"images": ("chord", "metalness")}),
]

WIDGET_SCALAR = {"STRING", "INT", "FLOAT", "BOOLEAN"}


def load_object_info():
    if len(sys.argv) > 1:
        return json.load(open(sys.argv[1]))
    return json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info"))


def is_socket(t):
    # COMBO inputs are lists; scalar widget types are strings in WIDGET_SCALAR.
    if isinstance(t, list):
        return False
    return t not in WIDGET_SCALAR


def ordered_inputs(oi, cls):
    d = oi[cls]["input"]
    items = list(d.get("required", {}).items()) + list(d.get("optional", {}).items())
    return items  # [(name, spec), ...] in declaration order


def out_index(oi, cls, out_name):
    names = oi[cls]["output_name"]
    return names.index(out_name)


def main():
    oi = load_object_info()
    key2id = {k: i + 1 for i, (k, *_2) in enumerate(NODES)}

    # ---- validate every node + connection against the live schema ----------
    errors = []
    for key, cls, widgets, inputs in NODES:
        if cls not in oi:
            errors.append(f"{key}: class {cls} not registered"); continue
        in_specs = dict(ordered_inputs(oi, cls))
        for in_name, (src_key, out_name) in inputs.items():
            if in_name not in in_specs:
                errors.append(f"{key}: no input '{in_name}' on {cls}"); continue
            dst_type = in_specs[in_name][0]
            src_cls = dict((k, c) for k, c, *_ in NODES)[src_key]
            if out_name not in oi[src_cls]["output_name"]:
                errors.append(f"{key}.{in_name}: src {src_key} has no output '{out_name}'"); continue
            src_type = oi[src_cls]["output"][out_index(oi, src_cls, out_name)]
            if isinstance(dst_type, list):
                errors.append(f"{key}.{in_name}: expected widget not socket"); continue
            if src_type != dst_type:
                errors.append(f"{key}.{in_name}: type {src_type} != {dst_type}")
    if errors:
        print("VALIDATION FAILED:"); [print("  -", e) for e in errors]; sys.exit(1)
    print(f"Structural validation OK: {len(NODES)} nodes, all links type-matched.")

    # ---- build API format --------------------------------------------------
    api = {}
    for key, cls, widgets, inputs in NODES:
        node_inputs = {}
        for in_name, spec in ordered_inputs(oi, cls):
            if in_name in inputs:
                src_key, out_name = inputs[in_name]
                node_inputs[in_name] = [str(key2id[src_key]), out_index(oi, dict((k,c) for k,c,*_ in NODES)[src_key], out_name)]
            elif in_name in widgets:
                node_inputs[in_name] = widgets[in_name]
        api[str(key2id[key])] = {"class_type": cls, "inputs": node_inputs}

    # ---- build UI / litegraph format --------------------------------------
    links = []
    link_id = 0
    ui_nodes = []
    # simple auto-layout: columns by dependency depth
    col_x = {}
    for order, (key, cls, widgets, inputs) in enumerate(NODES):
        in_specs = ordered_inputs(oi, cls)
        socket_inputs = [(n, s[0]) for n, s in in_specs if is_socket(s[0])]
        widget_inputs = [(n, s) for n, s in in_specs if not is_socket(s[0])]

        node = {
            "id": key2id[key], "type": cls,
            "pos": [200 + (order % 6) * 360, 80 + (order // 6) * 320],
            "size": [300, 200], "flags": {}, "order": order, "mode": 0,
            "inputs": [], "outputs": [], "properties": {"Node name for S&R": cls},
            "widgets_values": [],
        }
        # input sockets
        for slot, (in_name, in_type) in enumerate(socket_inputs):
            node["inputs"].append({"name": in_name, "type": in_type, "link": None})
        # outputs
        for oslot, (oname, otype) in enumerate(zip(oi[cls]["output_name"], oi[cls]["output"])):
            node["outputs"].append({"name": oname, "type": otype, "links": [], "slot_index": oslot})
        # widget values (in declared widget order); add control_after_generate for seeds
        wv = []
        for in_name, spec in widget_inputs:
            t = spec[0]
            default = spec[1].get("default") if len(spec) > 1 and isinstance(spec[1], dict) else None
            val = widgets.get(in_name, default if default is not None else (t[0] if isinstance(t, list) and t else ""))
            wv.append(val)
            if in_name in ("seed", "noise_seed"):
                wv.append("fixed")  # control_after_generate companion widget
        node["widgets_values"] = wv
        ui_nodes.append(node)

    id2node = {n["id"]: n for n in ui_nodes}
    for key, cls, widgets, inputs in NODES:
        dst = id2node[key2id[key]]
        in_specs = ordered_inputs(oi, cls)
        socket_names = [n for n, s in in_specs if is_socket(s[0])]
        for in_name, (src_key, out_name) in inputs.items():
            dst_slot = socket_names.index(in_name)
            src_node = id2node[key2id[src_key]]
            src_cls = dict((k, c) for k, c, *_ in NODES)[src_key]
            src_slot = out_index(oi, src_cls, out_name)
            link_id += 1
            ltype = src_node["outputs"][src_slot]["type"]
            links.append([link_id, src_node["id"], src_slot, dst["id"], dst_slot, ltype])
            dst["inputs"][dst_slot]["link"] = link_id
            src_node["outputs"][src_slot]["links"].append(link_id)

    ui = {
        "last_node_id": max(key2id.values()),
        "last_link_id": link_id,
        "nodes": ui_nodes,
        "links": links,
        "groups": [
            {"title": "Tiling (circular model + VAE)", "bounding": [180, 380, 740, 260], "color": "#3f789e"},
            {"title": "img2img realism (ControlNet-Tile)", "bounding": [930, 60, 740, 580], "color": "#a1309b"},
            {"title": "PBR (Ubisoft CHORD)", "bounding": [180, 700, 1100, 260], "color": "#b58b2a"},
        ],
        "config": {}, "extra": {}, "version": 0.4,
    }

    os.makedirs("workflows", exist_ok=True)
    json.dump(ui, open("workflows/texture_img2img_tiling.json", "w"), indent=2)
    json.dump(api, open("workflows/texture_img2img_tiling_api.json", "w"), indent=2)
    print("Wrote workflows/texture_img2img_tiling.json (UI) and _api.json (API).")


if __name__ == "__main__":
    main()
