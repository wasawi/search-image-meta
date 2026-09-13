#!/usr/bin/env python3
"""Build synthetic ComfyUI images for testing --only-connected.

Each image carries `prompt` and `workflow` metadata written the way ComfyUI
writes it (json.dumps into PNG tEXt chunks; EXIF Model/Make for WebP), with
the `prompt` matching what the frontend exports: every active node, wired or
not, minus muted/bypassed/virtual nodes.

Every scenario plants marker strings in nodes that are wired in and in nodes
that aren't. samples/expected.json (and EXPECTED.md) record which markers each
search mode should find; check_only_connected.py verifies that.

Drag any PNG into ComfyUI to inspect its graph.
"""

import json
import textwrap
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

OUT = Path(__file__).with_name("samples")
MODES = ("all", "linked", "output")


# --------------------------------------------------------------------------
# editor-format workflow builder
# --------------------------------------------------------------------------

class Workflow:
    def __init__(self):
        self.nodes: dict = {}
        self.links: list = []

    def add(self, nid, type_, inputs=(), outputs=(), widgets=(), mode=0,
            title=None):
        row, col = divmod(len(self.nodes), 5)
        node = {
            "id": nid, "type": type_, "pos": [col * 360, row * 280],
            "size": [320, 140], "flags": {}, "order": len(self.nodes),
            "mode": mode,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outputs)],
            "properties": {"Node name for S&R": type_},
            "widgets_values": list(widgets),
        }
        if title:
            node["title"] = title
        self.nodes[nid] = node
        return nid

    def link(self, src, src_slot, dst, dst_slot):
        lid = len(self.links) + 1
        out = self.nodes[src]["outputs"][src_slot]
        out["links"].append(lid)
        self.nodes[dst]["inputs"][dst_slot]["link"] = lid
        self.links.append([lid, src, src_slot, dst, dst_slot, out["type"]])

    def json(self, **extra):
        return {"last_node_id": max(self.nodes), "last_link_id": len(self.links),
                "nodes": list(self.nodes.values()), "links": self.links,
                "groups": [], "config": {},
                "extra": {"ds": {"scale": 0.7, "offset": [60, 60]}},
                "version": 0.4, **extra}


def api(class_type, title=None, **inputs):
    """One node of the API-format `prompt`."""
    return {"inputs": inputs, "class_type": class_type,
            "_meta": {"title": title or class_type}}


# node shapes, as the stock ComfyUI nodes declare them
def ckpt(wf, nid, name, mode=0):
    wf.add(nid, "CheckpointLoaderSimple",
           outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
           widgets=[name], mode=mode)


def text(wf, nid, s, mode=0):
    wf.add(nid, "CLIPTextEncode", inputs=[("clip", "CLIP")],
           outputs=[("CONDITIONING", "CONDITIONING")], widgets=[s], mode=mode)


def load_image(wf, nid, name, mode=0):
    wf.add(nid, "LoadImage", outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
           widgets=[name, "image"], mode=mode)


def preview(wf, nid, title=None):
    wf.add(nid, "PreviewImage", inputs=[("images", "IMAGE")], title=title)


def t2i(wf, p, tag, positive=True):
    """ComfyUI's stock text-to-image graph (node ids 3-9, as in the template)."""
    main_ckpt = f"MAIN_ckpt_{tag}.safetensors"
    ckpt(wf, 4, main_ckpt)
    wf.add(5, "EmptyLatentImage", outputs=[("LATENT", "LATENT")],
           widgets=[512, 512, 1])
    if positive:
        text(wf, 6, f"MAIN_prompt_{tag} a lighthouse at dusk")
    text(wf, 7, "blurry, lowres")
    wf.add(3, "KSampler",
           inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                   ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
           outputs=[("LATENT", "LATENT")],
           widgets=[42, "fixed", 20, 8, "euler", "normal", 1])
    wf.add(8, "VAEDecode", inputs=[("samples", "LATENT"), ("vae", "VAE")],
           outputs=[("IMAGE", "IMAGE")])
    wf.add(9, "SaveImage", inputs=[("images", "IMAGE")], widgets=["ComfyUI"])

    p.update({
        "4": api("CheckpointLoaderSimple", ckpt_name=main_ckpt),
        "5": api("EmptyLatentImage", width=512, height=512, batch_size=1),
        "7": api("CLIPTextEncode", text="blurry, lowres", clip=["4", 1]),
        "3": api("KSampler", seed=42, steps=20, cfg=8, sampler_name="euler",
                 scheduler="normal", denoise=1, model=["4", 0],
                 positive=["6", 0], negative=["7", 0], latent_image=["5", 0]),
        "8": api("VAEDecode", samples=["3", 0], vae=["4", 2]),
        "9": api("SaveImage", filename_prefix="ComfyUI", images=["8", 0]),
    })
    if positive:
        p["6"] = api("CLIPTextEncode",
                     text=f"MAIN_prompt_{tag} a lighthouse at dusk",
                     clip=["4", 1])


def t2i_links(wf, model=(4, 0), clip=(4, 1)):
    wf.link(*model, 3, 0)
    wf.link(6, 0, 3, 1)
    wf.link(7, 0, 3, 2)
    wf.link(5, 0, 3, 3)
    wf.link(*clip, 6, 0)
    wf.link(*clip, 7, 0)
    wf.link(3, 0, 8, 0)
    wf.link(4, 2, 8, 1)
    wf.link(8, 0, 9, 0)


# --------------------------------------------------------------------------
# scenarios: (title, description, workflow, prompt, {marker: "all/linked/output"})
# --------------------------------------------------------------------------

def s_stray_loader(tag):
    wf, p = Workflow(), {}
    t2i(wf, p, tag)
    ckpt(wf, 10, "STRAY_ckpt_krea.safetensors")
    wf.add(11, "Note", widgets=["NOTE_text: remember to try krea"])
    t2i_links(wf)
    p["10"] = api("CheckpointLoaderSimple",
                  ckpt_name="STRAY_ckpt_krea.safetensors")
    return ("Stray loader + note",
            "Stock text-to-image graph plus an unwired Checkpoint loader and a "
            "Note. The frontend still exports the unwired loader into `prompt`.",
            wf.json(), p,
            {f"MAIN_ckpt_{tag}": "yyy", f"MAIN_prompt_{tag}": "yyy",
             "STRAY_ckpt_krea": "ynn", "NOTE_text": "ynn",
             # literal JSON fragment: filtered text keeps ComfyUI's formatting
             f'"ckpt_name": "MAIN_ckpt_{tag}': "yyy"})


def s_stray_chain():
    wf, p = Workflow(), {}
    t2i(wf, p, "s02")
    load_image(wf, 10, "CHAIN_src.png")
    wf.add(11, "ImageScale", inputs=[("image", "IMAGE")],
           outputs=[("IMAGE", "IMAGE")],
           widgets=["nearest-exact", 512, 512, "disabled"], title="CHAIN_scale")
    t2i_links(wf)
    wf.link(10, 0, 11, 0)
    p["10"] = api("LoadImage", image="CHAIN_src.png")
    p["11"] = api("ImageScale", title="CHAIN_scale", upscale_method="nearest-exact",
                  width=512, height=512, crop="disabled", image=["10", 0])
    return ("Wired leftover chain",
            "LoadImage -> ImageScale, wired to each other but feeding nothing. "
            "'linked' keeps them (they have a link); 'output' drops them.",
            wf.json(), p,
            {"MAIN_ckpt_s02": "yyy", "CHAIN_src": "yyn", "CHAIN_scale": "yyn"})


def s_bypass():
    wf, p = Workflow(), {}
    t2i(wf, p, "s03")
    wf.add(10, "LoraLoader", inputs=[("model", "MODEL"), ("clip", "CLIP")],
           outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP")],
           widgets=["BYPASSED_lora.safetensors", 1, 1], mode=4)
    load_image(wf, 12, "VIA_BYPASS_src.png")
    wf.add(13, "ImageInvert", inputs=[("image", "IMAGE")],
           outputs=[("IMAGE", "IMAGE")], mode=4)
    preview(wf, 14)
    t2i_links(wf, model=(10, 0), clip=(10, 1))
    wf.link(4, 0, 10, 0)
    wf.link(4, 1, 10, 1)
    wf.link(12, 0, 13, 0)
    wf.link(13, 0, 14, 0)
    # bypassed nodes vanish from `prompt`; their links pass straight through
    p["12"] = api("LoadImage", image="VIA_BYPASS_src.png")
    p["14"] = api("PreviewImage", images=["12", 0])
    return ("Bypassed nodes",
            "A bypassed LoRA between the checkpoint and the sampler, and "
            "LoadImage -> bypassed ImageInvert -> PreviewImage. The bypassed "
            "nodes aren't searched; LoadImage stays connected through them.",
            wf.json(), p,
            {"MAIN_ckpt_s03": "yyy", "BYPASSED_lora": "ynn",
             "VIA_BYPASS_src": "yyy"})


def s_muted():
    wf, p = Workflow(), {}
    t2i(wf, p, "s04")
    load_image(wf, 10, "MUTED_src.png", mode=2)
    preview(wf, 11, title="MUTED_preview")
    t2i_links(wf)
    wf.link(10, 0, 11, 0)
    # the frontend drops the muted node and the input that pointed at it
    p["11"] = api("PreviewImage", title="MUTED_preview")
    return ("Muted node",
            "A muted LoadImage wired into a PreviewImage. The muted node is "
            "ignored, which leaves the preview wired to nothing.",
            wf.json(), p,
            {"MAIN_ckpt_s04": "yyy", "MUTED_src": "ynn", "MUTED_preview": "ynn"})


def s_reroute():
    wf, p = Workflow(), {}
    t2i(wf, p, "s05")
    load_image(wf, 10, "REROUTE_dangling_src.png")
    wf.add(11, "Reroute", inputs=[("", "*")], outputs=[("", "IMAGE")])
    load_image(wf, 13, "REROUTE_through_src.png")
    wf.add(14, "Reroute", inputs=[("", "*")], outputs=[("", "IMAGE")])
    preview(wf, 15)
    t2i_links(wf)
    wf.link(10, 0, 11, 0)
    wf.link(13, 0, 14, 0)
    wf.link(14, 0, 15, 0)
    p["10"] = api("LoadImage", image="REROUTE_dangling_src.png")
    p["13"] = api("LoadImage", image="REROUTE_through_src.png")
    p["15"] = api("PreviewImage", images=["13", 0])
    return ("Reroutes",
            "LoadImage -> Reroute that leads nowhere (not connected), and "
            "LoadImage -> Reroute -> PreviewImage (connected).",
            wf.json(), p,
            {"MAIN_ckpt_s05": "yyy", "REROUTE_dangling_src": "ynn",
             "REROUTE_through_src": "yyy"})


def s_set_get():
    wf, p = Workflow(), {}
    t2i(wf, p, "s06")
    wf.add(10, "SetNode", inputs=[("MODEL", "MODEL")], outputs=[("*", "*")],
           widgets=["model_var"])
    wf.add(11, "GetNode", outputs=[("MODEL", "MODEL")], widgets=["model_var"])
    load_image(wf, 12, "SETGET_src.png")
    wf.add(13, "SetNode", inputs=[("IMAGE", "IMAGE")], outputs=[("*", "*")],
           widgets=["img_var"])
    wf.add(14, "GetNode", outputs=[("IMAGE", "IMAGE")], widgets=["img_var"])
    preview(wf, 15)
    load_image(wf, 16, "SETGET_orphan_src.png")
    wf.add(17, "SetNode", inputs=[("IMAGE", "IMAGE")], outputs=[("*", "*")],
           widgets=["orphan_var"])
    t2i_links(wf, model=(11, 0))
    wf.link(4, 0, 10, 0)
    wf.link(12, 0, 13, 0)
    wf.link(14, 0, 15, 0)
    wf.link(16, 0, 17, 0)
    p["12"] = api("LoadImage", image="SETGET_src.png")
    p["15"] = api("PreviewImage", images=["12", 0])
    p["16"] = api("LoadImage", image="SETGET_orphan_src.png")
    return ("KJNodes Set/Get",
            "Model and an image passed through Set/Get pairs (no visible "
            "link), plus a SetNode nobody reads. Needs KJNodes to render in "
            "ComfyUI.",
            wf.json(), p,
            {"MAIN_ckpt_s06": "yyy", "SETGET_src": "yyy",
             "SETGET_orphan_src": "ynn"})


def _subgraph(name, texts):
    """A subgraph definition of CLIPTextEncode nodes: [(id, text, wired)]."""
    inner = Workflow()
    links = []
    for nid, s, wired in texts:
        text(inner, nid, s)
        if wired:
            lin, lout = len(links) + 1, len(links) + 2
            inner.nodes[nid]["inputs"][0]["link"] = lin
            inner.nodes[nid]["outputs"][0]["links"] = [lout]
            links += [
                {"id": lin, "origin_id": -10, "origin_slot": 0,
                 "target_id": nid, "target_slot": 0, "type": "CLIP"},
                {"id": lout, "origin_id": nid, "origin_slot": 0,
                 "target_id": -20, "target_slot": 0, "type": "CONDITIONING"},
            ]
    uid = lambda s: str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}/{s}"))
    return {
        "id": uid("def"), "version": 1,
        "state": {"lastGroupId": 0, "lastNodeId": max(inner.nodes),
                  "lastLinkId": len(links), "lastRerouteId": 0},
        "revision": 0, "config": {}, "name": name,
        "inputNode": {"id": -10, "bounding": [-320, 0, 120, 60]},
        "outputNode": {"id": -20, "bounding": [1000, 0, 120, 60]},
        "inputs": [{"id": uid("in"), "name": "clip", "type": "CLIP",
                    "linkIds": [l["id"] for l in links if l["origin_id"] == -10],
                    "localized_name": "clip", "pos": [-220, 20]}],
        "outputs": [{"id": uid("out"), "name": "CONDITIONING",
                     "type": "CONDITIONING",
                     "linkIds": [l["id"] for l in links if l["target_id"] == -20],
                     "localized_name": "CONDITIONING", "pos": [1020, 20]}],
        "widgets": [], "nodes": list(inner.nodes.values()), "groups": [],
        "links": links, "extra": {},
    }


def s_subgraph():
    wf, p = Workflow(), {}
    used = _subgraph("PromptBox", [(1, "SUBGRAPH_used_text, a lighthouse", True),
                                   (2, "SUBGRAPH_stray_text", False)])
    orphan = _subgraph("OrphanBox", [(1, "SUBGRAPH_orphan_instance_text", True)])
    unused = _subgraph("UnusedBox", [(1, "SUBGRAPH_unused_def_text", True)])
    t2i(wf, p, "s07", positive=False)
    sg_io = dict(inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")])
    # promoted widget: the instance keeps a copy of the inner node's text
    wf.add(6, used["id"], widgets=["SUBGRAPH_used_text, a lighthouse"], **sg_io)
    wf.add(10, orphan["id"], **sg_io)
    t2i_links(wf)
    # the frontend flattens subgraphs: inner nodes get "instance:inner" ids
    p["6:1"] = api("CLIPTextEncode", text="SUBGRAPH_used_text, a lighthouse",
                   clip=["4", 1])
    p["6:2"] = api("CLIPTextEncode", text="SUBGRAPH_stray_text")
    p["10:1"] = api("CLIPTextEncode", text="SUBGRAPH_orphan_instance_text")
    p["3"]["inputs"]["positive"] = ["6:1", 0]
    return ("Subgraphs",
            "Positive prompt lives in a subgraph that also holds an unwired "
            "text node; a second subgraph instance is wired to nothing; a "
            "third definition is never placed. Best effort: needs a frontend "
            "with subgraph support to open.",
            wf.json(definitions={"subgraphs": [used, orphan, unused]}), p,
            {"MAIN_ckpt_s07": "yyy", "SUBGRAPH_used_text": "yyy",
             "SUBGRAPH_stray_text": "ynn",
             "SUBGRAPH_orphan_instance_text": "ynn",
             "SUBGRAPH_unused_def_text": "ynn"})


def s_no_output():
    wf, p = Workflow(), {}
    load_image(wf, 1, "NOOUT_src.png")
    wf.add(2, "ImageInvert", inputs=[("image", "IMAGE")],
           outputs=[("IMAGE", "IMAGE")], title="NOOUT_invert")
    load_image(wf, 3, "NOOUT_stray.png")
    wf.link(1, 0, 2, 0)
    p["1"] = api("LoadImage", image="NOOUT_src.png")
    p["2"] = api("ImageInvert", title="NOOUT_invert", image=["1", 0])
    p["3"] = api("LoadImage", image="NOOUT_stray.png")
    return ("No recognisable output node",
            "Nothing that looks like Save/Preview. 'output' mode falls back to "
            "'linked' instead of discarding the whole graph.",
            wf.json(), p,
            {"NOOUT_src": "yyy", "NOOUT_invert": "yyy", "NOOUT_stray": "ynn"})


A1111 = ("a photo of a cat\nNegative prompt: blurry\n"
         "Steps: 20, Sampler: Euler a, Model: A1111_model_marker")


def s_node_scope():
    wf, p = Workflow(), {}
    t2i(wf, p, "s11")
    text(wf, 10, "PAIR_left and PAIR_right in one unwired node")
    load_image(wf, 11, "SPLIT_left.png")
    preview(wf, 12, title="SPLIT_right")
    t2i_links(wf)
    wf.link(11, 0, 12, 0)
    p["10"] = api("CLIPTextEncode",
                  text="PAIR_left and PAIR_right in one unwired node")
    p["11"] = api("LoadImage", image="SPLIT_left.png")
    p["12"] = api("PreviewImage", title="SPLIT_right", images=["11", 0])
    return ("--scope node",
            "SPLIT_left and SPLIT_right sit in two wired nodes; PAIR_left and "
            "PAIR_right share one unwired node. The multi-term cases are in "
            "EXPECTED.md.",
            wf.json(), p,
            {"MAIN_ckpt_s11": "yyy", "PAIR_left": "ynn", "SPLIT_left": "yyy",
             "SPLIT_right": "yyy"})


def s_a1111():
    return ("A1111 parameters",
            "No ComfyUI graph at all: --only-connected must leave other "
            "metadata alone.",
            None, None, {"A1111_model_marker": "yyy"})


SCENARIOS = [
    ("01_stray_loader.png", lambda: s_stray_loader("s01")),
    ("02_stray_chain.png", s_stray_chain),
    ("03_bypassed.png", s_bypass),
    ("04_muted.png", s_muted),
    ("05_reroute.png", s_reroute),
    ("06_set_get.png", s_set_get),
    ("07_subgraph.png", s_subgraph),
    ("08_no_output_node.png", s_no_output),
    ("09_stray_loader.webp", lambda: s_stray_loader("s09")),
    ("10_a1111.png", s_a1111),
    ("11_node_scope.png", s_node_scope),
]

# multi-term searches over the whole samples folder, mostly for --scope node:
# (patterns, extra args, files that should match, {file: matching fields})
MULTI = [
    # the same node vs different nodes of the same field
    (["MAIN_prompt_s11", "lighthouse"], ["--scope", "node"],
     ["11_node_scope.png"], None),
    (["MAIN_prompt_s11", "lighthouse"], ["--scope", "node", "--only-connected"],
     ["11_node_scope.png"], None),
    (["MAIN_prompt_s11", "MAIN_ckpt_s11"], [], ["11_node_scope.png"], None),
    (["MAIN_prompt_s11", "MAIN_ckpt_s11"], ["--scope", "field"],
     ["11_node_scope.png"], None),
    (["MAIN_prompt_s11", "MAIN_ckpt_s11"], ["--scope", "node"], [], None),
    (["SPLIT_left", "SPLIT_right"], ["--scope", "field"],
     ["11_node_scope.png"], None),
    (["SPLIT_left", "SPLIT_right"], ["--scope", "node"], [], None),
    (["SPLIT_left", "SPLIT_right"], ["--scope", "node", "--any"],
     ["11_node_scope.png"], None),
    # both terms in one node that isn't wired to anything
    (["PAIR_left", "PAIR_right"], ["--scope", "node"],
     ["11_node_scope.png"], None),
    (["PAIR_left", "PAIR_right"], ["--scope", "node", "--only-connected"],
     [], None),
    (["STRAY_ckpt_krea", "CheckpointLoaderSimple"], ["--scope", "node"],
     ["01_stray_loader.png", "09_stray_loader.webp"], None),
    (["STRAY_ckpt_krea", "CheckpointLoaderSimple"],
     ["--scope", "node", "--connected-mode", "output"], [], None),
    # subgraph insides are nodes too, and -v/--json name the node
    (["SUBGRAPH_used_text", "lighthouse"], ["--scope", "node"],
     ["07_subgraph.png"],
     {"07_subgraph.png": ["prompt#6:1", "workflow#PromptBox/1"]}),
    (["SUBGRAPH_used_text", "MAIN_ckpt_s07"], ["--scope", "node"], [], None),
    (["SUBGRAPH_used_text", "MAIN_ckpt_s07"], [], ["07_subgraph.png"], None),
    # metadata without nodes: each field is one unit
    (["cat", "A1111_model_marker"], ["--scope", "node"],
     ["10_a1111.png"], {"10_a1111.png": ["parameters"]}),
]


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def card(title, desc, expect):
    img = Image.new("RGB", (900, 150 + 26 * len(expect)), (30, 32, 38))
    d = ImageDraw.Draw(img)
    d.text((20, 14), title, font=_font(28), fill=(240, 240, 240))
    y = 54
    for line in textwrap.wrap(desc, 88):
        d.text((20, y), line, font=_font(16), fill=(175, 178, 190))
        y += 20
    y += 12
    small = _font(17)
    for x, head in ((20, "marker"), (560, "all"), (650, "linked"), (760, "output")):
        d.text((x, y), head, font=small, fill=(140, 190, 255))
    for marker, flags in expect.items():
        y += 26
        d.text((20, y), marker, font=small, fill=(220, 220, 220))
        for x, flag in zip((560, 650, 760), flags):
            d.text((x, y), "hit" if flag == "y" else "-", font=small,
                   fill=(120, 220, 140) if flag == "y" else (200, 110, 110))
    return img


def main():
    OUT.mkdir(exist_ok=True)
    expected, doc = {}, ["# --only-connected samples\n",
                         "`hit` = the marker should be found. Modes: `all` = no "
                         "flag, `linked` = `--only-connected`, `output` = "
                         "`--connected-mode output`.\n"]
    for name, build in SCENARIOS:
        title, desc, workflow, prompt, expect = build()
        img = card(title, desc, expect)
        path = OUT / name
        if name.endswith(".webp"):
            exif = img.getexif()
            exif[0x0110] = "prompt:" + json.dumps(prompt)       # EXIF Model
            exif[0x010F] = "workflow:" + json.dumps(workflow)   # EXIF Make
            img.save(path, exif=exif, lossless=True)
        else:
            info = PngInfo()
            if prompt is None:
                info.add_text("parameters", A1111)
            else:
                info.add_text("prompt", json.dumps(prompt))
                info.add_text("workflow", json.dumps(workflow))
            img.save(path, pnginfo=info)
        expected[name] = expect

        doc.append(f"\n## {name} — {title}\n\n{desc}\n\n"
                   "| marker | all | linked | output |\n|---|---|---|---|")
        for marker, flags in expect.items():
            cells = " | ".join("hit" if f == "y" else "—" for f in flags)
            doc.append(f"| `{marker}` | {cells} |")
        print(f"wrote {path}")

    multi = [{"patterns": pats, "args": args, "files": files, "fields": fields}
             for pats, args, files, fields in MULTI]
    doc.append("\n## Multi-term searches (whole folder)\n\n"
               "| patterns | options | should match |\n|---|---|---|")
    for case in multi:
        pats = " ".join(f'`{p}`' for p in case["patterns"])
        opts = f'`{" ".join(case["args"])}`' if case["args"] else "(default)"
        files = ", ".join(case["files"]) or "nothing"
        if case["fields"]:
            files += " — fields: " + "; ".join(
                ", ".join(f"`{f}`" for f in fl) for fl in case["fields"].values())
        doc.append(f"| {pats} | {opts} | {files} |")

    (OUT / "expected.json").write_text(json.dumps(expected, indent=2))
    (OUT / "expected_multi.json").write_text(json.dumps(multi, indent=2))
    (OUT / "EXPECTED.md").write_text("\n".join(doc) + "\n")


if __name__ == "__main__":
    main()
