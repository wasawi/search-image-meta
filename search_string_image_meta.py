#!/usr/bin/env python3
"""
search_string_image_meta.py — search image metadata for a string, in parallel.

Searches EXIF (incl. nested Exif/GPS IFDs), XMP, IPTC/APP13, JPEG comments
and PNG tEXt/iTXt chunks (ComfyUI `prompt` / `workflow` / `parameters`).

Prints the full path of every matching image, and can optionally place a
link/alias to each match in a separate folder.

Examples
--------
    # this folder only
    python3 search_string_image_meta.py ~/Pictures "sunset"

    # recurse into subfolders
    python3 search_string_image_meta.py ~/Pictures "sunset" -r

    # BOTH terms must be present (in any metadata field)
    python3 search_string_image_meta.py ~/output "Barcelona" "sunset" -r

    # either one is enough
    python3 search_string_image_meta.py ~/output "Barcelona" "Girona" -r --any

    # Barcelona, but not the images that also mention sunset or night
    python3 search_string_image_meta.py ~/output "Barcelona" -r --not "sunset" "night"

    # regex + symlinks to the matches, 24 workers
    python3 search_string_image_meta.py ~/output "wan.?2\\.2" -r --regex --link-dir ~/matches -j 24

    # real Finder aliases instead of symlinks (macOS only)
    python3 search_string_image_meta.py ~/output "Krea" -r --link-dir ~/matches --link-type alias

    # skip ComfyUI nodes that aren't wired into the graph (stray loaders, notes…)
    python3 search_string_image_meta.py ~/output "Krea" -r --only-connected

    # stricter: also skip wired-up leftovers that never reach a Save/Preview node
    python3 search_string_image_meta.py ~/output "Krea" -r --connected-mode output

    # both terms inside the SAME metadata field, not spread across fields
    python3 search_string_image_meta.py ~/output "Barcelona" "sunset" -r --scope field

    # both terms inside the SAME ComfyUI node (e.g. one prompt box), wired nodes only
    python3 search_string_image_meta.py ~/output "lighthouse" "dusk" -r --scope node --only-connected

    # only the ComfyUI graph chunks, only PNGs, exact case
    python3 search_string_image_meta.py ~/output "LoRA" -r -s --fields prompt workflow --ext png

    # show which field matched, with a snippet of the surrounding text
    python3 search_string_image_meta.py ~/output "Krea" -r -v

    # machine-readable results (paths, fields, snippets, stats)
    python3 search_string_image_meta.py ~/output "Krea" -r --json > krea_hits.json

    # resumable log: rerun the same command to skip folders already done
    python3 search_string_image_meta.py /Volumes/Photos "Krea" -r --results ~/krea_results.txt
    python3 search_string_image_meta.py /Volumes/Photos "Krea" -r --results ~/krea_results.txt --no-resume

    # real copies named by their relative path (a/b/img.png -> a__b__img.png)
    python3 search_string_image_meta.py ~/output "Krea" -r --link-dir ~/matches --link-type copy --flatten path

    # thorough pass: late PNG chunks and deep XMP, and list unreadable files
    python3 search_string_image_meta.py ~/Pictures "Barcelona" -r --deep --errors

    # big local PSD/TIFF files: process pool, include dot-folders and symlinked dirs
    python3 search_string_image_meta.py ~/Design "Photoshop" -r --pool process --ext psd tif --hidden --follow-symlinks

    # in scripts: no progress bar, no pre-count; exit status 0 = found, 1 = none
    python3 search_string_image_meta.py ~/output "Krea" -r --progress never --no-count > /dev/null && echo "found"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import warnings
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                ThreadPoolExecutor, wait)
from pathlib import Path

try:
    from PIL import Image, ExifTags, ImageFile, UnidentifiedImageError
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")

Image.MAX_IMAGE_PIXELS = None  # don't warn/refuse on huge generated images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # --deep calls load(); half-written
                                        # renders should still give up their text

# Pillow needs defusedxml to *parse* XMP into a dict. Without it we still
# search the raw XMP packet, so nothing is missed — just silence the warning.
try:
    import defusedxml  # noqa: F401
    HAVE_DEFUSEDXML = True
except ImportError:
    HAVE_DEFUSEDXML = False
    warnings.filterwarnings("ignore", message=".*defusedxml.*")


def raise_fd_limit(desired: int) -> int:
    """Lift RLIMIT_NOFILE as far as the OS allows. Returns the soft limit.

    macOS ships with a soft limit of 256, which a high -j will blow through:
    every open() then fails with EMFILE and the image looks 'unreadable'.
    """
    try:
        import resource
    except ImportError:  # non-POSIX
        return 1 << 30
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= desired:
        return soft
    for target in (desired, 24576, 10240, 4096, 1024):
        if target <= soft:
            break
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (target, max(target, hard)))
            return target
        except (ValueError, OSError):
            continue
    return soft

DEFAULT_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".gif", ".tif", ".tiff", ".webp",
    ".bmp", ".heic", ".heif", ".avif", ".jp2", ".dng", ".cr2", ".nef",
    ".arw", ".orf", ".rw2", ".raf", ".psd", ".ico",
}

TAGS = ExifTags.TAGS
GPSTAGS = ExifTags.GPSTAGS
SNIPPET_WIDTH = 140


# --------------------------------------------------------------------------
# metadata extraction
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse the binary padding that surrounds text in raw segments."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text)


def _decode_bytes(raw: bytes) -> str:
    """Decode a raw metadata value, including UTF-16 (Windows UserComment).

    EXIF UserComment carries an 8-byte charset marker; the rest of the time a
    high NUL density is the giveaway. Decoding UTF-16 as UTF-8 "works" — NUL is
    valid UTF-8 — and silently interleaves NULs through the text, so a search
    for "Barcelona" would never match it.
    """
    raw = bytes(raw)
    if raw[:8] == b"ASCII\x00\x00\x00":
        return raw[8:].decode("ascii", "ignore")
    if raw[:8] in (b"UNICODE\x00", b"UNICODE\x00\x00"[:8]):
        raw = raw[8:]
    if raw and raw.count(0) * 3 > len(raw):          # looks like UTF-16
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return raw.decode("utf-16", "ignore")
        enc = "utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be"
        return raw.decode(enc, "ignore")
    return raw.decode("utf-8", errors="ignore") or raw.decode("latin-1", "ignore")


def _latin1_as_utf8(text: str) -> str:
    """Undo Pillow reading UTF-8 bytes in a PNG tEXt chunk as Latin-1.

    tEXt is Latin-1 by the spec, but plenty of tools write UTF-8 into it.
    Genuine Latin-1 ("café") is almost never valid UTF-8 once re-encoded, and
    real iTXt text doesn't fit in Latin-1 at all, so both pass through.
    """
    if text.isascii():
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def _stringify(value) -> str:
    """Turn any metadata value into something searchable."""
    if isinstance(value, (bytes, bytearray)):
        return _decode_bytes(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _exif_fields(img) -> dict:
    out = {}
    try:
        exif = img.getexif()
    except Exception:
        return out
    if not exif:
        return out

    for tag_id, value in exif.items():
        out[TAGS.get(tag_id, f"Exif:{hex(tag_id)}")] = _stringify(value)

    # nested IFDs hold most of the interesting stuff (UserComment, lens, GPS…)
    for ifd_id, prefix, table in (
        (0x8769, "Exif", TAGS),
        (0x8825, "GPS", GPSTAGS),
        (0xA005, "Interop", TAGS),
    ):
        try:
            ifd = exif.get_ifd(ifd_id)
        except Exception:
            continue
        for tag_id, value in (ifd or {}).items():
            out[f"{prefix}:{table.get(tag_id, hex(tag_id))}"] = _stringify(value)
    return out


def _xmp_from_bytes(path: Path, limit: int = 4_000_000) -> str:
    """Fallback: scrape the raw XMP packet out of the file itself."""
    try:
        with open(path, "rb") as fh:
            blob = fh.read(limit)
    except OSError:
        return ""
    start = blob.find(b"<x:xmpmeta")
    if start == -1:
        start = blob.find(b"<?xpacket")
    if start == -1:
        return ""
    end = blob.find(b"</x:xmpmeta>", start)
    end = end + 12 if end != -1 else min(len(blob), start + 200_000)
    return blob[start:end].decode("utf-8", errors="ignore")


def extract_metadata(path: Path, deep: bool = False) -> dict:
    """Return {field_name: text} for everything we can read out of the image.

    Fast by default: only metadata that sits in the file header is touched, so
    the pixel data is never decoded. `deep` trades ~20x speed for two rare
    extras — PNG text chunks written *after* the image data, and XMP packets
    hiding further into the file than the header.
    """
    fields: dict[str, str] = {}
    try:
        with Image.open(path) as img:
            fmt = img.format

            if deep and fmt == "PNG":
                img.load()  # also pulls in chunks stored after IDAT

            info = img.info or {}
            for key, value in info.items():
                if key in ("icc_profile", "exif", "photoshop", "adobe"):
                    continue  # binary; parsed properly below
                text = _stringify(value)
                fields[key] = _latin1_as_utf8(text) if fmt == "PNG" else text

            # PNG.getexif() decodes the ENTIRE image looking for a trailing
            # eXIf chunk — 50x the cost of everything else here. Only pay it
            # when the header already told us EXIF is present.
            has_exif = "exif" in info or any("exif" in k.lower() for k in info)
            if fmt != "PNG" or has_exif:
                fields.update(_exif_fields(img))

            if any("xmp" in k.lower() for k in info) or fmt in ("JPEG", "TIFF", "MPO"):
                try:
                    xmp = img.getxmp()
                    if xmp:
                        fields["xmp_parsed"] = _stringify(xmp)
                except Exception:
                    pass

            # JPEG APPn segments: IPTC/Photoshop (APP13), XMP (APP1), …
            for marker, data in (getattr(img, "applist", None) or []):
                if marker in ("APP0", "APP2"):
                    continue  # JFIF / ICC, no text
                text = _clean(data.decode("utf-8", errors="ignore"))
                if text.strip():
                    fields[f"jpeg:{marker}"] = text
    except (UnidentifiedImageError, OSError, ValueError):
        raw = _xmp_from_bytes(path, 4_000_000 if deep else 131_072)
        if raw:
            return {"xmp_raw": raw}
        raise

    # PNG keeps its XMP in an iTXt chunk, which is already in `info`; for other
    # formats a header-sized scrape catches packets Pillow didn't surface.
    if fmt != "PNG" and not any("xmp" in k.lower() for k in fields):
        raw = _xmp_from_bytes(path, 4_000_000 if deep else 131_072)
        if raw:
            fields["xmp_raw"] = raw

    return fields


# --------------------------------------------------------------------------
# ComfyUI graphs (--only-connected, --scope node)
# --------------------------------------------------------------------------
#
# ComfyUI saves two copies of the graph, and both keep nodes wired to nothing:
#   prompt    API format, {id: {class_type, inputs}}. A link is an input whose
#             value is ["source id", output slot]. Every active node is
#             exported, wired or not; muted and bypassed ones are left out.
#   workflow  editor format, {nodes, links}: also notes, reroutes, muted and
#             bypassed nodes, and subgraph definitions.
# WebP/AVIF keep them in EXIF instead, as "prompt:{…}" and "workflow:{…}".

_MODE_MUTED, _MODE_BYPASS = 2, 4       # LiteGraph node modes NEVER / BYPASS
_SUBGRAPH_IO = ("-10", "-20")          # a subgraph's own input / output node
_GRAPH_HEAD = re.compile(r"\s*(?:[A-Za-z_]\w*:)?\s*(?=\{)")
_OUTPUT_TYPE = re.compile(r"save|preview|show|display|video_?combine|compar"
                          r"|websocket", re.IGNORECASE)
_JSON = json.JSONDecoder()


def _is_output_type(node_type) -> bool:
    return isinstance(node_type, str) and bool(_OUTPUT_TYPE.search(node_type))


def _is_relay_type(node_type) -> bool:
    """Plumbing that only forwards a link: Reroute, KJNodes Set/Get."""
    return isinstance(node_type, str) and (
        "reroute" in node_type.lower() or node_type in ("SetNode", "GetNode"))


def _wired(ids, edges, via=(), outputs=None) -> set:
    """Return the subset of `ids` that counts as connected.

    edges    (source, target) id pairs
    via      pass-through nodes (bypassed, relays): their links are joined up
             end to end, and they never count as connected themselves
    outputs  None keeps every node that has a link. Otherwise only nodes that
             are, or feed into, one of these are kept — falling back to None
             when none of them is wired, so an output node we don't recognise
             can't wipe out the whole graph.
    """
    ups: dict = {}
    downs: dict = {}
    for src, dst in edges:
        if src != dst and src in ids and dst in ids:
            downs.setdefault(src, set()).add(dst)
            ups.setdefault(dst, set()).add(src)

    for v in via:
        srcs, dsts = ups.pop(v, set()), downs.pop(v, set())
        for s in srcs:
            downs[s].discard(v)
        for d in dsts:
            ups[d].discard(v)
        for s in srcs:
            for d in dsts:
                if s != d:
                    downs[s].add(d)
                    ups[d].add(s)

    if outputs:
        keep: set = set()
        todo = [n for n in outputs if ups.get(n)]
        while todo:
            n = todo.pop()
            if n not in keep:
                keep.add(n)
                todo.extend(ups.get(n, ()))
        if keep:
            return keep
    return {n for n in ids if ups.get(n) or downs.get(n)}


def _connected_prompt(prompt: dict, mode: str) -> dict:
    edges = []
    for nid, node in prompt.items():
        inputs = node.get("inputs")
        for value in (inputs.values() if isinstance(inputs, dict) else ()):
            # the frontend wraps list-valued widgets in {"__value__": …}, so a
            # bare two-item list is always a link
            if isinstance(value, list) and len(value) == 2:
                edges.append((str(value[0]), nid))
    outputs = ({nid for nid, node in prompt.items()
                if _is_output_type(node.get("class_type"))}
               if mode == "output" else None)
    keep = _wired(set(prompt), edges, outputs=outputs)
    return {nid: node for nid, node in prompt.items() if nid in keep}


def _link_ends(link):
    """(id, origin, target) of a workflow link, in either serialisation."""
    if isinstance(link, dict):
        return link.get("id"), link.get("origin_id"), link.get("target_id")
    if isinstance(link, list) and len(link) >= 4:
        return link[0], link[1], link[3]
    return None, None, None


def _set_get_name(node):
    values = node.get("widgets_values")
    if isinstance(values, list) and values and isinstance(values[0], str):
        return values[0]
    return None


def _connected_nodes(nodes, links, mode, output_subgraphs, io=()):
    """Filter one level of a workflow (the root, or a subgraph definition).

    Returns (kept nodes, links between kept nodes).
    """
    nodes = [n for n in nodes or () if isinstance(n, dict) and "id" in n]
    live = {str(n["id"]): n for n in nodes if n.get("mode") != _MODE_MUTED}

    # the links list can carry stale entries; trust the ones nodes point at
    referenced = set()
    for n in nodes:
        for slot in n.get("inputs") or ():
            if isinstance(slot, dict) and slot.get("link") is not None:
                referenced.add(str(slot["link"]))
        for slot in n.get("outputs") or ():
            if isinstance(slot, dict):
                referenced.update(str(i) for i in slot.get("links") or ())

    edges, wiring = [], []
    for link in links or ():
        lid, src, dst = _link_ends(link)
        if src is None or dst is None:
            continue
        if referenced and str(lid) not in referenced:
            continue
        edges.append((str(src), str(dst)))
        wiring.append((link, str(src), str(dst)))

    # KJNodes Set/Get hand a value over by name, with no link on the canvas
    setters: dict = {}
    for nid, n in live.items():
        if n.get("type") == "SetNode" and _set_get_name(n) is not None:
            setters.setdefault(_set_get_name(n), []).append(nid)
    for nid, n in live.items():
        if n.get("type") == "GetNode":
            edges.extend((s, nid) for s in setters.get(_set_get_name(n), ()))

    via = [nid for nid, n in live.items()
           if n.get("mode") == _MODE_BYPASS or _is_relay_type(n.get("type"))]

    outputs = None
    if mode == "output":
        outputs = {nid for nid, n in live.items()
                   if _is_output_type(n.get("type"))
                   or str(n.get("type")) in output_subgraphs}
        if io:
            outputs.add(_SUBGRAPH_IO[1])

    keep = _wired(set(live) | set(io), edges, via, outputs)
    return ([n for n in nodes if str(n["id"]) in keep],
            [link for link, s, d in wiring if s in keep and d in keep])


def _subgraphs_with_outputs(defs: dict) -> set:
    """Ids of subgraph definitions holding an output node, at any depth."""
    found: set = set()
    grew = True
    while grew:
        grew = False
        for sid, sg in defs.items():
            if sid not in found and any(
                    isinstance(n, dict) and (_is_output_type(n.get("type"))
                                             or str(n.get("type")) in found)
                    for n in sg.get("nodes") or ()):
                found.add(sid)
                grew = True
    return found


def _connected_workflow(workflow: dict, mode: str) -> dict:
    definitions = workflow.get("definitions")
    subgraph_list = (definitions.get("subgraphs")
                     if isinstance(definitions, dict) else None)
    defs = {str(sg["id"]): sg for sg in subgraph_list or ()
            if isinstance(sg, dict) and "id" in sg}
    with_outputs = _subgraphs_with_outputs(defs) if mode == "output" else set()

    nodes, links = _connected_nodes(workflow.get("nodes"),
                                    workflow.get("links"), mode, with_outputs)
    graph: dict = {"nodes": nodes, "links": links}

    # a subgraph definition is searched only when a kept node instantiates it;
    # its insides are filtered by their own wiring (to its input/output nodes)
    used: set = set()
    todo = [str(n.get("type")) for n in nodes]
    subgraphs = []
    while todo:
        sid = todo.pop()
        if sid in used or sid not in defs:
            continue
        used.add(sid)
        sg = defs[sid]
        sg_nodes, sg_links = _connected_nodes(sg.get("nodes"), sg.get("links"),
                                              mode, with_outputs,
                                              io=_SUBGRAPH_IO)
        subgraphs.append({"id": sg["id"], "name": sg.get("name"),
                          "nodes": sg_nodes, "links": sg_links})
        todo.extend(str(n.get("type")) for n in sg_nodes)
    if subgraphs:
        graph["definitions"] = {"subgraphs": subgraphs}
    return graph


def _graph_kind(data):
    """"workflow", "prompt", or None if `data` isn't a ComfyUI graph."""
    if isinstance(data, dict):
        if isinstance(data.get("nodes"), list):
            return "workflow"
        if data and all(isinstance(v, dict) and "class_type" in v
                        for v in data.values()):
            return "prompt"
    return None


def _graph_nodes(graph: dict, kind: str) -> list:
    """(label, node) for every node, subgraph insides included.

    Prompt ids already say where a node lives ("30:15" is node 15 inside
    subgraph instance 30); workflow subgraph nodes are labelled "name/id".
    """
    if kind == "prompt":
        return list(graph.items())
    definitions = graph.get("definitions")
    subgraphs = [sg for sg in ((definitions.get("subgraphs")
                                if isinstance(definitions, dict) else None) or ())
                 if isinstance(sg, dict)]
    # a subgraph instance keeps copies of its inner nodes' promoted widgets,
    # which would pair up values from different nodes; the inner nodes are
    # units of their own (and `prompt` holds their executed values)
    instance_types = {str(sg.get("id")) for sg in subgraphs}

    def units(nodes, prefix=""):
        return [(f"{prefix}{n.get('id')}", n) for n in nodes or ()
                if isinstance(n, dict) and str(n.get("type")) not in instance_types]

    found = units(graph.get("nodes"))
    for sg in subgraphs:
        found += units(sg.get("nodes"), f"{sg.get('name') or sg.get('id')}/")
    return found


def search_units(field: str, text: str, rxs, mode: str, scope: str,
                 connected, not_rxs=()) -> list:
    """Split one metadata field into the (label, text) units to match against.

    Normally that's the field itself. When it holds a ComfyUI graph:
      connected     drops the unconnected nodes first ("linked" / "output")
      scope "node"  makes each node its own unit, labelled "field#node"
    What remains is re-serialised the way ComfyUI writes it (json.dumps, with
    the real characters prepare_text() restored), so patterns written against
    the raw metadata keep matching. A filtered
    workflow keeps only the kept nodes, their links and the subgraphs they
    use (not groups or canvas settings); with scope "node" only the nodes'
    own contents are searched.
    """
    if not connected and scope != "node":
        return [(field, text)]
    head = _GRAPH_HEAD.match(text)
    if not head or ('"class_type"' not in text and '"nodes"' not in text):
        return [(field, text)]

    # dropping or splitting up nodes can only lose matches: if the graph as a
    # whole can't match, there is nothing to find (or to exclude with --not)
    # and no JSON to parse
    found = [rx.search(text) for rx in rxs]
    need_all = mode == "all" and scope != "image"
    if (not (all(found) if need_all else any(found))
            and not any(n.search(text) for n in not_rxs)):
        return []

    try:
        data, _end = _JSON.raw_decode(text, head.end())
    except ValueError:
        return [(field, text)]
    kind = _graph_kind(data)
    if kind is None:
        return [(field, text)]
    if connected:
        data = (_connected_workflow(data, connected) if kind == "workflow"
                else _connected_prompt(data, connected))

    if scope == "node":
        return [(f"{field}#{label}", json.dumps(node, ensure_ascii=False))
                for label, node in _graph_nodes(data, kind)]
    return [(field, head.group(0) + json.dumps(data, ensure_ascii=False))]


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------
#
# ComfyUI (like anything using Python's json.dumps) writes every non-ASCII
# character as a backslash-u escape, so 女孩 is stored as two hex codes and
# é as one. Searching that text as-is would never find Chinese, Japanese,
# accents or emoji, so JSON-looking fields get their escapes decoded first.

_BACKSLASH_U = "\\" + "u"
_JSON_START = re.compile(r"\s*(?:[A-Za-z_]\w*:)?\s*[\[{]")
_JSON_ESCAPE = re.compile(
    r"\\\\"                                                   # "\\" itself
    r"|[\\]u([dD][89abAB][0-9a-fA-F]{2})[\\]u([dD][c-fC-F][0-9a-fA-F]{2})"
    r"|[\\]u([0-9a-fA-F]{4})")


def _unescape(m: re.Match) -> str:
    if m.group(1):  # UTF-16 surrogate pair, e.g. an emoji
        hi, lo = int(m.group(1), 16), int(m.group(2), 16)
        return chr(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00))
    if m.group(3):
        code = int(m.group(3), 16)
        # ASCII escapes (quotes, control characters) stay put so the JSON
        # still parses; a lone surrogate isn't a character at all
        if code >= 0x80 and not 0xD800 <= code <= 0xDFFF:
            return chr(code)
    return m.group(0)  # an escaped backslash is text, never an escape


def prepare_text(text: str) -> str:
    """Metadata text the way a person would type it into a search.

    Decodes JSON unicode escapes and composes accents (NFC), so "café" typed
    in a terminal matches however the file stored it.
    """
    if _BACKSLASH_U in text and _JSON_START.match(text):
        text = _JSON_ESCAPE.sub(_unescape, text)
    if not text.isascii():
        text = unicodedata.normalize("NFC", text)
    return text


# --------------------------------------------------------------------------
# parallel scanning
# --------------------------------------------------------------------------

_CFG: dict = {}  # per-worker state, set by _worker_init


def _worker_init(opts: dict):
    """Set up a worker from the options main() collected.

    patterns, exclude, regex, case_sensitive, fields, snippets, deep, mode
    ("all" / "any"), scope ("image" / "field" / "node"), connected (None /
    "linked" / "output").
    """
    flags = (0 if opts["case_sensitive"] else re.IGNORECASE) | re.DOTALL

    def compile_all(patterns):
        patterns = (unicodedata.normalize("NFC", p) for p in patterns or ())
        return [re.compile(p if opts["regex"] else re.escape(p), flags)
                for p in patterns]

    _CFG.clear()
    _CFG.update(opts)
    _CFG["rx"] = compile_all(opts["patterns"])
    _CFG["not_rx"] = compile_all(opts.get("exclude"))
    _CFG["fields"] = ({f.lower() for f in opts["fields"]}
                      if opts["fields"] else None)


def _snippet(text: str, match: re.Match) -> str:
    start = max(0, match.start() - SNIPPET_WIDTH // 3)
    end = min(len(text), match.end() + SNIPPET_WIDTH // 2)
    out = _clean(text[start:end]).replace("\n", " ")
    return ("…" if start else "") + out + ("…" if end < len(text) else "")


def scan_one(path_str: str):
    """Worker: returns (path, [(field, snippet)], error or None)."""
    path = Path(path_str)
    try:
        meta = extract_metadata(path, deep=_CFG.get("deep", False))
    except Exception as exc:
        kind = type(exc).__name__
        errno = getattr(exc, "errno", None)
        if errno in (23, 24):  # ENFILE / EMFILE — a limit, not a bad file
            kind = "TooManyOpenFiles"
        return path_str, [], f"{kind}: {exc}"

    rxs, only = _CFG["rx"], _CFG["fields"]
    units = []
    for f, t in meta.items():
        if t and not (only and f.lower() not in only):
            units.extend(search_units(f, prepare_text(t), rxs, _CFG["mode"],
                                      _CFG["scope"], _CFG["connected"],
                                      _CFG["not_rx"]))
    hits = match_units(units, rxs, _CFG["mode"], _CFG["scope"],
                       _CFG["snippets"])
    if hits and any(n.search(text) for _, text in units for n in _CFG["not_rx"]):
        hits = []  # --not: an excluded term turned up
    return path_str, hits, None


def match_units(units, rxs, mode: str, scope: str, want_snippets: bool) -> list:
    """[(field, snippet)] when the (label, text) units match, else []."""
    if scope in ("field", "node"):
        # every term has to turn up inside one and the same field (or node)
        hits = []
        for field, text in units:
            found = [rx.search(text) for rx in rxs]
            if all(found) if mode == "all" else any(found):
                m = next(x for x in found if x)
                hits.append((field, _snippet(text, m) if want_snippets else ""))
        return hits

    # scope "image": the terms may be spread across different fields
    matched: dict = {}
    for field, text in units:
        for i, rx in enumerate(rxs):
            if i not in matched:
                m = rx.search(text)
                if m:
                    matched[i] = (field,
                                  _snippet(text, m) if want_snippets else "")
        if len(matched) == len(rxs) or (mode == "any" and matched):
            break  # the remaining fields can't change the outcome
    ok = len(matched) == len(rxs) if mode == "all" else bool(matched)
    return [matched[i] for i in sorted(matched)] if ok else []


def iter_work(root: Path, recursive: bool, exts, follow_symlinks: bool,
              include_hidden: bool, skip_dirs=frozenset()):
    """Lazily yield ("file", dirpath, fullpath) then ("seal", dirpath, None).

    A "seal" event means the walker has emitted every candidate file in that
    directory, so once its outstanding jobs finish the directory is complete
    and can be recorded as such.
    """
    if recursive:
        walker = os.walk(root, followlinks=follow_symlinks)
    else:
        walker = [(str(root), [],
                   [e.name for e in os.scandir(root) if e.is_file()])]

    for dirpath, dirnames, filenames in walker:
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if dirpath in skip_dirs:
            continue  # finished on an earlier run
        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            suffix = os.path.splitext(filename)[1].lower()
            if exts is not None:
                if suffix not in exts:
                    continue
            elif suffix not in DEFAULT_EXTS:
                continue
            yield "file", dirpath, os.path.join(dirpath, filename)
        yield "seal", dirpath, None


# --------------------------------------------------------------------------
# progress bar (stderr, no dependencies)
# --------------------------------------------------------------------------

class Progress:
    """Single-line progress bar on stderr. Determinate if `total` is known."""

    MIN_INTERVAL = 0.1  # seconds between redraws

    def __init__(self, total: int | None, enabled: bool):
        self.total = total
        self.enabled = enabled
        self.done = 0
        self.hits = 0
        self.start = time.monotonic()
        self._last_draw = 0.0
        self._drawn = False

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = int(max(0, seconds))
        h, rem = divmod(seconds, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    def _line(self) -> str:
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        tail = f"{self.hits} hit{'' if self.hits == 1 else 's'} · {rate:,.0f}/s"

        if self.total:
            frac = self.done / self.total
            eta = (self.total - self.done) / rate if rate > 0 else 0
            tail = (f"{self.done:,}/{self.total:,} · {tail} · "
                    f"ETA {self._fmt_time(eta)}")
            width = shutil.get_terminal_size((80, 24)).columns
            bar_w = max(10, min(40, width - len(tail) - 12))
            filled = int(bar_w * frac)
            bar = "█" * filled + "░" * (bar_w - filled)
            return f"{bar} {frac * 100:3.0f}% {tail}"

        spin = "|/-\\"[int(elapsed * 8) % 4]
        return f"{spin} {self.done:,} scanned · {tail} · {self._fmt_time(elapsed)}"

    def _write(self, text: str) -> None:
        sys.stderr.write(text)
        sys.stderr.flush()

    # -- public ------------------------------------------------------------

    def clear(self) -> None:
        """Erase the bar so something else can write to the terminal."""
        if self.enabled and self._drawn:
            self._write("\r\x1b[2K")
            self._drawn = False

    def draw(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_draw < self.MIN_INTERVAL:
            return
        self._last_draw = now
        self._write("\r\x1b[2K" + self._line())
        self._drawn = True

    def advance(self, hit: bool = False) -> None:
        self.done += 1
        if hit:
            self.hits += 1
        self.draw()

    def finish(self) -> None:
        if self.enabled and self._drawn:
            self.draw(force=True)
            self._write("\n")
            self._drawn = False


def count_images(paths_factory, enabled: bool) -> int:
    """Fast stat-only pre-pass so the bar can show a percentage and ETA."""
    total = 0
    last = 0.0
    for kind, _dirpath, _path in paths_factory():
        if kind != "file":
            continue
        total += 1
        if enabled and total % 512 == 0:
            now = time.monotonic()
            if now - last > 0.1:
                last = now
                sys.stderr.write(f"\r\x1b[2Kindexing… {total:,} images")
                sys.stderr.flush()
    if enabled:
        sys.stderr.write("\r\x1b[2K")
        sys.stderr.flush()
    return total


# --------------------------------------------------------------------------
# timestamp preservation (macOS creation date)
# --------------------------------------------------------------------------

# macOS keeps a real creation time (st_birthtime) that os.utime cannot touch.
# setattrlist(2) can, via ATTR_CMN_CRTIME.
_ATTR_BIT_MAP_COUNT = 5
_ATTR_CMN_CRTIME = 0x00000200
_FSOPT_NOFOLLOW = 0x00000001

_libc = None
if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util

        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

        class _Attrlist(ctypes.Structure):
            _fields_ = [("bitmapcount", ctypes.c_ushort),
                        ("reserved", ctypes.c_ushort),
                        ("commonattr", ctypes.c_uint),
                        ("volattr", ctypes.c_uint),
                        ("dirattr", ctypes.c_uint),
                        ("fileattr", ctypes.c_uint),
                        ("forkattr", ctypes.c_uint)]

        class _Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_int64),
                        ("tv_nsec", ctypes.c_int64)]

        _libc.setattrlist.argtypes = [ctypes.c_char_p,
                                      ctypes.POINTER(_Attrlist),
                                      ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_ulong]
        _libc.setattrlist.restype = ctypes.c_int
    except Exception:
        _libc = None


def set_creation_time(path: Path, birthtime_ns: int, follow: bool = True) -> bool:
    """Set a file's macOS creation date. Returns False where unsupported."""
    if _libc is None:
        return False
    import ctypes
    attrs = _Attrlist(bitmapcount=_ATTR_BIT_MAP_COUNT, reserved=0,
                      commonattr=_ATTR_CMN_CRTIME, volattr=0,
                      dirattr=0, fileattr=0, forkattr=0)
    ts = _Timespec(tv_sec=birthtime_ns // 1_000_000_000,
                   tv_nsec=birthtime_ns % 1_000_000_000)
    rc = _libc.setattrlist(os.fsencode(str(path)), ctypes.byref(attrs),
                           ctypes.byref(ts), ctypes.sizeof(ts),
                           0 if follow else _FSOPT_NOFOLLOW)
    return rc == 0


def copy_timestamps(src: Path, dest: Path, mode: str) -> bool:
    """Give `dest` the same creation and modification dates as `src`.

    Hardlinks share an inode, so they already have them. For a symlink the
    link's own timestamps are set, not the target's.
    """
    if mode == "hardlink":
        return True
    try:
        st = os.stat(src)  # the original, following any link
    except OSError:
        return False

    follow = mode != "symlink"
    ok = True

    try:
        os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns), follow_symlinks=follow)
    except (OSError, NotImplementedError):
        ok = False

    birth_ns = getattr(st, "st_birthtime_ns", None)
    if birth_ns is None:
        birth = getattr(st, "st_birthtime", None)
        birth_ns = int(birth * 1_000_000_000) if birth else st.st_mtime_ns
    if not set_creation_time(dest, birth_ns, follow=follow):
        ok = False
    return ok


# --------------------------------------------------------------------------
# results file (parameters, completed dirs, hits, stats — and resume)
# --------------------------------------------------------------------------

class ParamMismatch(Exception):
    pass


class ResultsFile:
    """Append-only log that doubles as a resume checkpoint.

    Layout (comments start with '#', data lines are prefixed and parsed):
        PARAMS  {json}          the search that produced this file
        DIR     /path           every candidate file in that folder was scanned
        HIT     /path           a matching image (with the fields that matched)
        ERR     /path           could not be read; its folder is left
                                unrecorded so a rerun retries it
    """

    def __init__(self, path: Path, params: dict, allow_resume: bool):
        self.path = path
        self.params = params
        self.completed_dirs: set[str] = set()
        self.prior_hits = 0
        self.prior_errors = 0
        self.prior_runs = 0
        self.resumed = False

        existing = path.exists() and path.stat().st_size > 0
        if existing and allow_resume:
            self._load()
            self.resumed = True
            self._fh = open(path, "a", encoding="utf-8")
            self._comment(f"resumed {_now()} — skipping "
                          f"{len(self.completed_dirs)} completed folder(s)")
        else:
            self._fh = open(path, "w", encoding="utf-8")
            self._comment(f"imgmetasearch results — started {_now()}")
            for key in ("root", "patterns", "match", "scope", "regex",
                        "case_sensitive", "recursive", "ext", "fields",
                        "hidden", "follow_symlinks", "only_connected",
                        "exclude"):
                if key in params:
                    self._comment(f"{key}: {params[key]}")
            self._write(f"PARAMS\t{json.dumps(params, sort_keys=True)}")
        self._fh.flush()

    # -- reading an earlier run --------------------------------------------

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PARAMS\t"):
                    old = json.loads(line[7:])
                    diff = {k: (old.get(k), self.params.get(k))
                            for k in dict.fromkeys([*self.params, *old])
                            if old.get(k) != self.params.get(k)}
                    if diff:
                        raise ParamMismatch(
                            "existing results file used different settings: "
                            + ", ".join(f"{k}={was!r} -> {now!r}"
                                        for k, (was, now) in diff.items())
                        )
                    self.prior_runs += 1
                elif line.startswith("DIR\t"):
                    self.completed_dirs.add(line[4:].rstrip("\n"))
                elif line.startswith("HIT\t"):
                    self.prior_hits += 1
                elif line.startswith("ERR\t"):
                    self.prior_errors += 1

    # -- writing ------------------------------------------------------------

    def _write(self, text: str) -> None:
        self._fh.write(text + "\n")

    def _comment(self, text: str) -> None:
        self._write(f"# {text}")

    def record_hit(self, path: str, fields, link=None) -> None:
        extra = ",".join(f for f, _ in fields)
        self._write(f"HIT\t{path}\t{extra}" + (f"\t-> {link}" if link else ""))
        self._fh.flush()  # survive a kill mid-run

    def record_error(self, path: str, reason: str) -> None:
        self._write(f"ERR\t{path}\t{reason}")

    def record_dir(self, dirpath: str) -> None:
        self.completed_dirs.add(dirpath)
        self._write(f"DIR\t{dirpath}")
        self._fh.flush()

    def close(self, scanned: int, found: int, errors: int,
              elapsed: float, completed: bool) -> None:
        self._comment("")
        self._comment(f"run {'finished' if completed else 'INTERRUPTED'} {_now()}")
        self._comment(f"scanned this run: {scanned:,}")
        self._comment(f"matches this run: {found:,}")
        self._comment(f"unreadable this run: {errors:,}")
        self._comment(f"elapsed: {elapsed:.1f}s"
                      + (f" ({scanned / elapsed:,.0f} img/s)" if elapsed > 0 else ""))
        self._comment(f"folders completed (cumulative): {len(self.completed_dirs):,}")
        self._comment(f"matches in this file (cumulative): "
                      f"{self.prior_hits + found:,}")
        if completed:
            self._write("COMPLETE")
        self._fh.flush()
        self._fh.close()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# link / alias creation
# --------------------------------------------------------------------------

def _unique_target(dest_dir: Path, name: str, src: Path | None = None) -> Path:
    candidate = dest_dir / name
    if not candidate.exists() and not candidate.is_symlink():
        return candidate
    # a link we already made on a previous run, pointing at the same file
    if src is not None and candidate.is_symlink():
        try:
            if Path(os.readlink(candidate)) == src:
                return candidate
        except OSError:
            pass
    stem, suffix = os.path.splitext(name)
    n = 1
    while True:
        candidate = dest_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        n += 1


def _make_finder_alias(src: Path, dest_dir: Path, name: str) -> Path:
    """Create a real macOS Finder alias via AppleScript."""
    target = _unique_target(dest_dir, name)
    script = (
        'tell application "Finder"\n'
        f'  set theFile to POSIX file "{src}" as alias\n'
        f'  set theDir to POSIX file "{dest_dir}" as alias\n'
        '  set newAlias to make new alias file at theDir to theFile\n'
        f'  set name of newAlias to "{target.name}"\n'
        'end tell'
    )
    subprocess.run(["osascript", "-e", script],
                   check=True, capture_output=True, text=True)
    return target


def make_link(src: Path, dest_dir: Path, mode: str, flatten: str, root: Path,
              preserve_dates: bool = True) -> Path:
    if flatten == "path":
        try:
            name = "__".join(src.relative_to(root).parts)
        except ValueError:
            name = src.name
    else:
        name = src.name

    if mode == "alias":
        target = _make_finder_alias(src, dest_dir, name)
        if preserve_dates:
            copy_timestamps(src, target, mode)
        return target

    target = _unique_target(dest_dir, name, src)
    if mode == "symlink":
        if target.is_symlink():
            target.unlink()
        target.symlink_to(src)
    elif mode == "hardlink":
        if not target.exists():
            os.link(src, target)
    elif mode == "copy":
        import shutil
        shutil.copy2(src, target)
    else:
        raise ValueError(f"unknown link type: {mode}")

    if preserve_dates:
        copy_timestamps(src, target, mode)
    return target


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Search image metadata for a string, in parallel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("folder", type=Path, help="folder to search")
    ap.add_argument("patterns", nargs="+", metavar="PATTERN",
                    help="string(s) to look for (regex with --regex). Give "
                         "several and all of them must match, unless --any")
    ap.add_argument("--any", action="store_const", const="any", default="all",
                    dest="match", help="match images containing ANY of the "
                                       "patterns (default: all of them)")
    ap.add_argument("--scope", default="image",
                    choices=["image", "field", "node"],
                    help="with several patterns: 'image' (default) lets them "
                         "match in different metadata fields, 'field' requires "
                         "them all in the same one, 'node' all in the same "
                         "ComfyUI node (metadata without nodes counts one "
                         "field at a time, as with 'field')")
    ap.add_argument("--not", nargs="+", metavar="PATTERN", dest="exclude",
                    help="skip images where any of these turns up. Checks "
                         "the same text the search does, so --fields, "
                         "--regex, -s and --only-connected apply, but the "
                         "whole image (whatever --scope says). Put it after "
                         "the search patterns")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="descend into subfolders (default: top level only)")
    ap.add_argument("-j", "--workers", type=int, default=0, metavar="N",
                    help="parallel workers (default: auto, 0 = auto, 1 = serial)")
    ap.add_argument("--pool", default="thread", choices=["thread", "process"],
                    help="thread pool (default, best for network/external drives) "
                         "or process pool (best for big local XMP/PSD files)")
    ap.add_argument("--regex", action="store_true", help="treat pattern as a regex")
    ap.add_argument("--case-sensitive", "-s", action="store_true",
                    help="case-sensitive match (default: insensitive)")
    ap.add_argument("--ext", nargs="+", metavar="EXT",
                    help="only these extensions, e.g. --ext png jpg")
    ap.add_argument("--fields", nargs="+", metavar="NAME",
                    help="only search these metadata fields, e.g. --fields prompt workflow")
    ap.add_argument("--hidden", action="store_true",
                    help="include dotfiles and dot-directories")
    ap.add_argument("--follow-symlinks", action="store_true",
                    help="descend into symlinked directories (with -r)")
    ap.add_argument("--link-dir", type=Path, metavar="DIR",
                    help="create a link/alias to each match in this folder")
    ap.add_argument("--link-type", default="symlink",
                    choices=["symlink", "alias", "hardlink", "copy"],
                    help="symlink (default), alias = real macOS Finder alias, "
                         "hardlink, or copy")
    ap.add_argument("--no-preserve-dates", action="store_true",
                    help="don't copy the original's creation and modification "
                         "dates onto the link/alias (they are copied by default)")
    ap.add_argument("--flatten", default="name", choices=["name", "path"],
                    help="link name: original filename (default) or the full "
                         "relative path joined with '__'")
    ap.add_argument("--deep", action="store_true",
                    help="also read PNG text chunks stored after the image "
                         "data and scan deep into files for stray XMP "
                         "packets — roughly 20x slower, rarely needed")
    ap.add_argument("--only-connected", "--only_connected", action="store_true",
                    help="in ComfyUI prompt/workflow metadata, only search "
                         "nodes that are wired to other nodes; unconnected, "
                         "muted and bypassed nodes, notes and reroutes are "
                         "ignored. Other metadata is searched as usual")
    ap.add_argument("--connected-mode", choices=["linked", "output"],
                    help="what --only-connected keeps: 'linked' (default) = "
                         "any node with a link; 'output' = only nodes that "
                         "feed a Save/Preview-type node, which also drops "
                         "wired-up leftovers. Implies --only-connected")
    ap.add_argument("--results", type=Path, metavar="FILE",
                    help="write a results.txt log (parameters, completed "
                         "folders, hit paths, stats); rerunning with the same "
                         "file resumes and skips completed folders")
    ap.add_argument("--no-resume", action="store_true",
                    help="with --results: start the file fresh instead of "
                         "resuming from it")
    ap.add_argument("--progress", default="auto", choices=["auto", "always", "never"],
                    help="progress bar on stderr (default: auto = only on a terminal)")
    ap.add_argument("--no-count", action="store_true",
                    help="skip the pre-count pass; show a running counter "
                         "instead of a percentage bar")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="also print the matching field and a snippet")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit results as JSON instead of plain paths")
    ap.add_argument("--errors", action="store_true",
                    help="report unreadable files on stderr")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        return _die(f"not a directory: {root}")

    exts = {("." + e.lower().lstrip(".")) for e in args.ext} if args.ext else None
    connected = ((args.connected_mode or "linked")
                 if args.only_connected or args.connected_mode else None)

    dest_dir = None
    if args.link_dir:
        dest_dir = args.link_dir.expanduser().resolve()
        if dest_dir == root or (args.recursive and root in dest_dir.parents):
            return _die("--link-dir must be outside the searched folder")
        if args.link_type == "alias" and sys.platform != "darwin":
            return _die("--link-type alias only works on macOS")
        if (not args.no_preserve_dates and _libc is None
                and args.link_type != "hardlink"):
            print("note: creation dates can only be set on macOS; "
                  "modification dates will still be copied", file=sys.stderr)
        dest_dir.mkdir(parents=True, exist_ok=True)

    if args.regex:
        for pattern in [*args.patterns, *(args.exclude or ())]:
            try:
                re.compile(pattern)
            except re.error as exc:
                return _die(f"bad regex {pattern!r}: {exc}")

    log = None
    if args.results:
        params = {
            "root": str(root),
            "patterns": args.patterns,
            "match": args.match,
            "scope": (args.scope if len(args.patterns) > 1
                      or args.scope == "node" else "image"),
            "regex": args.regex,
            "case_sensitive": args.case_sensitive,
            "recursive": args.recursive,
            "ext": sorted(exts) if exts else None,
            "fields": sorted(args.fields) if args.fields else None,
            "hidden": args.hidden,
            "follow_symlinks": args.follow_symlinks,
            "deep": args.deep,
        }
        if connected:  # only when set, so older results files still resume
            params["only_connected"] = connected
        if args.exclude:
            params["exclude"] = args.exclude
        try:
            log = ResultsFile(args.results.expanduser().resolve(), params,
                              allow_resume=not args.no_resume)
        except ParamMismatch as exc:
            return _die(f"{exc}\n       use --no-resume to overwrite it, "
                        f"or point --results at a different file")
        except OSError as exc:
            return _die(f"cannot open results file: {exc}")
        if log.resumed:
            print(f"resuming: {len(log.completed_dirs):,} folder(s) already "
                  f"done, {log.prior_hits:,} match(es) on record",
                  file=sys.stderr)

    skip_dirs = log.completed_dirs if log else frozenset()

    workers = args.workers or min(32, (os.cpu_count() or 4) * 2)
    workers = max(1, workers)

    # Every worker holds at least one file descriptor; leave headroom for the
    # results file, the pool's own pipes and stdio.
    fd_soft = raise_fd_limit(workers * 4 + 128)
    safe_workers = max(1, (fd_soft - 128) // 4)
    if workers > safe_workers:
        print(f"note: capping workers at {safe_workers} "
              f"(open-file limit is {fd_soft}; raise it with "
              f"`ulimit -n 10240` to use more)", file=sys.stderr)
        workers = safe_workers

    def paths_factory():
        return iter_work(root, args.recursive, exts,
                         args.follow_symlinks, args.hidden, skip_dirs)

    show_progress = (args.progress == "always" or
                     (args.progress == "auto" and sys.stderr.isatty()))
    total = None if args.no_count else count_images(paths_factory, show_progress)
    bar = Progress(total, show_progress)

    paths = paths_factory()

    init_args = ({
        "patterns": args.patterns, "exclude": args.exclude,
        "regex": args.regex,
        "case_sensitive": args.case_sensitive, "fields": args.fields,
        "snippets": args.verbose or args.as_json, "deep": args.deep,
        "mode": args.match, "scope": args.scope, "connected": connected,
    },)

    scanned = found = errors = 0
    results = []
    started = time.monotonic()
    interrupted = False

    # directory bookkeeping, so an interrupted run can resume folder by folder
    outstanding: dict[str, int] = {}   # dir -> jobs still in flight
    sealed: set[str] = set()           # walker has emitted all files for dir
    dirty: set[str] = set()            # dir had a read failure this run
    error_kinds: dict[str, int] = {}   # error type -> count

    def finish_dir(dirpath: str) -> None:
        if dirpath not in sealed or outstanding.get(dirpath, 0) != 0:
            return
        outstanding.pop(dirpath, None)
        sealed.discard(dirpath)
        if dirpath in dirty:
            # something in here couldn't be read; leave it unrecorded so a
            # rerun retries it instead of assuming the folder is done
            dirty.discard(dirpath)
            return
        if log:
            log.record_dir(dirpath)

    def seal(dirpath: str) -> None:
        sealed.add(dirpath)
        finish_dir(dirpath)

    def handle(path_str, hits, error, dirpath):
        nonlocal scanned, found, errors
        scanned += 1
        outstanding[dirpath] = outstanding.get(dirpath, 1) - 1

        if error:
            errors += 1
            kind = error.split(":", 1)[0]
            error_kinds[kind] = error_kinds.get(kind, 0) + 1
            dirty.add(dirpath)
            if log:
                log.record_error(path_str, error)
            bar.advance()
            if args.errors:
                bar.clear()
                print(f"[skip] {path_str}: {error}", file=sys.stderr)
                bar.draw(force=True)
            finish_dir(dirpath)
            return
        if not hits:
            bar.advance()
            finish_dir(dirpath)
            return

        found += 1
        bar.advance(hit=True)
        bar.clear()

        linked = None
        if dest_dir:  # done here in the parent, never in a worker
            try:
                linked = make_link(Path(path_str), dest_dir, args.link_type,
                                   args.flatten, root,
                                   preserve_dates=not args.no_preserve_dates)
            except Exception as exc:
                print(f"[link failed] {path_str}: {exc}", file=sys.stderr)

        if log:  # after the link, so the log never promises one that failed
            log.record_hit(path_str, hits, linked)

        if args.as_json:
            results.append({
                "path": path_str,
                "matches": [{"field": f, "snippet": s} for f, s in hits],
                "link": str(linked) if linked else None,
            })
        else:
            print(path_str, flush=True)
            if args.verbose:
                for field, snippet in hits:
                    print(f"    ({field}) {snippet}")
                if linked:
                    print(f"    -> {linked}")
        bar.draw(force=True)
        finish_dir(dirpath)

    try:
        if workers == 1:
            _worker_init(*init_args)
            for kind, dirpath, path_str in paths:
                if kind == "seal":
                    seal(dirpath)
                    continue
                outstanding[dirpath] = outstanding.get(dirpath, 0) + 1
                handle(*scan_one(path_str), dirpath)
        else:
            pool_cls = (ThreadPoolExecutor if args.pool == "thread"
                        else ProcessPoolExecutor)
            with pool_cls(max_workers=workers,
                          initializer=_worker_init,
                          initargs=init_args) as pool:
                futures: dict = {}          # future -> dirpath
                queue_cap = workers * 8     # keep memory flat on huge trees
                exhausted = False
                while True:
                    while not exhausted and len(futures) < queue_cap:
                        try:
                            kind, dirpath, path_str = next(paths)
                        except StopIteration:
                            exhausted = True
                            break
                        if kind == "seal":
                            seal(dirpath)
                            continue
                        outstanding[dirpath] = outstanding.get(dirpath, 0) + 1
                        futures[pool.submit(scan_one, path_str)] = dirpath
                    if not futures:
                        break
                    # handle everything that finished, not just the first one:
                    # each wait() registers a callback on every pending future,
                    # so draining the batch amortises that over the whole round
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        handle(*fut.result(), futures.pop(fut))
    except KeyboardInterrupt:
        interrupted = True
        bar.clear()
        print("interrupted — progress saved" if log else "interrupted",
              file=sys.stderr)

    bar.finish()

    if log:
        log.close(scanned, found, errors,
                  time.monotonic() - started, completed=not interrupted)

    if args.as_json:
        print(json.dumps(
            {"scanned": scanned, "matched": found, "unreadable": errors,
             "root": str(root), "recursive": args.recursive,
             "link_dir": str(dest_dir) if dest_dir else None,
             "results_file": str(log.path) if log else None,
             "resumed": bool(log and log.resumed),
             "interrupted": interrupted,
             "error_kinds": error_kinds,
             "results": results},
            indent=2, ensure_ascii=False,
        ))
    else:
        summary = f"\n{found} match(es) in {scanned} image(s) scanned"
        if errors:
            summary += f", {errors} unreadable"
            top = sorted(error_kinds.items(), key=lambda kv: -kv[1])[:3]
            for kind, count in top:
                summary += f"\n  {count:,} × {kind}"
                if kind == "TooManyOpenFiles":
                    summary += ("  <- lower -j, or run `ulimit -n 10240` "
                                "first; these files were NOT searched")
            summary += "\n  (rerun to retry them; use --errors to list them)"
        if dest_dir:
            summary += f"\nlinks in: {dest_dir}"
        if log:
            summary += f"\nresults:  {log.path}"
        print(summary, file=sys.stderr)

    if interrupted:
        return 130
    return 0 if (found or (log and log.prior_hits)) else 1  # grep-style status


def cli() -> None:
    """Console-script entry point."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("\r\x1b[2Kinterrupted\n")
        sys.exit(130)


if __name__ == "__main__":
    cli()