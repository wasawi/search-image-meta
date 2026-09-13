#!/usr/bin/env python3
"""
search_string_image_meta.py — search image metadata for a string, in parallel.

Searches EXIF (incl. nested Exif/GPS IFDs), XMP, IPTC/APP13, JPEG comments,
PNG tEXt/iTXt chunks (ComfyUI `prompt` / `workflow` / `parameters`) and the
metadata of MP4 / MOV / WebM / MKV videos (ComfyUI, VideoHelperSuite).

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

    # real Finder aliases (macOS) or shortcuts (Windows) instead of symlinks
    python3 search_string_image_meta.py ~/output "Krea" -r --link-dir ~/matches --link-type alias

    # skip ComfyUI nodes that aren't wired into the graph (stray loaders, notes…)
    python3 search_string_image_meta.py ~/output "Krea" -r --only-connected

    # stricter: also skip wired-up leftovers that never reach a Save/Preview node
    python3 search_string_image_meta.py ~/output "Krea" -r --connected-mode output

    # both terms inside the SAME metadata field, not spread across fields
    python3 search_string_image_meta.py ~/output "Barcelona" "sunset" -r --scope field

    # both terms inside the SAME ComfyUI node (e.g. one prompt box), wired nodes only
    python3 search_string_image_meta.py ~/output "lighthouse" "dusk" -r --scope node --only-connected

    # a LoRA name, but only where a LoRA loader uses it
    python3 search_string_image_meta.py ~/output "darkbrush" -r --node-type Lora

    # only prompt text, not file names, titles or settings
    python3 search_string_image_meta.py ~/output "lighthouse" -r --input text value

    # only the ComfyUI graph chunks, only PNGs, exact case
    python3 search_string_image_meta.py ~/output "LoRA" -r -s --fields prompt workflow --ext png

    # show which field matched, with a snippet of the surrounding text
    python3 search_string_image_meta.py ~/output "Krea" -r -v

    # what made each match: prompts, models, LoRAs, seed / steps / sampler
    python3 search_string_image_meta.py ~/output "fox" -r --show

    # machine-readable results (paths, fields, snippets, stats)
    python3 search_string_image_meta.py ~/output "Krea" -r --json > krea_hits.json

    # a spreadsheet of matches: this month only, skipping old render folders
    python3 search_string_image_meta.py ~/output "Krea" -r --csv --since 2026-09-01 --exclude-dir "old_*" > krea.csv

    # hand the matches to another command, safe with any file name
    python3 search_string_image_meta.py ~/output "Krea" -r -0 --since 3d | xargs -0 ls -l

    # resumable log: rerun the same command to skip folders already done
    python3 search_string_image_meta.py /Volumes/Photos "Krea" -r --results ~/krea_results.txt
    python3 search_string_image_meta.py /Volumes/Photos "Krea" -r --results ~/krea_results.txt --no-resume

    # cache what was read; every later search of that drive only reads new files
    python3 search_string_image_meta.py /Volumes/Photos "Krea" -r --index ~/data_meta.sqlite

    # real copies named by their relative path (a/b/img.png -> a__b__img.png)
    python3 search_string_image_meta.py ~/output "Krea" -r --link-dir ~/matches --link-type copy --flatten path

    # thorough pass: late PNG chunks and deep XMP, and list unreadable files
    python3 search_string_image_meta.py ~/Pictures "Barcelona" -r --deep --errors

    # a slow network share: threads instead of processes; dot-folders and linked dirs too
    python3 search_string_image_meta.py /Volumes/NAS/Design "Photoshop" -r --pool thread -j 32 --ext psd tif --hidden --follow-symlinks

    # in scripts: no progress bar, no pre-count; exit status 0 = found, 1 = none
    python3 search_string_image_meta.py ~/output "Krea" -r --progress never --no-count > /dev/null && echo "found"
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata
import warnings
import zlib
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                ThreadPoolExecutor, wait)
from datetime import datetime, timedelta
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

# HEIC / HEIF (iPhone photos) need the optional pillow-heif package; without
# it those files are reported as unreadable.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


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

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
DEFAULT_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".png", ".gif", ".tif", ".tiff", ".webp",
    ".bmp", ".heic", ".heif", ".avif", ".jp2", ".dng", ".cr2", ".nef",
    ".arw", ".orf", ".rw2", ".raf", ".psd", ".ico",
} | VIDEO_EXTS

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


# --------------------------------------------------------------------------
# video containers (MP4 / MOV, WebM / MKV)
# --------------------------------------------------------------------------
#
# ComfyUI's SaveVideo / SaveWEBM store `prompt` and `workflow` as container
# metadata (MP4 "mdta" keys, Matroska tags); VideoHelperSuite puts both into
# one JSON `comment`. Only the metadata is read: boxes and elements are
# skipped by their size, so the video data itself is never loaded.

_META_LIMIT = 64 * 1024 * 1024  # never load a metadata block bigger than this
_MP4_FIRST_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide",
                    b"pnot", b"uuid"}
_MP4_ITEM_NAMES = {"\xa9cmt": "comment", "\xa9nam": "title",
                   "\xa9too": "encoder", "\xa9day": "date", "\xa9ART": "artist",
                   "desc": "description"}
_EBML_HEADER, _MKV_SEGMENT, _MKV_TAGS = 0x1A45DFA3, 0x18538067, 0x1254C367
_MKV_TAG, _MKV_SIMPLE_TAG = 0x7373, 0x67C8
_MKV_TAG_NAME, _MKV_TAG_STRING = 0x45A3, 0x4487


def _iter_boxes(buf: bytes, start: int, end: int):
    """(type, body start, end) of the ISO-BMFF boxes in buf[start:end]."""
    pos = start
    while pos + 8 <= end:
        size, kind, header = int.from_bytes(buf[pos:pos + 4], "big"), buf[pos + 4:pos + 8], 8
        if size == 1:
            size, header = int.from_bytes(buf[pos + 8:pos + 16], "big"), 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def _mp4_moov(path: Path):
    """The moov box of an MP4 / MOV, found by hopping over top-level boxes."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        pos, first = 0, True
        while pos + 8 <= size:
            fh.seek(pos)
            header = fh.read(16)
            box, kind, offset = int.from_bytes(header[:4], "big"), header[4:8], 8
            if first and kind not in _MP4_FIRST_BOXES:
                raise ValueError("not an MP4/MOV container")
            first = False
            if box == 1:
                box, offset = int.from_bytes(header[8:16], "big"), 16
            elif box == 0:
                box = size - pos
            if box < offset:
                raise ValueError("damaged MP4/MOV box")
            if kind == b"moov":
                if box > _META_LIMIT:
                    return b""
                fh.seek(pos + offset)
                return fh.read(box - offset)
            pos += box
    return b""


def _mp4_metadata(path: Path) -> dict:
    moov = _mp4_moov(path)
    fields: dict = {}

    def read_meta(start: int, end: int) -> None:
        # ISO "meta" is a full box (4 bytes of version/flags); QuickTime's isn't
        skip = 0 if moov[start + 4:start + 8] == b"hdlr" else 4
        children = list(_iter_boxes(moov, start + skip, end))
        keys = []
        for kind, body, stop in children:
            if kind == b"keys":
                pos = body + 8  # version/flags, entry count
                while pos + 8 <= stop:
                    size = int.from_bytes(moov[pos:pos + 4], "big")
                    if size < 8:
                        break
                    keys.append(moov[pos + 8:pos + size].decode("utf-8", "replace"))
                    pos += size
        for kind, body, stop in children:
            if kind != b"ilst":
                continue
            for item, item_body, item_end in _iter_boxes(moov, body, stop):
                number = int.from_bytes(item, "big")
                if 1 <= number <= len(keys):
                    name = keys[number - 1]
                else:
                    tag = item.decode("latin-1")
                    name = _MP4_ITEM_NAMES.get(tag, tag)
                for data, data_body, data_end in _iter_boxes(moov, item_body, item_end):
                    if data == b"data":  # 4 bytes type, 4 bytes locale
                        fields.setdefault(name, _decode_bytes(moov[data_body + 8:data_end]))

    def walk(start: int, end: int) -> None:
        for kind, body, stop in _iter_boxes(moov, start, end):
            if kind == b"udta":
                walk(body, stop)
            elif kind == b"meta":
                read_meta(body, stop)

    walk(0, len(moov))
    return fields


def _ebml_number(read, keep_marker: bool):
    """An EBML variable-length number from read(n); None if unknown or EOF."""
    first = read(1)
    if not first:
        return None, 0
    length, mask = 1, 0x80
    while length <= 8 and not first[0] & mask:
        length, mask = length + 1, mask >> 1
    if length > 8:
        raise ValueError("damaged Matroska element")
    rest = read(length - 1)
    if len(rest) < length - 1:
        return None, 0
    value = first[0] if keep_marker else first[0] & (mask - 1)
    for byte in rest:
        value = (value << 8) | byte
    if not keep_marker and value == (1 << (7 * length)) - 1:
        return None, length  # "unknown size"
    return value, length


def _ebml_children(buf: bytes, start: int, end: int):
    """(id, body start, end) of the EBML elements in buf[start:end]."""
    stream = io.BytesIO(buf)
    stream.seek(start)
    while stream.tell() < end:
        eid, _ = _ebml_number(stream.read, True)
        size, _ = _ebml_number(stream.read, False)
        body = stream.tell()
        if eid is None or size is None or body + size > end:
            return
        yield eid, body, body + size
        stream.seek(body + size)


def _mkv_metadata(path: Path) -> dict:
    fields: dict = {}
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if _ebml_number(fh.read, True)[0] != _EBML_HEADER:
            raise ValueError("not a WebM/MKV container")
        header_size, _ = _ebml_number(fh.read, False)
        fh.seek(header_size or 0, 1)
        if _ebml_number(fh.read, True)[0] != _MKV_SEGMENT:
            return fields
        segment, _ = _ebml_number(fh.read, False)
        end = size if segment is None else min(size, fh.tell() + segment)
        while fh.tell() < end:
            eid, _ = _ebml_number(fh.read, True)
            length, _ = _ebml_number(fh.read, False)
            if eid is None or length is None:
                break  # an unknown-size element can't be skipped
            body = fh.tell()
            if eid == _MKV_TAGS and length <= _META_LIMIT:
                buf = fh.read(length)

                def simple_tag(start, stop):
                    name = value = None
                    for cid, cstart, cstop in _ebml_children(buf, start, stop):
                        if cid == _MKV_TAG_NAME:
                            name = buf[cstart:cstop].decode("utf-8", "replace")
                        elif cid == _MKV_TAG_STRING:
                            value = buf[cstart:cstop].decode("utf-8", "replace")
                        elif cid == _MKV_SIMPLE_TAG:
                            simple_tag(cstart, cstop)
                    if name and value is not None:
                        fields.setdefault(name.lower(), value.rstrip("\x00"))

                for tid, tstart, tstop in _ebml_children(buf, 0, len(buf)):
                    if tid == _MKV_TAG:
                        for cid, cstart, cstop in _ebml_children(buf, tstart, tstop):
                            if cid == _MKV_SIMPLE_TAG:
                                simple_tag(cstart, cstop)
            fh.seek(body + length)
    return fields


def extract_video_metadata(path: Path) -> dict:
    """Container metadata of an MP4 / MOV or WebM / MKV file, {name: text}."""
    if path.suffix.lower() in (".webm", ".mkv"):
        fields = _mkv_metadata(path)
    else:
        fields = _mp4_metadata(path)
    # VideoHelperSuite: one JSON comment carrying both prompt and workflow
    for name, text in list(fields.items()):
        text = text.strip()
        if not text.startswith("{") or ('"prompt"' not in text
                                        and '"workflow"' not in text):
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        split = False
        for key in ("prompt", "workflow"):
            value = data.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            if isinstance(value, str) and key not in fields:
                fields[key] = value
                split = True
        if split:
            # searching the combined copy too would bypass the graph filters
            del fields[name]
    return fields


def extract_metadata(path: Path, deep: bool = False) -> dict:
    """Return {field_name: text} for everything we can read out of the image.

    Fast by default: only metadata that sits in the file header is touched, so
    the pixel data is never decoded. `deep` trades ~20x speed for two rare
    extras — PNG text chunks written *after* the image data, and XMP packets
    hiding further into the file than the header.
    """
    if path.suffix.lower() in VIDEO_EXTS:
        return extract_video_metadata(path)
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
# not strict: decoded control-character escapes sit raw inside JSON strings
_JSON = json.JSONDecoder(strict=False)


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


def _named_inputs(node: dict, kind: str, names) -> dict:
    """The node's inputs / widget values whose name is in `names` (lowercase).

    Prompt nodes name every input. Workflow nodes only do when a newer
    frontend saved `widgets_values_named`, or when the node keeps a dict of
    values (VideoHelperSuite); others are skipped — the prompt has them.
    """
    if kind == "prompt":
        values = node.get("inputs")
    else:
        values = node.get("widgets_values_named")
        if not isinstance(values, dict):
            values = node.get("widgets_values")
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items()
            if isinstance(k, str) and k.lower() in names}


def search_units(field: str, text: str, rxs, mode: str, scope: str,
                 connected, not_rxs=(), node_types=(), inputs=()) -> list:
    """Split one metadata field into the (label, text) units to match against.

    Normally that's the field itself. When it holds a ComfyUI graph:
      connected     drops the unconnected nodes first ("linked" / "output")
      scope "node"  makes each node its own unit, labelled "field#node"
      node_types    keeps only nodes whose type contains one of these
      inputs        keeps only these named inputs of each node
    The last two also make each node its own unit, and drop text that isn't
    a graph at all.
    What remains is re-serialised the way ComfyUI writes it (json.dumps, with
    the real characters prepare_text() restored), so patterns written against
    the raw metadata keep matching. A filtered
    workflow keeps only the kept nodes, their links and the subgraphs they
    use (not groups or canvas settings); with scope "node" only the nodes'
    own contents are searched.
    """
    nodes_only = bool(node_types or inputs)
    if not (connected or nodes_only) and scope != "node":
        return [(field, text)]
    unchanged = [] if nodes_only else [(field, text)]  # text that isn't a graph
    head = _GRAPH_HEAD.match(text)
    if not head or ('"class_type"' not in text and '"nodes"' not in text):
        return unchanged

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
        return unchanged
    kind = _graph_kind(data)
    if kind is None:
        return unchanged
    if connected:
        data = (_connected_workflow(data, connected) if kind == "workflow"
                else _connected_prompt(data, connected))
    if scope != "node" and not nodes_only:
        return [(field, head.group(0) + json.dumps(data, ensure_ascii=False))]

    units = []
    for label, node in _graph_nodes(data, kind):
        node_type = str(node.get("class_type" if kind == "prompt" else "type"))
        if node_types and not any(t in node_type.lower() for t in node_types):
            continue
        body = _named_inputs(node, kind, inputs) if inputs else node
        if body:
            units.append((f"{field}#{label}",
                          json.dumps(body, ensure_ascii=False)))
    return units


# --------------------------------------------------------------------------
# summaries (--show)
# --------------------------------------------------------------------------

_MODEL_FILE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx)$",
                         re.IGNORECASE)
_TEXT_INPUTS = {"text", "text_g", "text_l", "t5xxl", "clip_l", "prompt",
                "string", "value", "string_a", "string_b", "text_a", "text_b"}
# links that never lead to prompt text; not following them keeps traces short
_NON_TEXT_LINKS = {"clip", "model", "vae", "image", "images", "pixels",
                   "samples", "latent", "latent_image", "mask", "control_net",
                   "noise", "sigmas", "sampler", "guider", "upscale_model",
                   "clip_vision"}
_SAMPLER_KEYS = ("seed", "noise_seed", "steps", "cfg", "sampler_name",
                 "scheduler", "denoise")
_A1111_KEYS = {"Seed": "seed", "Steps": "steps", "CFG scale": "cfg",
               "Sampler": "sampler_name", "Schedule type": "scheduler",
               "Denoising strength": "denoise"}


def _is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2


def _lora(name, strength) -> str:
    return name if strength in (None, "") else f"{name} ({strength})"


def _texts_feeding(prompt: dict, nid: str, seen: set) -> list:
    """Prompt text that reaches node `nid`, traced upstream in input order."""
    if nid in seen or nid not in prompt:
        return []
    seen.add(nid)
    texts = []
    for name, value in (prompt[nid].get("inputs") or {}).items():
        name = name.lower()
        if _is_link(value):
            if name not in _NON_TEXT_LINKS:
                texts += _texts_feeding(prompt, str(value[0]), seen)
        elif name in _TEXT_INPUTS and isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def _summarize_prompt(prompt: dict) -> dict:
    prompt = _connected_prompt(prompt, "output")  # only what made the image

    def trace(role, class_hint=""):
        texts, seen = [], set()
        for node in prompt.values():
            value = (node.get("inputs") or {}).get(role)
            if (_is_link(value)
                    and class_hint in str(node.get("class_type")).lower()):
                texts += _texts_feeding(prompt, str(value[0]), seen)
        return list(dict.fromkeys(texts))

    # Flux-style guiders take a plain "conditioning" instead of positive
    positive = trace("positive") or trace("conditioning", "guider")
    negative = [t for t in trace("negative") if t not in positive]

    models, loras, primary, extra = [], [], [], {}
    for node in prompt.values():
        inputs = node.get("inputs") or {}
        for name, value in inputs.items():
            if isinstance(value, dict) and value.get("lora"):
                if value.get("on", True):  # rgthree Power Lora slot
                    loras.append(_lora(value["lora"], value.get("strength")))
            elif isinstance(value, str) and _MODEL_FILE.search(value):
                if "lora" in name.lower():
                    loras.append(_lora(value, inputs.get(
                        "strength_model", inputs.get("strength"))))
                else:
                    models.append(value)
        settings = {k: inputs[k] for k in _SAMPLER_KEYS
                    if k in inputs and not _is_link(inputs[k])}
        if "steps" in settings:
            primary.append(settings)
        else:  # custom sampling splits these over noise/sampler/guider nodes
            for k, v in settings.items():
                extra.setdefault(k, v)

    samplers = primary or ([extra] if extra else [])
    if len(samplers) == 1:
        samplers = [{**samplers[0],
                     **{k: v for k, v in extra.items() if k not in samplers[0]}}]
    unique = []
    for s in samplers:
        if s not in unique:
            unique.append(s)
    return {"positive": positive, "negative": negative,
            "models": list(dict.fromkeys(models)),
            "loras": list(dict.fromkeys(loras)), "samplers": unique}


def _summarize_a1111(text: str) -> dict:
    lines = text.strip().splitlines()
    settings = lines.pop() if lines and "Steps:" in lines[-1] else ""
    positive, _, negative = "\n".join(lines).partition("Negative prompt:")
    positive, negative = positive.strip(), negative.strip()
    values = {k.strip(): v.strip().strip('"') for k, v in
              re.findall(r'(?:^|,)\s*([\w ]+):\s*("[^"]*"|[^,]*)', settings)}
    sampler = {key: values[name] for name, key in _A1111_KEYS.items()
               if name in values}
    return {
        "positive": [positive] if positive else [],
        "negative": [negative] if negative else [],
        "models": [values["Model"]] if values.get("Model") else [],
        "loras": [_lora(name, weight) for name, weight in
                  re.findall(r"<lora:([^:>]+):?([^:>]*)", positive)],
        "samplers": [sampler] if sampler else [],
    }


def summarize(meta: dict) -> dict:
    """What produced an image: prompts, models, LoRAs, sampler settings.

    Read from a ComfyUI prompt (only the nodes that reach an output) or from
    A1111-style parameters; {} when the image has neither.
    """
    for text in meta.values():
        text = prepare_text(text)
        head = _GRAPH_HEAD.match(text)
        if head and '"class_type"' in text:
            try:
                data, _end = _JSON.raw_decode(text, head.end())
            except ValueError:
                continue
            if _graph_kind(data) == "prompt":
                return _summarize_prompt(data)
    for key, text in meta.items():
        if key.lower() == "parameters" and "Steps:" in text:
            return _summarize_a1111(prepare_text(text))
    return {}


def summary_text(summary: dict) -> dict:
    """The summary as flat strings, for plain and CSV output."""
    def one_line(text):
        return " ".join(text.split())

    return {
        "positive": " | ".join(one_line(t) for t in summary.get("positive", ())),
        "negative": " | ".join(one_line(t) for t in summary.get("negative", ())),
        "models": ", ".join(summary.get("models", ())),
        "loras": ", ".join(summary.get("loras", ())),
        "sampler": "; ".join(", ".join(f"{k} {v}" for k, v in s.items())
                             for s in summary.get("samplers", ())),
    }


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


_SURROGATE = re.compile("[" + chr(0xD800) + "-" + chr(0xDFFF) + "]")
_QUOTE_ESCAPES = re.compile(r"[\\]u00(?:22|5[cC])")


def _decode_escapes(text: str) -> str:
    """Turn JSON unicode escapes into characters, leaving everything else."""
    if not _QUOTE_ESCAPES.search(text):
        try:
            # raw_unicode_escape decodes backslash-u escapes and nothing else,
            # tells an escaped backslash from an escape, and runs in C; text
            # beyond Latin-1 goes in as escapes itself and comes back out
            decoded = (text.encode("latin-1", "backslashreplace")
                       .decode("raw_unicode_escape"))
            if _SURROGATE.search(decoded):  # emoji arrive as surrogate pairs
                decoded = decoded.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
            return decoded
        except UnicodeError:
            pass  # a cut-off or lone escape: go one at a time
    # escaped quotes / backslashes would break the JSON if decoded wholesale
    return _JSON_ESCAPE.sub(_unescape, text)


def prepare_text(text: str) -> str:
    """Metadata text the way a person would type it into a search.

    Decodes JSON unicode escapes and composes accents (NFC), so "café" typed
    in a terminal matches however the file stored it.
    """
    if _BACKSLASH_U in text and _JSON_START.match(text):
        text = _decode_escapes(text)
    if not text.isascii():
        text = unicodedata.normalize("NFC", text)
    return text


# --------------------------------------------------------------------------
# metadata index (--index)
# --------------------------------------------------------------------------
#
# A SQLite file remembering what extract_metadata() found in each file, keyed
# by path, size and modification time. Workers read it, each thread or
# process on its own connection; only the parent writes, in batches.

INDEX_VERSION = "1"  # bump whenever extraction changes, to re-read every file
_INDEX_LOCAL = threading.local()


class MetadataIndex:
    """The parent's side of --index: schema, counters and batched writes."""

    BATCH = 256

    def __init__(self, path: Path):
        self.path = path
        self.cached = self.read = 0
        self.rebuilt = False
        self._pending: list = []
        self._db = sqlite3.connect(str(path), timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS info "
                         "(key TEXT PRIMARY KEY, value TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS files ("
                         "path TEXT PRIMARY KEY, size INTEGER NOT NULL, "
                         "mtime_ns INTEGER NOT NULL, deep INTEGER NOT NULL, "
                         "meta BLOB NOT NULL)")
        row = self._db.execute(
            "SELECT value FROM info WHERE key = 'version'").fetchone()
        if row is None or row[0] != INDEX_VERSION:
            self.rebuilt = row is not None
            self._db.execute("DELETE FROM files")
            self._db.execute("INSERT OR REPLACE INTO info VALUES ('version', ?)",
                             (INDEX_VERSION,))
        self._db.commit()

    def add(self, path: str, size: int, mtime_ns: int, deep: bool,
            meta: dict) -> None:
        blob = zlib.compress(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self._pending.append((path, size, mtime_ns, int(deep), blob))
        self.read += 1
        if len(self._pending) >= self.BATCH:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self._db.executemany(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?)",
                self._pending)
            self._db.commit()
            self._pending.clear()

    def close(self) -> None:
        self.flush()
        self._db.close()


def index_lookup(index_path: str, path: str, size: int, mtime_ns: int,
                 deep: bool):
    """Cached metadata for a file that hasn't changed, else None (workers)."""
    connections = getattr(_INDEX_LOCAL, "connections", None)
    if connections is None:
        connections = _INDEX_LOCAL.connections = {}
    try:
        db = connections.get(index_path)
        if db is None:
            db = connections[index_path] = sqlite3.connect(index_path,
                                                           timeout=60)
        row = db.execute("SELECT deep, meta FROM files "
                         "WHERE path = ? AND size = ? AND mtime_ns = ?",
                         (path, size, mtime_ns)).fetchone()
    except sqlite3.Error:
        return None  # a busy or damaged index just means reading the file
    if row is None or (deep and not row[0]):
        return None  # a --deep search can't use a shallow read
    return json.loads(zlib.decompress(row[1]).decode("utf-8"))


# --------------------------------------------------------------------------
# parallel scanning
# --------------------------------------------------------------------------

_CFG: dict = {}  # per-worker state, set by _worker_init


def _worker_init(opts: dict):
    """Set up a worker from the options main() collected.

    patterns, exclude, regex, case_sensitive, fields, snippets, deep, mode
    ("all" / "any"), scope ("image" / "field" / "node"), connected (None /
    "linked" / "output"), node_types, inputs.
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
    # Decoding escapes and composing accents only ever produces non-ASCII
    # characters, so all-ASCII terms can't gain a match from it: skip the work
    # (graph views still decode, since parsing the JSON does)
    _CFG["unicode"] = not all(p.isascii() for p in
                              [*opts["patterns"], *(opts.get("exclude") or ())])
    _CFG["node_types"] = [t.lower() for t in opts.get("node_types") or ()]
    _CFG["inputs"] = {n.lower() for n in opts.get("inputs") or ()}
    _CFG["fields"] = ({f.lower() for f in opts["fields"]}
                      if opts["fields"] else None)


def _snippet(text: str, match: re.Match) -> str:
    start = max(0, match.start() - SNIPPET_WIDTH // 3)
    end = min(len(text), match.end() + SNIPPET_WIDTH // 2)
    out = text[start:end]
    if _BACKSLASH_U in out:  # readable even when the text wasn't decoded
        out = _decode_escapes(out)
    out = _clean(out).replace("\n", " ")
    return ("…" if start else "") + out + ("…" if end < len(text) else "")


def scan_one(path_str: str):
    """Worker: (path, [(field, snippet)], error, summary, index record).

    The index record is None without --index, "cached" when the metadata
    came from it, or (size, mtime_ns, metadata) for the parent to store.
    """
    deep, index_path = _CFG.get("deep", False), _CFG.get("index")
    record = None
    try:
        meta = None
        if index_path:
            st = os.stat(path_str)
            meta = index_lookup(index_path, path_str, st.st_size,
                                st.st_mtime_ns, deep)
            record = "cached" if meta is not None else None
        if meta is None:
            meta = extract_metadata(Path(path_str), deep=deep)
            if index_path:
                record = (st.st_size, st.st_mtime_ns, meta)
    except Exception as exc:
        kind = type(exc).__name__
        errno = getattr(exc, "errno", None)
        if errno in (23, 24):  # ENFILE / EMFILE — a limit, not a bad file
            kind = "TooManyOpenFiles"
        return path_str, [], f"{kind}: {exc}", None, None

    rxs, only = _CFG["rx"], _CFG["fields"]
    units = []
    for f, t in meta.items():
        if t and not (only and f.lower() not in only):
            text = prepare_text(t) if _CFG["unicode"] else t
            units.extend(search_units(f, text, rxs, _CFG["mode"],
                                      _CFG["scope"], _CFG["connected"],
                                      _CFG["not_rx"], _CFG["node_types"],
                                      _CFG["inputs"]))
    hits = match_units(units, rxs, _CFG["mode"], _CFG["scope"],
                       _CFG["snippets"])
    if hits and any(n.search(text) for _, text in units for n in _CFG["not_rx"]):
        hits = []  # --not: an excluded term turned up
    summary = summarize(meta) if hits and _CFG.get("show") else None
    return path_str, hits, None, summary, record


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


def _is_hidden(dirpath: str, name: str) -> bool:
    """Dot-names everywhere; on Windows also the hidden file attribute."""
    if name.startswith("."):
        return True
    if os.name == "nt":
        try:
            attributes = os.stat(os.path.join(dirpath, name),
                                 follow_symlinks=False).st_file_attributes
        except OSError:
            return False
        return bool(attributes & 0x2)  # FILE_ATTRIBUTE_HIDDEN
    return False


def iter_work(root: Path, recursive: bool, exts, follow_symlinks: bool,
              include_hidden: bool, skip_dirs=frozenset(), exclude_dirs=(),
              since=None, until=None):
    """Lazily yield ("file", dirpath, fullpath) then ("seal", dirpath, None).

    A "seal" event means the walker has emitted every candidate file in that
    directory, so once its outstanding jobs finish the directory is complete
    and can be recorded as such. `exclude_dirs` are folder-name wildcards
    (any case); `since` / `until` bound the modification time.
    """
    if recursive:
        walker = os.walk(root, followlinks=follow_symlinks)
    else:
        walker = [(str(root), [],
                   [e.name for e in os.scandir(root) if e.is_file()])]
    excluded = [pattern.lower() for pattern in exclude_dirs]
    dated = since is not None or until is not None

    for dirpath, dirnames, filenames in walker:
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not _is_hidden(dirpath, d)]
        if excluded:
            dirnames[:] = [d for d in dirnames if not any(
                fnmatch.fnmatchcase(d.lower(), pattern) for pattern in excluded)]
        if dirpath in skip_dirs:
            continue  # finished on an earlier run
        for filename in filenames:
            suffix = os.path.splitext(filename)[1].lower()
            if exts is not None:
                if suffix not in exts:
                    continue
            elif suffix not in DEFAULT_EXTS:
                continue
            if not include_hidden and _is_hidden(dirpath, filename):
                continue
            path = os.path.join(dirpath, filename)
            if dated:
                try:
                    mtime = os.stat(path).st_mtime
                except OSError:
                    continue  # vanished while we were looking
                if ((since is not None and mtime < since)
                        or (until is not None and mtime >= until)):
                    continue
            yield "file", dirpath, path
        yield "seal", dirpath, None


_RELATIVE_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*([smhdw])", re.IGNORECASE)
_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_when(text: str, end_of_day: bool = False) -> float:
    """Timestamp for --since / --until.

    Accepts 2026-09-01, "2026-09-01 18:30" (local time), or an age such as
    30m, 12h, 3d, 2w. With `end_of_day` a bare date means the midnight after
    it, so --until 2026-09-01 still includes that whole day.
    """
    text = text.strip()
    age = _RELATIVE_TIME.fullmatch(text)
    if age:
        return time.time() - float(age.group(1)) * _SECONDS[age.group(2).lower()]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            when = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if end_of_day and fmt == "%Y-%m-%d":
            when += timedelta(days=1)
        return when.timestamp()
    raise ValueError(f"not a date or age: {text!r} (use 2026-09-01, "
                     f"'2026-09-01 18:30', or an age like 3d, 12h, 2w)")


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
# timestamp preservation (creation dates on macOS and Windows)
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


# Windows keeps one too; SetFileTime can change it.
_kernel32 = None
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes

        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                          wintypes.DWORD, wintypes.LPVOID,
                                          wintypes.DWORD, wintypes.DWORD,
                                          wintypes.HANDLE]
        _kernel32.CreateFileW.restype = wintypes.HANDLE
        _kernel32.SetFileTime.argtypes = [wintypes.HANDLE,
                                          ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME),
                                          ctypes.POINTER(wintypes.FILETIME)]
        _kernel32.SetFileTime.restype = wintypes.BOOL
        _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    except Exception:
        _kernel32 = None

CAN_SET_CREATION_TIME = _libc is not None or _kernel32 is not None


def _set_creation_time_windows(path: Path, birthtime_ns: int, follow: bool) -> bool:
    import ctypes
    from ctypes import wintypes

    flags = 0x02000000                      # FILE_FLAG_BACKUP_SEMANTICS
    if not follow:
        flags |= 0x00200000                 # FILE_FLAG_OPEN_REPARSE_POINT
    handle = _kernel32.CreateFileW(str(path), 0x100,  # FILE_WRITE_ATTRIBUTES
                                   0x1 | 0x2 | 0x4, None, 3,  # OPEN_EXISTING
                                   flags, None)
    if handle is None or handle == ctypes.c_void_p(-1).value:
        return False
    try:
        ticks = birthtime_ns // 100 + 116444736000000000  # since 1601, 100ns
        stamp = wintypes.FILETIME(ticks & 0xFFFFFFFF, ticks >> 32)
        return bool(_kernel32.SetFileTime(handle, ctypes.byref(stamp), None, None))
    finally:
        _kernel32.CloseHandle(handle)


def set_creation_time(path: Path, birthtime_ns: int, follow: bool = True) -> bool:
    """Set a file's creation date (macOS, Windows). False where unsupported."""
    if _kernel32 is not None:
        return _set_creation_time_windows(path, birthtime_ns, follow)
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
    if birth_ns is None and os.name == "nt":
        birth_ns = st.st_ctime_ns  # the creation time on Windows before 3.12
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
                        "exclude", "node_types", "inputs",
                        "exclude_dirs", "since", "until"):
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
        f'  set theFile to POSIX file {_applescript_string(src)} as alias\n'
        f'  set theDir to POSIX file {_applescript_string(dest_dir)} as alias\n'
        '  set newAlias to make new alias file at theDir to theFile\n'
        f'  set name of newAlias to {_applescript_string(target.name)}\n'
        'end tell'
    )
    subprocess.run(["osascript", "-e", script],
                   check=True, capture_output=True, text=True)
    return target


def _applescript_string(text) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _make_windows_shortcut(src: Path, dest_dir: Path, name: str) -> Path:
    """A Windows shortcut (.lnk), made through PowerShell's WScript.Shell."""
    target = _unique_target(dest_dir, name + ".lnk")

    def quoted(path) -> str:
        return "'" + str(path).replace("'", "''") + "'"

    script = (f"$s = (New-Object -ComObject WScript.Shell)"
              f".CreateShortcut({quoted(target)}); "
              f"$s.TargetPath = {quoted(src)}; $s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                    script], check=True, capture_output=True, text=True)
    return target


def _link_once(src: Path, target: Path, mode: str) -> None:
    if mode == "symlink":
        if target.is_symlink():
            target.unlink()
        target.symlink_to(src)
    elif mode == "hardlink":
        if not target.exists():
            os.link(src, target)
    elif mode == "copy":
        shutil.copy2(src, target)
    else:
        raise ValueError(f"unknown link type: {mode}")


# what --link-type auto tries, in order: on Windows symlinks need Developer
# Mode or admin rights, and hardlinks only work within one drive
AUTO_LINK_TYPES = (["symlink", "hardlink", "copy"] if os.name == "nt"
                   else ["symlink"])


def make_link(src: Path, dest_dir: Path, mode: str, flatten: str, root: Path,
              preserve_dates: bool = True) -> Path:
    """Put a link to `src` into `dest_dir`; returns what was created.

    "auto" tries AUTO_LINK_TYPES in order until one works.
    """
    if flatten == "path":
        try:
            name = "__".join(src.relative_to(root).parts)
        except ValueError:
            name = src.name
    else:
        name = src.name

    if mode == "alias":
        make = _make_windows_shortcut if os.name == "nt" else _make_finder_alias
        target = make(src, dest_dir, name)
        if preserve_dates:
            copy_timestamps(src, target, mode)
        return target

    attempts = AUTO_LINK_TYPES if mode == "auto" else [mode]
    for attempt in attempts:
        target = _unique_target(dest_dir, name, src)
        try:
            _link_once(src, target, attempt)
        except OSError:
            if attempt == attempts[-1]:
                raise
            continue
        if preserve_dates:
            copy_timestamps(src, target, attempt)
        return target
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _setup_stdio() -> None:
    """Never crash printing a path or prompt the output can't encode.

    Piped output is UTF-8 everywhere (Windows would otherwise use its ANSI
    code page); a console that can't show a character gets a replacement.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # replaced by something that isn't a real text stream
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _ansi_terminal(stream) -> bool:
    """Whether `stream` is a terminal that understands the bar's escape codes.

    Windows consoles only do after ENABLE_VIRTUAL_TERMINAL_PROCESSING is on.
    """
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


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
    ap.add_argument("--node-type", nargs="+", metavar="TYPE", dest="node_types",
                    help="only search ComfyUI nodes whose type contains one "
                         "of these (any case), e.g. Lora, CLIPTextEncode, "
                         "KSampler. Metadata that isn't a ComfyUI graph is "
                         "skipped")
    ap.add_argument("--input", nargs="+", metavar="NAME", dest="inputs",
                    help="only search these node inputs / widget values, "
                         "e.g. text ckpt_name lora_name seed (exact names, "
                         "any case). Uses the prompt, and workflows saved by "
                         "newer ComfyUI frontends. Metadata that isn't a "
                         "ComfyUI graph is skipped")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="descend into subfolders (default: top level only)")
    ap.add_argument("-j", "--workers", type=int, default=0, metavar="N",
                    help="parallel workers (default 0 = auto: half the CPU cores "
                         "for processes, twice the cores for threads; 1 = serial)")
    ap.add_argument("--pool", default="process", choices=["process", "thread"],
                    help="process pool (default: uses every CPU core, about "
                         "twice as fast on big searches) or thread pool "
                         "(starts instantly; can suit slow network drives)")
    ap.add_argument("--regex", action="store_true", help="treat pattern as a regex")
    ap.add_argument("--case-sensitive", "-s", action="store_true",
                    help="case-sensitive match (default: insensitive)")
    ap.add_argument("--ext", nargs="+", metavar="EXT",
                    help="only these extensions, e.g. --ext png jpg")
    ap.add_argument("--fields", nargs="+", metavar="NAME",
                    help="only search these metadata fields, e.g. --fields prompt workflow")
    ap.add_argument("--exclude-dir", nargs="+", metavar="NAME",
                    dest="exclude_dirs",
                    help="with -r: skip folders whose name matches one of "
                         "these (wildcards allowed, any case), e.g. "
                         "--exclude-dir 'old_*' thumbnails")
    ap.add_argument("--since", metavar="WHEN",
                    help="only files modified at or after WHEN: 2026-09-01, "
                         "'2026-09-01 18:30', or an age like 30m, 12h, 3d, 2w")
    ap.add_argument("--until", metavar="WHEN",
                    help="only files modified before WHEN (a date on its own "
                         "includes that whole day)")
    ap.add_argument("--hidden", action="store_true",
                    help="include hidden files and folders (dot-names, "
                         "and the hidden attribute on Windows)")
    ap.add_argument("--follow-symlinks", action="store_true",
                    help="descend into symlinked directories (with -r)")
    ap.add_argument("--link-dir", type=Path, metavar="DIR",
                    help="create a link/alias to each match in this folder")
    ap.add_argument("--link-type", default="auto",
                    choices=["auto", "symlink", "alias", "hardlink", "copy"],
                    help="auto (default) = symlink, or on Windows a symlink "
                         "if allowed, else a hardlink, else a copy; alias = "
                         "macOS Finder alias or Windows shortcut; symlink; "
                         "hardlink; copy")
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
    ap.add_argument("--index", type=Path, metavar="FILE",
                    help="remember what was read from every file in this "
                         "SQLite file; later searches (any pattern, any "
                         "option) skip files that haven't changed. Delete it "
                         "to start over")
    ap.add_argument("--progress", default="auto", choices=["auto", "always", "never"],
                    help="progress bar on stderr (default: auto = only on a terminal)")
    ap.add_argument("--no-count", action="store_true",
                    help="skip the pre-count pass; show a running counter "
                         "instead of a percentage bar")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="also print the matching field and a snippet")
    ap.add_argument("--show", action="store_true",
                    help="summarise what made each match: prompts, models, "
                         "LoRAs and sampler settings, from a ComfyUI prompt "
                         "(nodes that reach an output) or A1111 parameters")
    output = ap.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", dest="as_json",
                        help="emit results as JSON instead of plain paths")
    output.add_argument("--csv", action="store_true", dest="as_csv",
                        help="emit CSV, one row per matching field: path, "
                             "field, snippet, link")
    output.add_argument("-0", "--null", action="store_true",
                        help="print paths separated by NUL characters, for "
                             "xargs -0")
    ap.add_argument("--errors", action="store_true",
                    help="report unreadable files on stderr")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    _setup_stdio()
    args = parse_args(argv)

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        return _die(f"not a directory: {root}")

    exts = {("." + e.lower().lstrip(".")) for e in args.ext} if args.ext else None
    connected = ((args.connected_mode or "linked")
                 if args.only_connected or args.connected_mode else None)
    try:
        since = parse_when(args.since) if args.since else None
        until = parse_when(args.until, end_of_day=True) if args.until else None
    except ValueError as exc:
        return _die(str(exc))

    dest_dir = None
    if args.link_dir:
        dest_dir = args.link_dir.expanduser().resolve()
        if dest_dir == root or (args.recursive and root in dest_dir.parents):
            return _die("--link-dir must be outside the searched folder")
        if args.link_type == "alias" and sys.platform not in ("darwin", "win32"):
            return _die("--link-type alias makes a macOS Finder alias or a "
                        "Windows shortcut; use symlink on this system")
        if (not args.no_preserve_dates and not CAN_SET_CREATION_TIME
                and args.link_type != "hardlink"):
            print("note: creation dates can only be set on macOS and Windows; "
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
        if args.node_types:
            params["node_types"] = args.node_types
        if args.inputs:
            params["inputs"] = args.inputs
        for key in ("exclude_dirs", "since", "until"):
            if getattr(args, key):
                params[key] = getattr(args, key)
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

    index = None
    if args.index:
        try:
            index = MetadataIndex(args.index.expanduser().resolve())
        except (sqlite3.Error, OSError) as exc:
            return _die(f"cannot open index {args.index}: {exc}")

    skip_dirs = log.completed_dirs if log else frozenset()

    cpus = os.cpu_count() or 4
    # Processes do the CPU work in parallel and gain nothing past about half
    # the cores (they end up waiting on the disk and each other; measured on
    # 55k images). Threads mostly wait on I/O, so more of them still help.
    auto = (max(2, min(16, cpus // 2)) if args.pool == "process"
            else min(32, cpus * 2))
    workers = max(1, args.workers or auto)

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
                         args.follow_symlinks, args.hidden, skip_dirs,
                         args.exclude_dirs or (), since, until)

    ansi = _ansi_terminal(sys.stderr)  # also switches it on for Windows
    show_progress = (args.progress == "always" or
                     (args.progress == "auto" and ansi))
    total = None if args.no_count else count_images(paths_factory, show_progress)
    bar = Progress(total, show_progress)

    paths = paths_factory()

    init_args = ({
        "patterns": args.patterns, "exclude": args.exclude,
        "regex": args.regex,
        "case_sensitive": args.case_sensitive, "fields": args.fields,
        "snippets": args.verbose or args.as_json or args.as_csv,
        "deep": args.deep,
        "mode": args.match, "scope": args.scope, "connected": connected,
        "node_types": args.node_types, "inputs": args.inputs,
        "show": args.show, "index": str(index.path) if index else None,
    },)

    scanned = found = errors = 0
    results = []
    csv_out = csv.writer(sys.stdout, lineterminator="\n") if args.as_csv else None
    if csv_out:
        csv_out.writerow(["path", "field", "snippet", "link"]
                         + (["positive", "negative", "models", "loras",
                             "sampler"] if args.show else []))
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

    def handle(path_str, hits, error, summary, record, dirpath):
        nonlocal scanned, found, errors
        scanned += 1
        if record == "cached":
            index.cached += 1
        elif record is not None:
            index.add(path_str, *record[:2], args.deep, record[2])
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
            entry = {
                "path": path_str,
                "matches": [{"field": f, "snippet": s} for f, s in hits],
                "link": str(linked) if linked else None,
            }
            if args.show:
                entry["summary"] = summary or {}
            results.append(entry)
        elif csv_out:
            extra = (list(summary_text(summary or {}).values())
                     if args.show else [])
            for field, snippet in hits:
                csv_out.writerow([path_str, field, snippet,
                                  str(linked) if linked else "", *extra])
            sys.stdout.flush()
        elif args.null:
            sys.stdout.write(path_str + "\0")
            sys.stdout.flush()
        else:
            print(path_str, flush=True)
            if args.verbose:
                for field, snippet in hits:
                    print(f"    ({field}) {snippet}")
                if linked:
                    print(f"    -> {linked}")
            if args.show:
                for key, value in summary_text(summary or {}).items():
                    if value:
                        print(f"    {key + ':':<9} {value}")
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
    if index:
        index.close()

    if log:
        log.close(scanned, found, errors,
                  time.monotonic() - started, completed=not interrupted)

    if args.as_json:
        print(json.dumps(
            {"scanned": scanned, "matched": found, "unreadable": errors,
             "root": str(root), "recursive": args.recursive,
             "link_dir": str(dest_dir) if dest_dir else None,
             "results_file": str(log.path) if log else None,
             "index": ({"path": str(index.path), "cached": index.cached,
                        "read": index.read, "rebuilt": index.rebuilt}
                       if index else None),
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
        if index:
            summary += (f"\nindex:    {index.cached:,} from cache, "
                        f"{index.read:,} read"
                        + (" (rebuilt for a new version)" if index.rebuilt else ""))
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