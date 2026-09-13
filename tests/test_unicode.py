"""Non-ASCII text: ComfyUI's JSON escapes, UTF-8 in tEXt chunks, Unicode case
and accent normalisation."""

import json
import unicodedata

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from helpers import hit_names, run_search

TEXT = "一个女孩在海边 少女 소녀 niña café 😀"


def graph(text, stray="STRAY 孤立节点"):
    """A tiny API-format prompt: a wired text node and an unwired one."""
    return {
        "4": {"inputs": {"ckpt_name": "模型.safetensors"},
              "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load"}},
        "6": {"inputs": {"text": text, "clip": ["4", 1]},
              "class_type": "CLIPTextEncode", "_meta": {"title": "正面提示词"}},
        "9": {"inputs": {"images": ["6", 0]},
              "class_type": "SaveImage", "_meta": {"title": "Save"}},
        "10": {"inputs": {"text": stray},
               "class_type": "CLIPTextEncode", "_meta": {"title": "Stray"}},
    }


def png(path, **chunks):
    info = PngInfo()
    for key, value in chunks.items():
        info.add_text(key, value)  # bytes are written raw into a tEXt chunk
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)


GRAPHS = {"comfy_escaped.png", "comfy_utf8.png", "comfy_escaped.webp"}
PLAIN = {"a1111_itxt.png", "utf8_in_text.png"}


@pytest.fixture(scope="module")
def uni_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("unicode")
    png(d / "comfy_escaped.png", prompt=json.dumps(graph(TEXT)))  # as ComfyUI writes it
    png(d / "comfy_utf8.png", prompt=json.dumps(graph(TEXT), ensure_ascii=False))
    png(d / "a1111_itxt.png", parameters=TEXT + "\nSteps: 20")
    png(d / "utf8_in_text.png", parameters=(TEXT + "\nSteps: 20").encode("utf-8"))
    # a literal backslash-u in the text, not an escape
    png(d / "escaped_backslash.png",
        prompt=json.dumps(graph("literal \\u4e00 is not a character")))
    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif[0x0110] = "prompt:" + json.dumps(graph(TEXT))  # ComfyUI WebP: EXIF Model
    img.save(d / "comfy_escaped.webp", exif=exif, lossless=True)
    return d


@pytest.mark.parametrize("word", ["女孩", "少女", "소녀", "niña", "café", "😀"])
def test_found_however_it_was_stored(uni_dir, word):
    assert hit_names(run_search(uni_dir, word)) == GRAPHS | PLAIN


def test_graph_only_text(uni_dir):
    everything = GRAPHS | {"escaped_backslash.png"}
    assert hit_names(run_search(uni_dir, "正面提示词")) == everything
    assert hit_names(run_search(uni_dir, "模型")) == everything


def test_literal_backslash_u_is_left_alone(uni_dir):
    assert "escaped_backslash.png" not in hit_names(run_search(uni_dir, "一"))
    # an all-ASCII term skips decoding, so it may also meet the raw escapes
    assert "escaped_backslash.png" in hit_names(run_search(uni_dir, "\\u4e00"))


def test_unicode_case_and_accent_composition(uni_dir):
    assert hit_names(run_search(uni_dir, "CAFÉ")) == GRAPHS | PLAIN
    decomposed = unicodedata.normalize("NFD", "café")  # e + combining accent
    assert hit_names(run_search(uni_dir, decomposed)) == GRAPHS | PLAIN
    assert hit_names(run_search(uni_dir, "CAFÉ", "--case-sensitive")) == set()


def test_only_connected_with_chinese(uni_dir):
    assert hit_names(run_search(uni_dir, "女孩", "--only-connected")) == GRAPHS | PLAIN
    assert hit_names(run_search(uni_dir, "孤立节点")) == GRAPHS | {"escaped_backslash.png"}
    assert hit_names(run_search(uni_dir, "孤立节点", "--only-connected")) == set()


def test_scope_node_with_chinese(uni_dir):
    # same node (or, for A1111, same field)
    assert hit_names(run_search(uni_dir, "女孩", "café", "--scope", "node")) == GRAPHS | PLAIN
    # prompt text and checkpoint name live in different nodes
    assert hit_names(run_search(uni_dir, "女孩", "模型", "--scope", "node")) == set()
    assert hit_names(run_search(uni_dir, "女孩", "模型")) == GRAPHS


def test_snippets_show_real_characters(uni_dir):
    report = run_search(uni_dir, "女孩", "--fields", "prompt", "--only-connected")
    snippets = [m["snippet"] for r in report["results"] for m in r["matches"]]
    assert snippets and all("女孩" in s and "\\u" not in s for s in snippets)


def test_ascii_terms_still_get_readable_snippets(uni_dir):
    report = run_search(uni_dir, "STRAY", "--fields", "prompt")
    snippets = [m["snippet"] for r in report["results"] for m in r["matches"]]
    assert snippets and all("孤立节点" in s for s in snippets)


def test_lone_surrogate_escape(tmp_path):
    png(tmp_path / "odd.png", prompt=json.dumps(graph("broken " + chr(0xD800) + " 女孩")))
    assert hit_names(run_search(tmp_path, "女孩")) == {"odd.png"}
    assert hit_names(run_search(tmp_path, "女孩", "--only-connected")) == {"odd.png"}


def test_escaped_quotes_from_other_writers(tmp_path):
    backslash = chr(92)
    text = json.dumps(graph('say "hi" 女孩'))
    # a JavaScript-style writer escaping quotes as backslash-u0022
    text = text.replace(backslash + '"hi' + backslash + '"',
                        backslash + "u0022hi" + backslash + "u0022")
    assert backslash + "u0022" in text
    png(tmp_path / "quotes.png", prompt=text)
    assert hit_names(run_search(tmp_path, "女孩", "--only-connected")) == {"quotes.png"}
    assert hit_names(run_search(tmp_path, "hi", "女孩", "--scope", "node")) == {"quotes.png"}


def test_control_character_escapes_still_parse(tmp_path):
    png(tmp_path / "bell.png", prompt=json.dumps(graph("bell" + chr(7) + " 一个女孩")))
    assert hit_names(run_search(tmp_path, "一个女孩", "--only-connected")) == {"bell.png"}
    assert hit_names(run_search(tmp_path, "一个女孩", "--scope", "node")) == {"bell.png"}
