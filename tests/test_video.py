"""Videos and AVIF: ComfyUI SaveVideo / SaveWEBM metadata, VideoHelperSuite
comments, ComfyUI AVIF EXIF. Files are written with PyAV the way ComfyUI does."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, features

from helpers import hit_names, run_search

av = pytest.importorskip("av")

PROMPT = {
    "1": {"inputs": {"text": "VIDEO_marker 视频 a paper boat", "clip": ["3", 1]},
          "class_type": "CLIPTextEncode", "_meta": {"title": "Positive"}},
    "2": {"inputs": {"text": "VIDEO_stray unused"},
          "class_type": "CLIPTextEncode", "_meta": {"title": "Stray"}},
    "3": {"inputs": {"ckpt_name": "wan2.2_video.safetensors"},
          "class_type": "CheckpointLoaderSimple", "_meta": {"title": "Load"}},
    "4": {"inputs": {"seed": 5, "steps": 12, "cfg": 6, "sampler_name": "uni_pc",
                     "scheduler": "simple", "denoise": 1, "model": ["3", 0],
                     "positive": ["1", 0]},
          "class_type": "KSampler", "_meta": {"title": "KSampler"}},
    "5": {"inputs": {"images": ["4", 0], "filename_prefix": "video/boat"},
          "class_type": "SaveVideo", "_meta": {"title": "Save Video"}},
}
WORKFLOW = {"nodes": [{"id": 1, "type": "CLIPTextEncode", "mode": 0,
                       "inputs": [], "outputs": [],
                       "widgets_values": ["VIDEO_marker 视频 a paper boat"]}],
            "links": []}
COMFY_TAGS = {"prompt": json.dumps(PROMPT), "workflow": json.dumps(WORKFLOW)}


def first_codec(*names):
    for name in names:
        if name in av.codecs_available:
            return name
    return None


def write_video(path, container, codec, metadata, options=None):
    out = av.open(str(path), mode="w", format=container, options=options or {})
    for key, value in metadata.items():
        out.metadata[key] = value
    stream = out.add_stream(codec, rate=Fraction(8, 1))
    stream.width = stream.height = 32
    stream.pix_fmt = "yuv420p"
    frame = av.VideoFrame(32, 32, "rgb24").reformat(format="yuv420p")
    for _ in range(3):
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    """{file name: what it stands for}; formats this PyAV build can't encode
    are left out."""
    d = tmp_path_factory.mktemp("videos")
    made = {}
    h264 = first_codec("libx264", "h264", "mpeg4")
    if h264:
        write_video(d / "comfy_faststart.mp4", "mp4", h264, COMFY_TAGS,
                    {"movflags": "use_metadata_tags+faststart"})
        write_video(d / "comfy_moov_at_end.mp4", "mp4", h264, COMFY_TAGS,
                    {"movflags": "use_metadata_tags"})
        # VideoHelperSuite: everything in the standard comment tag
        write_video(d / "vhs_comment.mp4", "mp4", h264,
                    {"comment": json.dumps({"prompt": PROMPT, "workflow": WORKFLOW})})
        write_video(d / "no_metadata.mp4", "mp4", h264, {})
        made.update({"comfy_faststart.mp4": "mp4", "comfy_moov_at_end.mp4": "mp4",
                     "vhs_comment.mp4": "mp4"})
    vp9 = first_codec("libvpx-vp9", "libvpx")
    if vp9:
        write_video(d / "comfy.webm", "webm", vp9, COMFY_TAGS)
        made["comfy.webm"] = "webm"
    if h264:
        write_video(d / "comfy.mkv", "matroska", h264, COMFY_TAGS)
        made["comfy.mkv"] = "mkv"
    if features.check("avif"):
        img = Image.new("RGB", (16, 16))
        exif = img.getexif()
        exif[0x0110] = "prompt:" + json.dumps(PROMPT)       # as ComfyUI's AVIF
        exif[0x010F] = "workflow:" + json.dumps(WORKFLOW)
        img.save(d / "comfy.avif", exif=exif)
        made["comfy.avif"] = "avif"
    if not made:
        pytest.skip("this PyAV build has no usable encoders")
    return d, made


def test_found_in_every_container(videos):
    folder, made = videos
    assert hit_names(run_search(folder, "VIDEO_marker")) == set(made)
    assert hit_names(run_search(folder, "视频")) == set(made)


def test_prompt_and_workflow_are_named_fields(videos):
    folder, made = videos
    containers = {name for name, kind in made.items() if kind != "avif"}
    assert hit_names(run_search(folder, "VIDEO_marker", "--fields", "prompt")) == containers
    assert hit_names(run_search(folder, "VIDEO_marker", "--fields", "workflow")) == containers


def test_graph_filters_work_on_videos(videos):
    folder, made = videos
    assert hit_names(run_search(folder, "VIDEO_stray")) == set(made)
    assert hit_names(run_search(folder, "VIDEO_stray", "--only-connected")) == set()
    assert hit_names(run_search(folder, "VIDEO_marker", "wan2.2", "--scope", "node")) == set()
    assert hit_names(run_search(folder, "wan2.2", "--node-type", "Checkpoint")) == set(made)


def test_show_on_a_video(videos):
    folder, made = videos
    report = run_search(folder, "paper boat", "--show")
    summaries = {Path(r["path"]).name: r["summary"] for r in report["results"]}
    assert set(summaries) == set(made)
    for summary in summaries.values():
        assert summary["positive"] == ["VIDEO_marker 视频 a paper boat"]
        assert summary["models"] == ["wan2.2_video.safetensors"]


def test_video_without_metadata_is_not_an_error(videos):
    folder, made = videos
    if "comfy_faststart.mp4" not in made:
        pytest.skip("no MP4 encoder")
    report = run_search(folder, "anything", "--ext", "mp4")
    assert report["scanned"] == 4 and report["unreadable"] == 0


def test_ext_filter_leaves_videos_out(videos):
    folder, _made = videos
    assert run_search(folder, "VIDEO_marker", "--ext", "png")["scanned"] == 0


def test_damaged_files_are_unreadable(tmp_path):
    (tmp_path / "bad.mp4").write_bytes(b"\x00\x00\x00\x10junk" + b"x" * 64)
    (tmp_path / "bad.webm").write_bytes(b"not matroska at all")
    report = run_search(tmp_path, "anything", allow_errors=True)
    assert report["unreadable"] == 2
