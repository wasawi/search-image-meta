"""--show: a summary of what produced each match (prompts, models, LoRAs, sampler)."""

import csv
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from helpers import run_cli, run_search


def summary_of(folder, name, *args):
    report = run_search(folder, *args, "--show")
    by_name = {Path(r["path"]).name: r for r in report["results"]}
    return by_name[name]["summary"]


def test_stock_graph(sample_dir):
    s = summary_of(sample_dir, "01_stray_loader.png", "MAIN_ckpt_s01")
    assert s["positive"] == ["MAIN_prompt_s01 a lighthouse at dusk"]
    assert s["negative"] == ["blurry, lowres"]
    assert s["models"] == ["MAIN_ckpt_s01.safetensors"]  # not the stray loader
    assert s["loras"] == []
    assert s["samplers"] == [{"seed": 42, "steps": 20, "cfg": 8,
                              "sampler_name": "euler", "scheduler": "normal",
                              "denoise": 1}]


def test_bypassed_lora_is_not_listed(sample_dir):
    s = summary_of(sample_dir, "03_bypassed.png", "MAIN_ckpt_s03")
    assert s["loras"] == []
    assert s["models"] == ["MAIN_ckpt_s03.safetensors"]


def test_prompt_inside_a_subgraph(sample_dir):
    s = summary_of(sample_dir, "07_subgraph.png", "MAIN_ckpt_s07")
    assert s["positive"] == ["SUBGRAPH_used_text, a lighthouse"]


def test_a1111_parameters(sample_dir):
    s = summary_of(sample_dir, "10_a1111.png", "A1111_model_marker")
    assert s["positive"] == ["a photo of a cat"]
    assert s["negative"] == ["blurry"]
    assert s["models"] == ["A1111_model_marker"]
    assert s["samplers"] == [{"steps": "20", "sampler_name": "Euler a"}]


def api(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs,
            "_meta": {"title": class_type}}


@pytest.fixture(scope="module")
def fox_dir(tmp_path_factory):
    """A Flux-style graph: text built from a primitive + concatenation, two
    LoRA loaders (one rgthree Power Lora with a disabled slot), a zeroed-out
    negative, and an unused loader."""
    d = tmp_path_factory.mktemp("show")
    prompt = {
        "1": api("UNETLoader", unet_name="flux1-dev.safetensors",
                 weight_dtype="default"),
        "2": api("DualCLIPLoader", clip_name1="t5xxl_fp16.safetensors",
                 clip_name2="clip_l.safetensors", type="flux"),
        "3": api("LoraLoader", lora_name="detail_tweaker.safetensors",
                 strength_model=0.6, strength_clip=1, model=["1", 0],
                 clip=["2", 0]),
        "4": api("Power Lora Loader (rgthree)",
                 lora_1={"on": True, "lora": "style_a.safetensors", "strength": 0.9},
                 lora_2={"on": False, "lora": "disabled_b.safetensors", "strength": 1},
                 model=["3", 0], clip=["3", 1]),
        "5": api("PrimitiveStringMultiline", value="a red fox in snow"),
        "6": api("StringConcatenate", string_a=["5", 0],
                 string_b="cinematic lighting", delimiter=", "),
        "7": api("CLIPTextEncode", text=["6", 0], clip=["4", 1]),
        "8": api("ConditioningZeroOut", conditioning=["7", 0]),
        "9": api("KSampler", seed=7, steps=28, cfg=1, sampler_name="euler",
                 scheduler="simple", denoise=1, model=["4", 0],
                 positive=["7", 0], negative=["8", 0], latent_image=["10", 0]),
        "10": api("EmptySD3LatentImage", width=1024, height=1024, batch_size=1),
        "11": api("VAELoader", vae_name="ae.safetensors"),
        "12": api("VAEDecode", samples=["9", 0], vae=["11", 0]),
        "13": api("SaveImage", filename_prefix="fox", images=["12", 0]),
        "14": api("CheckpointLoaderSimple", ckpt_name="unused_stray.safetensors"),
    }
    info = PngInfo()
    info.add_text("prompt", json.dumps(prompt))
    Image.new("RGB", (8, 8)).save(d / "fox.png", pnginfo=info)
    return d


def test_traced_prompt_models_and_loras(fox_dir):
    s = summary_of(fox_dir, "fox.png", "red fox")
    assert s["positive"] == ["a red fox in snow", "cinematic lighting"]
    assert s["negative"] == []  # a zeroed-out copy of the positive
    assert s["models"] == ["flux1-dev.safetensors", "t5xxl_fp16.safetensors",
                           "clip_l.safetensors", "ae.safetensors"]
    assert s["loras"] == ["detail_tweaker.safetensors (0.6)",
                          "style_a.safetensors (0.9)"]
    assert s["samplers"] == [{"seed": 7, "steps": 28, "cfg": 1,
                              "sampler_name": "euler", "scheduler": "simple",
                              "denoise": 1}]


def test_plain_output(fox_dir):
    code, out, err = run_cli(fox_dir, "red fox", "--show", "--progress", "never")
    assert code == 0, err
    lines = out.splitlines()
    assert Path(lines[0]).name == "fox.png"
    assert "    positive: a red fox in snow | cinematic lighting" in lines
    assert ("    loras:    detail_tweaker.safetensors (0.6), "
            "style_a.safetensors (0.9)") in lines
    assert ("    sampler:  seed 7, steps 28, cfg 1, sampler_name euler, "
            "scheduler simple, denoise 1") in lines
    assert not any(line.startswith("    negative") for line in lines)


def test_csv_columns(fox_dir):
    code, out, err = run_cli(fox_dir, "red fox", "--show", "--csv",
                             "--progress", "never")
    assert code == 0, err
    rows = list(csv.DictReader(io.StringIO(out)))
    assert list(rows[0]) == ["path", "field", "snippet", "link", "positive",
                             "negative", "models", "loras", "sampler"]
    assert rows[0]["positive"] == "a red fox in snow | cinematic lighting"
    assert rows[0]["models"].startswith("flux1-dev.safetensors, ")


def test_nothing_to_summarise(tmp_path):
    info = PngInfo()
    info.add_text("Description", "just a red fox photo")
    Image.new("RGB", (8, 8)).save(tmp_path / "photo.png", pnginfo=info)
    assert summary_of(tmp_path, "photo.png", "red fox") == {}
