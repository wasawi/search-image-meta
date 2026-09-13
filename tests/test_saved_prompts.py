"""--saved-prompts and --show with prompts that a save node stored itself.

ComfyUI's own metadata only holds a graph's inputs, so a prompt written by an
LLM or a run-time wildcard never reaches it. Save nodes such as MetaWriter
take the final text on input pins and store it in a JSON record (`gen_meta`);
others write A1111-style `parameters`.
"""

import csv
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from helpers import hit_names, run_cli, run_search


def api(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs,
            "_meta": {"title": class_type}}


# the graph only knows the instruction given to the LLM node
GRAPH = {
    "1": api("CheckpointLoaderSimple", ckpt_name="saved_ckpt.safetensors"),
    "2": api("TextGenerate", prompt="INSTRUCTION_write a prompt about lighthouses",
             clip=["1", 1], seed=3),
    "3": api("CLIPTextEncode", text=["2", 0], clip=["1", 1]),
    "5": api("CLIPTextEncode", text="graph negative text", clip=["1", 1]),
    "4": api("KSampler", seed=11, steps=9, cfg=2, sampler_name="euler",
             scheduler="simple", denoise=1, model=["1", 0], positive=["3", 0],
             negative=["5", 0]),
    "6": api("VAEDecode", samples=["4", 0], vae=["1", 2]),
    "7": api("MetaWriter", images=["6", 0], filename_prefix="out", filename="x",
             prompt=["2", 0], negative_prompt="NEG_saved blurry"),
    # wired to the checkpoint but feeding nothing
    "8": api("LoraLoader", lora_name="leftover_lora.safetensors", strength_model=1,
             strength_clip=1, model=["1", 0], clip=["1", 1]),
}


def png(path, **chunks):
    info = PngInfo()
    for key, value in chunks.items():
        info.add_text(key, value)
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)


def jpeg_with_user_comment(path, prefix, record):
    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif.get_ifd(0x8769)[0x9286] = prefix + json.dumps(record, ensure_ascii=False).encode("utf-8")
    img.save(path, format="JPEG", exif=exif)


@pytest.fixture(scope="module")
def saved_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("saved")
    meta = {"id": "abc", "prompt": "FINAL_llm_prompt a lighthouse at dawn",
            "negative": "NEG_saved blurry", "intermediate_prompt_1": "DRAFT_one idea",
            "intermediate_prompt_2": "DRAFT_two 草稿"}
    png(d / "metawriter.png", gen_meta=json.dumps(meta, ensure_ascii=False),
        prompt=json.dumps(GRAPH), workflow=json.dumps({"nodes": [], "links": []}))
    png(d / "legacy_request.png", pw_json=json.dumps(
        {"request": {"prompt": "LEGACY_request_prompt", "negativePrompt": "LEGACY_negative"}}))
    png(d / "civitai_raw.png", pw_json=json.dumps(
        {"raw": {"meta": {"prompt": "RAW_meta_prompt", "negativePrompt": "RAW_negative"}}}))
    png(d / "typo.png", gen_meta=json.dumps({"id": "t", "promt": "TYPO_prompt"}))
    png(d / "saver_parameters.png",
        parameters="PARAMS_positive a lighthouse\nNegative prompt: PARAMS_negative\n"
                   "Steps: 9, Sampler: euler, Model: params_model",
        prompt=json.dumps(GRAPH))
    png(d / "graph_only.png", prompt=json.dumps(GRAPH))
    jpeg_with_user_comment(d / "usercomment_ascii.jpg", b"ASCII\x00\x00\x00",
                           {"id": "j1", "prompt": "女孩 JPEG_saved_prompt"})
    jpeg_with_user_comment(d / "usercomment_unicode.jpg", b"UNICODE\x00",
                           {"id": "j2", "prompt": "女孩 JPEG_saved_prompt"})
    return d


def fields_by_file(report):
    return {Path(r["path"]).name: sorted(m["field"] for m in r["matches"])
            for r in report["results"]}


# searching --------------------------------------------------------------------

def test_only_the_saved_prompts_are_searched(saved_dir):
    assert hit_names(run_search(saved_dir, "INSTRUCTION")) == {
        "metawriter.png", "saver_parameters.png", "graph_only.png"}
    assert hit_names(run_search(saved_dir, "INSTRUCTION", "--saved-prompts")) == set()
    assert hit_names(run_search(saved_dir, "FINAL_llm_prompt", "--saved-prompts")) == {
        "metawriter.png"}


def test_older_record_shapes(saved_dir):
    for term, name in (("LEGACY_request_prompt", "legacy_request.png"),
                       ("RAW_meta_prompt", "civitai_raw.png"),
                       ("TYPO_prompt", "typo.png")):
        assert hit_names(run_search(saved_dir, term, "--saved-prompts")) == {name}


def test_a1111_style_parameters(saved_dir):
    assert hit_names(run_search(saved_dir, "PARAMS_positive", "--saved-prompts",
                                "prompt")) == {"saver_parameters.png"}
    assert hit_names(run_search(saved_dir, "PARAMS_negative", "--saved-prompts",
                                "prompt")) == set()
    assert hit_names(run_search(saved_dir, "PARAMS_negative", "--saved-prompts",
                                "negative")) == {"saver_parameters.png"}
    # the settings line isn't a prompt
    assert hit_names(run_search(saved_dir, "params_model", "--saved-prompts")) == set()


def test_choosing_which_prompts(saved_dir):
    assert hit_names(run_search(saved_dir, "NEG_saved", "--saved-prompts", "prompt")) == set()
    assert hit_names(run_search(saved_dir, "NEG_saved", "--saved-prompts",
                                "negative")) == {"metawriter.png"}
    assert hit_names(run_search(saved_dir, "DRAFT_two", "--saved-prompts",
                                "intermediate")) == {"metawriter.png"}
    assert hit_names(run_search(saved_dir, "DRAFT_two", "--saved-prompts",
                                "prompt", "negative")) == set()


def test_choosing_single_intermediate_prompts(saved_dir):
    # DRAFT_one is intermediate_prompt_1, DRAFT_two is intermediate_prompt_2
    assert hit_names(run_search(saved_dir, "DRAFT_one", "--saved-prompts",
                                "intermediate1")) == {"metawriter.png"}
    assert hit_names(run_search(saved_dir, "DRAFT_one", "--saved-prompts",
                                "intermediate2")) == set()
    assert hit_names(run_search(saved_dir, "DRAFT_one", "--saved-prompts",
                                "intermediate2", "intermediate3")) == set()
    report = run_search(saved_dir, "DRAFT", "--saved-prompts", "intermediate1",
                        "intermediate2", "--scope", "field")
    assert fields_by_file(report) == {"metawriter.png": [
        "gen_meta:intermediate_prompt_1", "gen_meta:intermediate_prompt_2"]}
    report = run_search(saved_dir, "DRAFT", "--saved-prompts", "prompt",
                        "intermediate2", "--scope", "field")
    assert fields_by_file(report) == {"metawriter.png": ["gen_meta:intermediate_prompt_2"]}


def test_long_field_names_are_accepted(saved_dir):
    assert hit_names(run_search(saved_dir, "DRAFT_two", "--saved-prompts",
                                "intermediate_prompt_2")) == {"metawriter.png"}
    assert hit_names(run_search(saved_dir, "NEG_saved", "--saved-prompts",
                                "negative_prompt")) == {"metawriter.png"}
    assert hit_names(run_search(saved_dir, "FINAL_llm_prompt", "--saved-prompts",
                                "Positive")) == {"metawriter.png"}


def test_unknown_saved_prompt_is_a_usage_error(saved_dir):
    code, _out, err = run_cli(saved_dir, "x", "--saved-prompts", "intermediate_x")
    assert code == 2 and "isn't a saved prompt" in err


def test_labels_name_the_record_and_pin(saved_dir):
    report = run_search(saved_dir, "DRAFT", "--saved-prompts", "--scope", "field")
    assert fields_by_file(report) == {"metawriter.png": [
        "gen_meta:intermediate_prompt_1", "gen_meta:intermediate_prompt_2"]}


def test_scope_means_the_same_prompt(saved_dir):
    both = ["FINAL_llm_prompt", "DRAFT_one"]
    assert hit_names(run_search(saved_dir, *both, "--saved-prompts")) == {"metawriter.png"}
    assert hit_names(run_search(saved_dir, *both, "--saved-prompts", "--scope",
                                "field")) == set()


def test_not_and_non_ascii(saved_dir):
    assert hit_names(run_search(saved_dir, "lighthouse", "--saved-prompts")) == {
        "metawriter.png", "saver_parameters.png"}
    assert hit_names(run_search(saved_dir, "lighthouse", "--saved-prompts",
                                "--not", "DRAFT_one")) == {"saver_parameters.png"}
    assert hit_names(run_search(saved_dir, "草稿", "--saved-prompts")) == {"metawriter.png"}


def test_jpeg_user_comment_records(saved_dir):
    expected = {"usercomment_ascii.jpg", "usercomment_unicode.jpg"}
    assert hit_names(run_search(saved_dir, "女孩", "--saved-prompts")) == expected
    assert hit_names(run_search(saved_dir, "女孩")) == expected


def test_field_names_with_colons(saved_dir):
    # EXIF fields are named like "Exif:UserComment"; the pin comes after
    report = run_search(saved_dir, "JPEG_saved_prompt", "--saved-prompts",
                        "--fields", "Exif:UserComment")
    assert fields_by_file(report) == {
        "usercomment_ascii.jpg": ["Exif:UserComment:prompt"],
        "usercomment_unicode.jpg": ["Exif:UserComment:prompt"]}
    s = summaries(saved_dir, "JPEG_saved_prompt")["usercomment_ascii.jpg"]
    assert s["prompt_source"] == "Exif:UserComment"


def test_graph_filters_cant_be_combined(saved_dir):
    code, _out, err = run_cli(saved_dir, "x", "--saved-prompts", "--node-type", "Lora")
    assert code == 2 and "--saved-prompts" in err


# --show -----------------------------------------------------------------------

def summaries(folder, *args):
    report = run_search(folder, *args, "--show")
    return {Path(r["path"]).name: r["summary"] for r in report["results"]}


def test_show_prefers_the_saved_prompt(saved_dir):
    s = summaries(saved_dir, "FINAL_llm_prompt")["metawriter.png"]
    assert s["positive"] == ["FINAL_llm_prompt a lighthouse at dawn"]
    assert s["negative"] == ["NEG_saved blurry"]
    assert s["intermediate"] == ["DRAFT_one idea", "DRAFT_two 草稿"]
    assert s["prompt_source"] == "gen_meta"
    # everything else still comes from the graph
    assert s["models"] == ["saved_ckpt.safetensors"]
    assert s["samplers"] == [{"seed": 11, "steps": 9, "cfg": 2, "sampler_name": "euler",
                              "scheduler": "simple", "denoise": 1}]


def test_show_uses_saver_parameters_over_the_graph(saved_dir):
    s = summaries(saved_dir, "PARAMS_positive")["saver_parameters.png"]
    assert s["positive"] == ["PARAMS_positive a lighthouse"]
    assert s["negative"] == ["PARAMS_negative"]
    assert s["prompt_source"] == "parameters"
    assert s["models"] == ["saved_ckpt.safetensors"]


def test_show_without_a_saver_traces_the_graph(saved_dir):
    s = summaries(saved_dir, "INSTRUCTION")["graph_only.png"]
    assert s["positive"] == ["INSTRUCTION_write a prompt about lighthouses"]
    assert s["prompt_source"] == "graph"
    assert "intermediate" not in s


def test_show_plain_output_lines_up(saved_dir):
    code, out, err = run_cli(saved_dir, "FINAL_llm_prompt", "--show", "--progress", "never")
    assert code == 0, err
    lines = out.splitlines()
    assert "    positive:     FINAL_llm_prompt a lighthouse at dawn" in lines
    assert "    intermediate: DRAFT_one idea | DRAFT_two 草稿" in lines
    assert "    models:       saved_ckpt.safetensors" in lines


def test_show_csv_has_an_intermediate_column(saved_dir):
    code, out, err = run_cli(saved_dir, "FINAL_llm_prompt", "--show", "--csv",
                             "--progress", "never")
    assert code == 0, err
    rows = list(csv.DictReader(io.StringIO(out)))
    assert rows[0]["intermediate"] == "DRAFT_one idea | DRAFT_two 草稿"
    assert rows[0]["positive"] == "FINAL_llm_prompt a lighthouse at dawn"


def test_writer_nodes_count_as_outputs(saved_dir):
    # MetaWriter is the graph's only save node. Recognised as an output, the
    # wired-up leftover LoRA loader drops out; unrecognised, the summary would
    # fall back to every linked node and list it.
    s = summaries(saved_dir, "INSTRUCTION")["graph_only.png"]
    assert s["loras"] == []
    assert hit_names(run_search(saved_dir, "leftover_lora", "--connected-mode",
                                "output")) == set()
    assert hit_names(run_search(saved_dir, "leftover_lora", "--only-connected")) == {
        "metawriter.png", "saver_parameters.png", "graph_only.png"}
