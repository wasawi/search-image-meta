"""--node-type and --input: searching inside particular nodes and inputs."""

import json
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from helpers import hit_names, run_search

LIGHTHOUSE = {"01_stray_loader.png", "02_stray_chain.png", "03_bypassed.png",
              "04_muted.png", "05_reroute.png", "06_set_get.png",
              "07_subgraph.png", "09_stray_loader.webp", "11_node_scope.png"}


def fields_by_file(report):
    return {Path(r["path"]).name: sorted(m["field"] for m in r["matches"])
            for r in report["results"]}


# --node-type ---------------------------------------------------------------

def test_node_type_selects_nodes(sample_dir):
    assert hit_names(run_search(sample_dir, "MAIN_ckpt_s01", "--node-type",
                                "checkpoint")) == {"01_stray_loader.png"}
    assert hit_names(run_search(sample_dir, "MAIN_ckpt_s01", "--node-type",
                                "KSampler")) == set()


def test_node_type_any_of_several(sample_dir):
    assert hit_names(run_search(sample_dir, "lighthouse", "--node-type",
                                "KSampler", "CLIPTextEncode")) == LIGHTHOUSE


def test_node_type_skips_non_graph_metadata(sample_dir):
    assert hit_names(run_search(sample_dir, "Sampler: Euler")) == {"10_a1111.png"}
    assert hit_names(run_search(sample_dir, "Sampler: Euler", "--node-type",
                                "KSampler")) == set()


def test_node_type_with_only_connected(sample_dir):
    stray = {"01_stray_loader.png", "09_stray_loader.webp"}
    assert hit_names(run_search(sample_dir, "STRAY_ckpt_krea", "--node-type",
                                "Checkpoint")) == stray
    assert hit_names(run_search(sample_dir, "STRAY_ckpt_krea", "--node-type",
                                "Checkpoint", "--only-connected")) == set()


# --input -------------------------------------------------------------------

def test_input_selects_values(sample_dir):
    assert hit_names(run_search(sample_dir, "STRAY_ckpt_krea", "--input",
                                "CKPT_NAME")) == {"01_stray_loader.png",
                                                  "09_stray_loader.webp"}
    assert hit_names(run_search(sample_dir, "lighthouse", "--input",
                                "ckpt_name")) == set()
    assert hit_names(run_search(sample_dir, "lighthouse", "--input",
                                "text")) == LIGHTHOUSE


def test_input_and_scope(sample_dir):
    same_node = ["MAIN_prompt_s11", "lighthouse"]
    across = ["MAIN_prompt_s11", "MAIN_ckpt_s11"]
    assert hit_names(run_search(sample_dir, *same_node, "--input", "text",
                                "--scope", "node")) == {"11_node_scope.png"}
    assert hit_names(run_search(sample_dir, *across, "--input", "text",
                                "ckpt_name")) == {"11_node_scope.png"}
    assert hit_names(run_search(sample_dir, *across, "--input", "text",
                                "ckpt_name", "--scope", "node")) == set()
    # with node filters every unit is one node, so "field" behaves like "node"
    assert hit_names(run_search(sample_dir, *across, "--input", "text",
                                "ckpt_name", "--scope", "field")) == set()


def test_input_labels_name_the_node(sample_dir):
    report = run_search(sample_dir, "SUBGRAPH_used_text", "--input", "text",
                        "--fields", "prompt")
    assert fields_by_file(report) == {"07_subgraph.png": ["prompt#6:1"]}


def test_not_looks_at_the_same_nodes(sample_dir):
    # the checkpoint name isn't in any text node, so nothing is excluded
    assert hit_names(run_search(sample_dir, "lighthouse", "--node-type",
                                "CLIPTextEncode", "--not", "MAIN_ckpt_s02")) == LIGHTHOUSE
    assert hit_names(run_search(sample_dir, "lighthouse", "--not",
                                "MAIN_ckpt_s02")) == LIGHTHOUSE - {"02_stray_chain.png"}


# workflows with named widget values -----------------------------------------

@pytest.fixture(scope="module")
def named_dir(tmp_path_factory):
    """Workflow-only images (no prompt chunk): named widgets, VHS dict values,
    and an old-style positional node."""
    d = tmp_path_factory.mktemp("named")
    workflow = {
        "nodes": [
            {"id": 1, "type": "LoraLoaderModelOnly", "mode": 0,
             "inputs": [{"name": "model", "type": "MODEL", "link": None}],
             "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}],
             "widgets_values": ["NAMED_lora.safetensors", 0.8],
             "widgets_values_named": {"lora_name": "NAMED_lora.safetensors",
                                      "strength_model": 0.8}},
            {"id": 2, "type": "VHS_VideoCombine", "mode": 0,
             "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
             "outputs": [],
             "widgets_values": {"filename_prefix": "VHS_prefix_marker",
                                "frame_rate": 16}},
            {"id": 3, "type": "CheckpointLoaderSimple", "mode": 0,
             "inputs": [], "outputs": [],
             "widgets_values": ["POSITIONAL_ckpt.safetensors"]},
        ],
        "links": [[1, 1, 0, 2, 0, "MODEL"]],
    }
    info = PngInfo()
    info.add_text("workflow", json.dumps(workflow))
    Image.new("RGB", (8, 8)).save(d / "named.png", pnginfo=info)
    return d


def test_workflow_named_widgets(named_dir):
    report = run_search(named_dir, "NAMED_lora", "--input", "lora_name")
    assert fields_by_file(report) == {"named.png": ["workflow#1"]}
    assert hit_names(run_search(named_dir, "0.8", "--input", "lora_name")) == set()


def test_workflow_dict_widgets(named_dir):
    report = run_search(named_dir, "VHS_prefix", "--input", "filename_prefix",
                        "--node-type", "VideoCombine", "--only-connected")
    assert fields_by_file(report) == {"named.png": ["workflow#2"]}


def test_positional_widgets_have_no_names(named_dir):
    assert hit_names(run_search(named_dir, "POSITIONAL_ckpt")) == {"named.png"}
    assert hit_names(run_search(named_dir, "POSITIONAL_ckpt", "--input",
                                "ckpt_name")) == set()
    assert hit_names(run_search(named_dir, "POSITIONAL_ckpt", "--node-type",
                                "Checkpoint")) == {"named.png"}
