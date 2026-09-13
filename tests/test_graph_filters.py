"""--only-connected, --connected-mode and --scope node on the ComfyUI samples."""

from pathlib import Path

import pytest

import samples
from helpers import hit_names, run_search

EXPECTED, MULTI = samples.expectations()
MARKERS = sorted({m for marks in EXPECTED.values() for m in marks})

MODES = {"all": [], "linked": ["--only_connected"],
         "output": ["--connected-mode", "output"]}
# WebP keeps prompt/workflow in EXIF Model/Make
CHUNKS = {"any field": [], "prompt": ["--fields", "prompt", "Model"],
          "workflow": ["--fields", "workflow", "Make"]}


@pytest.mark.parametrize("marker", MARKERS)
def test_marker(sample_dir, marker):
    """Found exactly where expected in every mode: over all metadata, and in
    the prompt and workflow chunks separately, so a filter that works on one
    format can't hide a bug in the other."""
    problems = []
    for chunk, chunk_args in CHUNKS.items():
        present = hit_names(run_search(sample_dir, marker, *chunk_args))
        for i, (mode, mode_args) in enumerate(MODES.items()):
            want = {f for f, marks in EXPECTED.items()
                    if marks.get(marker, "nnn")[i] == "y"}
            if chunk != "any field":
                want &= present  # only files whose chunk holds the marker
            got = hit_names(run_search(sample_dir, marker, *chunk_args, *mode_args))
            if got != want:
                problems.append(f"[{chunk} / {mode}] missing {sorted(want - got)}, "
                                f"unexpected {sorted(got - want)}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", MULTI,
                         ids=lambda c: " ".join([*c["patterns"], *c["args"]]))
def test_multi_term(sample_dir, case):
    report = run_search(sample_dir, *case["patterns"], *case["args"])
    got = {Path(r["path"]).name: sorted(m["field"] for m in r["matches"])
           for r in report["results"]}
    assert set(got) == set(case["files"])
    for name, fields in (case["fields"] or {}).items():
        assert got[name] == sorted(fields)
