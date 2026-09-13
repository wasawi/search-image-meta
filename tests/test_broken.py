"""--broken: listing unreadable files, with a hint of what they contain."""

import json
import shutil
from pathlib import Path

import pytest

from helpers import hit_names, run_cli, run_search

BROKEN = {
    "empty.png": "empty file",
    "zeros.png": "only zero bytes",
    "cut_short.png": "PNG, damaged or cut short",
    "error_page.png": "HTML or XML text",
    "record.png": "JSON text",
    "jpeg_start.png": "JPEG, damaged or cut short",
    "noise.png": "unrecognised content",
}


@pytest.fixture
def folder(sample_dir, tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    good = sample_dir / "01_stray_loader.png"
    shutil.copy(good, d / "good.png")
    (d / "empty.png").write_bytes(b"")
    (d / "zeros.png").write_bytes(bytes(2048))
    (d / "cut_short.png").write_bytes(good.read_bytes()[:20])
    (d / "error_page.png").write_bytes(b"<!DOCTYPE html><html><body>403 Forbidden</body></html>")
    (d / "record.png").write_bytes(json.dumps({"prompt": "not an image"}).encode())
    (d / "jpeg_start.png").write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x01\x02")
    (d / "noise.png").write_bytes(bytes([7, 200, 13, 99, 250, 1, 2, 3]) * 8)
    (d / "notes.txt").write_text("not an image type, never scanned")
    return d


def test_lists_only_unreadable_files(folder):
    report = run_search(folder, "--broken")
    assert hit_names(report) == set(BROKEN)
    assert report["matched"] == len(BROKEN) and report["scanned"] == len(BROKEN) + 1
    assert report["unreadable"] == 0  # they're the results, not failures


def test_each_file_says_what_it_holds(folder):
    report = run_search(folder, "--broken")
    reasons = {Path(r["path"]).name: r["matches"][0] for r in report["results"]}
    for name, category in BROKEN.items():
        assert reasons[name]["field"] == "error"
        assert category in reasons[name]["snippet"], (name, reasons[name]["snippet"])
    assert "starts 07 c8 0d 63" in reasons["noise.png"]["snippet"]
    assert "2.0 KB" in reasons["zeros.png"]["snippet"]
    assert str(folder) not in reasons["empty.png"]["snippet"]  # no repeated path


def test_summary_counts_each_kind(folder):
    code, out, err = run_cli(folder, "--broken", "-v", "--progress", "never")
    assert code == 0, err
    assert f"{len(BROKEN)} broken file(s) in {len(BROKEN) + 1} image(s) scanned" in err
    assert "1 × empty file" in err
    lines = out.splitlines()
    assert any(line.startswith("    (error) ") and "empty file" in line for line in lines)


def test_json_counts_reasons(folder):
    report = run_search(folder, "--broken")
    assert report["broken_reasons"]["empty file"] == 1
    assert sum(report["broken_reasons"].values()) == len(BROKEN)


def test_links_to_broken_files(folder, tmp_path):
    out = tmp_path / "broken"
    report = run_search(folder, "--broken", "--link-dir", out, "--link-type", "copy")
    assert {Path(r["link"]).name for r in report["results"]} == set(BROKEN)
    assert sorted(p.name for p in out.iterdir()) == sorted(BROKEN)


def test_normal_searches_explain_errors_too(folder):
    code, _out, err = run_cli(folder, "lighthouse", "--errors", "--progress", "never")
    assert code == 0
    assert "[skip]" in err and "empty file" in err and "HTML or XML text" in err


def test_usage(folder):
    code, _out, err = run_cli(folder, "lighthouse", "--broken")
    assert code == 2 and "takes no search patterns" in err
    code, _out, err = run_cli(folder, "--progress", "never")
    assert code == 2 and "at least one pattern" in err


def test_results_file_resumes_broken_listings(folder, tmp_path):
    log = tmp_path / "results.txt"
    assert run_cli(folder, "--broken", "--results", log, "--progress", "never")[0] == 0
    code, _out, err = run_cli(folder, "--broken", "--results", log, "--progress", "never")
    assert code == 0 and "1 folder(s) already done" in err
