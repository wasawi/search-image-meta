"""--index: a SQLite cache of what was read from each file."""

import json
import shutil
import sqlite3
import subprocess
import sys

import pytest

from helpers import ROOT, hit_names, run_cli, run_search, ssim

IMAGES = 11  # files in the sample folder


@pytest.fixture
def images(sample_dir, tmp_path):
    folder = tmp_path / "images"
    shutil.copytree(sample_dir, folder)
    return folder


@pytest.fixture
def db(tmp_path):
    return tmp_path / "meta.sqlite"


def stats(report):
    return report["index"]["cached"], report["index"]["read"]


@pytest.mark.parametrize("jobs", ["1", "4"])
def test_second_run_comes_from_the_cache(images, db, jobs):
    first = run_search(images, "lighthouse", "--index", db, "-j", jobs)
    assert stats(first) == (0, IMAGES)
    second = run_search(images, "lighthouse", "--index", db, "-j", jobs)
    assert stats(second) == (IMAGES, 0)
    assert hit_names(second) == hit_names(first)


def test_other_searches_never_open_the_files(images, db, monkeypatch):
    run_search(images, "lighthouse", "--index", db)

    def fail(*_args, **_kwargs):
        raise AssertionError("a cached file was read again")

    monkeypatch.setattr(ssim, "extract_metadata", fail)
    report = run_search(images, "STRAY_ckpt_krea", "--scope", "node", "--show",
                        "--index", db)
    assert hit_names(report) == {"01_stray_loader.png", "09_stray_loader.webp"}
    assert all(r["summary"]["models"] for r in report["results"])


def test_changed_files_are_read_again(images, db):
    run_search(images, "lighthouse", "--index", db)
    shutil.copy(images / "10_a1111.png", images / "02_stray_chain.png")
    report = run_search(images, "A1111_model_marker", "--index", db)
    assert stats(report) == (IMAGES - 1, 1)
    assert hit_names(report) == {"10_a1111.png", "02_stray_chain.png"}


def test_deep_searches_need_deep_reads(images, db):
    run_search(images, "lighthouse", "--index", db)
    assert stats(run_search(images, "lighthouse", "--index", db, "--deep")) == (0, IMAGES)
    # a deep read serves normal searches too
    assert stats(run_search(images, "lighthouse", "--index", db)) == (IMAGES, 0)


def test_new_version_rebuilds(images, db):
    run_search(images, "lighthouse", "--index", db)
    con = sqlite3.connect(db)
    with con:
        con.execute("UPDATE info SET value = 'old' WHERE key = 'version'")
    con.close()
    report = run_search(images, "lighthouse", "--index", db)
    assert stats(report) == (0, IMAGES)
    assert report["index"]["rebuilt"] is True


def test_unreadable_files_are_not_cached(images, db):
    (images / "broken.png").write_bytes(b"not an image at all")
    for _ in range(2):
        report = run_search(images, "lighthouse", "--index", db, allow_errors=True)
        assert report["unreadable"] == 1
    assert stats(report) == (IMAGES, 0)


def test_bad_index_path_is_a_usage_error(sample_dir, tmp_path):
    code, _out, err = run_cli(sample_dir, "x", "--index",
                              tmp_path / "missing" / "meta.sqlite")
    assert code == 2 and "cannot open index" in err


def test_process_pool_workers_use_the_cache(images, db):
    cmd = [sys.executable, str(ROOT / "search_string_image_meta.py"), str(images),
           "lighthouse", "--index", str(db), "--pool", "process", "-j", "2",
           "--json", "--progress", "never"]
    runs = [subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            for _ in range(2)]
    assert all(r.returncode == 0 for r in runs), runs[-1].stderr
    assert stats(json.loads(runs[1].stdout)) == (IMAGES, 0)
