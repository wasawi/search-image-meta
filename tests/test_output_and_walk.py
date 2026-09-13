"""--csv, -0, --exclude-dir, --since / --until."""

import csv
import io
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pytest

from helpers import hit_names, run_cli, run_search


# output formats ------------------------------------------------------------

def test_csv_rows(sample_dir):
    # --scope field lists every matching field (image scope stops at the first)
    code, out, err = run_cli(sample_dir, "MAIN_ckpt_s01", "--csv", "--scope",
                             "field", "--progress", "never")
    assert code == 0, err
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == ["path", "field", "snippet", "link"]
    assert {Path(r[0]).name for r in rows[1:]} == {"01_stray_loader.png"}
    assert {r[1] for r in rows[1:]} == {"prompt", "workflow"}
    assert all("MAIN_ckpt_s01" in r[2] for r in rows[1:])


def test_csv_with_only_a_header_when_nothing_matches(sample_dir):
    code, out, _err = run_cli(sample_dir, "no_such_marker", "--csv",
                              "--progress", "never")
    assert code == 1
    assert list(csv.reader(io.StringIO(out))) == [["path", "field", "snippet", "link"]]


def test_null_separated_paths(sample_dir):
    code, out, err = run_cli(sample_dir, "STRAY_ckpt_krea", "-0",
                             "--progress", "never")
    assert code == 0, err
    assert "\n" not in out and out.endswith("\0")
    assert {Path(p).name for p in out.split("\0") if p} == {
        "01_stray_loader.png", "09_stray_loader.webp"}


def test_output_formats_are_exclusive(sample_dir):
    code, _out, err = run_cli(sample_dir, "x", "--csv", "--json")
    assert code == 2 and "not allowed" in err


# walking -------------------------------------------------------------------

@pytest.fixture
def tree(sample_dir, tmp_path):
    """root/keep/a.png, root/old_renders/b.png, root/Old_Stuff/c.png"""
    source = sample_dir / "01_stray_loader.png"
    for folder, name in (("keep", "a.png"), ("old_renders", "b.png"),
                         ("Old_Stuff", "c.png")):
        (tmp_path / folder).mkdir()
        shutil.copy(source, tmp_path / folder / name)
    return tmp_path


def test_exclude_dir_wildcards_any_case(tree):
    assert hit_names(run_search(tree, "lighthouse", "-r")) == {"a.png", "b.png", "c.png"}
    assert hit_names(run_search(tree, "lighthouse", "-r",
                                "--exclude-dir", "OLD_*")) == {"a.png"}
    assert hit_names(run_search(tree, "lighthouse", "-r",
                                "--exclude-dir", "old_renders")) == {"a.png", "c.png"}


def _set_mtime(path, when):
    stamp = when.timestamp() if isinstance(when, datetime) else when
    os.utime(path, (stamp, stamp))


@pytest.fixture
def dated(sample_dir, tmp_path):
    source = sample_dir / "01_stray_loader.png"
    stamps = {"jan.png": datetime(2026, 1, 10, 15, 0),
              "mar.png": datetime(2026, 3, 5, 9, 30),
              "recent.png": time.time() - 3600}
    for name, when in stamps.items():
        shutil.copy(source, tmp_path / name)
        _set_mtime(tmp_path / name, when)
    return tmp_path


@pytest.mark.parametrize("args, expected", [
    (["--since", "2026-02-01"], {"mar.png", "recent.png"}),
    (["--until", "2026-01-10"], {"jan.png"}),                 # the whole day
    (["--until", "2026-01-10 12:00"], set()),
    (["--since", "2026-01-01", "--until", "2026-03-31"], {"jan.png", "mar.png"}),
    (["--since", "2h"], {"recent.png"}),
    (["--until", "2h"], {"jan.png", "mar.png"}),
    (["--since", "1w", "--until", "30m"], {"recent.png"}),
])
def test_modification_date_filters(dated, args, expected):
    assert hit_names(run_search(dated, "lighthouse", *args)) == expected


def test_bad_date_is_a_usage_error(dated):
    code, _out, err = run_cli(dated, "lighthouse", "--since", "last tuesday")
    assert code == 2 and "not a date" in err
