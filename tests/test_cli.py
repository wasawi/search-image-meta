"""The script run as a real process: argument parsing, exit codes, process pool."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "search_string_image_meta.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8")


def test_help():
    res = run("--help")
    assert res.returncode == 0
    assert "--only-connected" in res.stdout


def test_process_pool_and_exit_codes(sample_dir):
    found = run(sample_dir, "MAIN_ckpt_s01", "--only-connected", "--pool",
                "process", "-j", "2", "--json", "--progress", "never")
    assert found.returncode == 0, found.stderr
    names = {Path(r["path"]).name for r in json.loads(found.stdout)["results"]}
    assert names == {"01_stray_loader.png"}

    # the stray loader is the only place this marker lives
    missing = run(sample_dir, "STRAY_ckpt_krea", "--only-connected", "--pool",
                  "process", "-j", "2", "--progress", "never")
    assert missing.returncode == 1, missing.stderr


def test_bad_regex_is_a_usage_error(sample_dir):
    res = run(sample_dir, "(", "--regex")
    assert res.returncode == 2
    assert "bad regex" in res.stderr
