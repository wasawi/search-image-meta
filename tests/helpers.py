"""Run the search in-process, the way the command line would, and read its report."""

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import search_string_image_meta as ssim  # noqa: E402


def run_cli(*argv):
    """(exit code, stdout, stderr) of one in-process run."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = ssim.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def run_search(folder, *args, allow_errors=False):
    """Search `folder` with --json; returns the parsed report."""
    code, out, err = run_cli(folder, *args, "--json", "--progress", "never")
    assert code in (0, 1), f"exit {code}: {err}"
    report = json.loads(out)
    if not allow_errors:
        assert not report["unreadable"], err
    return report


def hit_names(report):
    return {Path(r["path"]).name for r in report["results"]}
