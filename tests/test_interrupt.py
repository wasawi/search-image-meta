"""Ctrl-C must stop a search promptly, even while a read is stuck."""

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from helpers import ROOT

SCRIPT = ROOT / "search_string_image_meta.py"


@pytest.mark.skipif(os.name == "nt" or not hasattr(os, "mkfifo"),
                    reason="needs POSIX process groups and named pipes")
@pytest.mark.parametrize("pool", ["process", "thread"])
def test_ctrl_c_stops_a_search_stuck_on_a_read(sample_dir, tmp_path, pool):
    folder = tmp_path / "images"
    folder.mkdir()
    shutil.copy(sample_dir / "01_stray_loader.png", folder / "a.png")
    # opening a named pipe for reading blocks until someone writes to it,
    # just like a read that hangs on a slow or unresponsive disk
    os.mkfifo(folder / "stuck.png")

    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), str(folder), "lighthouse", "-r",
         "--pool", pool, "-j", "2", "--no-count", "--progress", "never",
         "--results", str(tmp_path / "results.txt")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        time.sleep(4)  # workers start and one blocks on the pipe
        assert proc.poll() is None, "the search should still be stuck"
        # a terminal delivers Ctrl-C to the whole process group
        os.killpg(proc.pid, signal.SIGINT)
        try:
            _out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("Ctrl-C did not stop the search within 15 s")
        assert proc.returncode == 130, err.decode("utf-8", "replace")
        assert "interrupted" in err.decode("utf-8", "replace")
        assert "INTERRUPTED" in (tmp_path / "results.txt").read_text(encoding="utf-8")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # never leave anything behind
        except (ProcessLookupError, PermissionError):
            pass
