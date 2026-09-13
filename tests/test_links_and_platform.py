"""--link-dir link types and dates, hidden files, and platform-specific paths."""

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from helpers import ROOT, hit_names, run_cli, run_search, ssim

MTIME = 1_700_000_000        # 2023-11-14
BIRTH = 1_600_000_000        # 2020-09-13


def birth_of(path, follow=True):
    st = os.stat(path, follow_symlinks=follow)
    if hasattr(st, "st_birthtime"):
        return st.st_birthtime
    return st.st_ctime if os.name == "nt" else None


@pytest.fixture
def source(sample_dir, tmp_path):
    """src/sub/match.png with known modification (and creation) dates."""
    folder = tmp_path / "src" / "sub"
    folder.mkdir(parents=True)
    image = folder / "match.png"
    shutil.copy(sample_dir / "01_stray_loader.png", image)
    os.utime(image, (MTIME, MTIME))
    if ssim.CAN_SET_CREATION_TIME:
        assert ssim.set_creation_time(image, BIRTH * 1_000_000_000)
    return tmp_path / "src"


def link_one(folder, dest, *args):
    report = run_search(folder, "lighthouse", "-r", "--link-dir", dest, *args)
    [result] = report["results"]
    assert result["link"], "no link was made"
    return Path(result["path"]), Path(result["link"])


def can_symlink(tmp_path):
    try:
        os.symlink(tmp_path, tmp_path / "probe-link")
        return True
    except OSError:
        return False


def test_copy_keeps_both_dates(source, tmp_path):
    src, link = link_one(source, tmp_path / "out", "--link-type", "copy")
    assert link.read_bytes() == src.read_bytes()
    assert int(link.stat().st_mtime) == MTIME
    if ssim.CAN_SET_CREATION_TIME:
        assert abs(birth_of(link) - BIRTH) < 2


def test_hardlink(source, tmp_path):
    src, link = link_one(source, tmp_path / "out", "--link-type", "hardlink")
    assert os.path.samefile(src, link)


def test_symlink(source, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks need Developer Mode or admin rights here")
    src, link = link_one(source, tmp_path / "out", "--link-type", "symlink")
    assert link.is_symlink() and os.path.samefile(src, link)
    assert int(os.lstat(link).st_mtime) == MTIME
    if ssim.CAN_SET_CREATION_TIME:
        assert abs(birth_of(link, follow=False) - BIRTH) < 2


def test_auto_always_produces_a_link(source, tmp_path):
    src, link = link_one(source, tmp_path / "out")
    assert os.path.samefile(src, link) or link.read_bytes() == src.read_bytes()
    if os.name != "nt":
        assert link.is_symlink()


def test_auto_falls_back_when_symlinks_are_refused(source, tmp_path, monkeypatch):
    def refuse(self, *_args, **_kwargs):
        raise OSError("A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", refuse)
    monkeypatch.setattr(ssim, "AUTO_LINK_TYPES", ["symlink", "hardlink", "copy"])
    src, link = link_one(source, tmp_path / "out", "--no-preserve-dates")
    assert not link.is_symlink() and os.path.samefile(src, link)


def test_flatten_and_name_clashes(source, sample_dir, tmp_path):
    other = source / "other"
    other.mkdir()
    shutil.copy(sample_dir / "01_stray_loader.png", other / "match.png")
    report = run_search(source, "lighthouse", "-r", "--link-dir", tmp_path / "a",
                        "--link-type", "copy")
    assert sorted(Path(r["link"]).name for r in report["results"]) == [
        "match.png", "match_1.png"]
    report = run_search(source, "lighthouse", "-r", "--link-dir", tmp_path / "b",
                        "--link-type", "copy", "--flatten", "path")
    assert sorted(Path(r["link"]).name for r in report["results"]) == [
        "other__match.png", "sub__match.png"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcuts")
def test_windows_shortcut(source, tmp_path):
    _src, link = link_one(source, tmp_path / "out", "--link-type", "alias")
    assert link.suffix == ".lnk" and link.stat().st_size > 0


@pytest.mark.skipif(sys.platform in ("darwin", "win32"),
                    reason="aliases exist on this system")
def test_alias_elsewhere_is_a_usage_error(source, tmp_path):
    code, _out, err = run_cli(source, "lighthouse", "--link-dir", tmp_path / "out",
                              "--link-type", "alias")
    assert code == 2 and "symlink" in err


@pytest.mark.skipif(sys.platform != "darwin" or os.environ.get("TEST_FINDER_ALIAS") != "1",
                    reason="drives Finder; run with TEST_FINDER_ALIAS=1 on macOS")
def test_finder_alias_with_quotes_in_the_path(sample_dir, tmp_path):
    folder = tmp_path / 'say "hi" \\ there'
    folder.mkdir()
    shutil.copy(sample_dir / "01_stray_loader.png", folder / "match.png")
    _src, link = link_one(folder, tmp_path / "out", "--link-type", "alias")
    assert link.exists()


def test_hidden_names_are_skipped(sample_dir, tmp_path):
    image = sample_dir / "01_stray_loader.png"
    (tmp_path / ".cache").mkdir()
    shutil.copy(image, tmp_path / ".cache" / "a.png")
    shutil.copy(image, tmp_path / ".b.png")
    shutil.copy(image, tmp_path / "c.png")
    assert hit_names(run_search(tmp_path, "lighthouse", "-r")) == {"c.png"}
    assert hit_names(run_search(tmp_path, "lighthouse", "-r", "--hidden")) == {
        "a.png", ".b.png", "c.png"}


@pytest.mark.skipif(os.name != "nt", reason="the hidden attribute is Windows-only")
def test_windows_hidden_attribute(sample_dir, tmp_path):
    import ctypes

    image = sample_dir / "01_stray_loader.png"
    shutil.copy(image, tmp_path / "secret.png")
    shutil.copy(image, tmp_path / "plain.png")
    assert ctypes.windll.kernel32.SetFileAttributesW(str(tmp_path / "secret.png"), 0x2)
    assert hit_names(run_search(tmp_path, "lighthouse")) == {"plain.png"}
    assert hit_names(run_search(tmp_path, "lighthouse", "--hidden")) == {
        "plain.png", "secret.png"}


def test_progress_bar_needs_a_real_terminal():
    assert ssim._ansi_terminal(io.StringIO()) is False


def test_piped_output_is_utf8_whatever_the_locale(tmp_path):
    info = PngInfo()
    info.add_text("prompt", json.dumps({"1": {"class_type": "CLIPTextEncode",
                                              "inputs": {"text": "女孩 on a beach"}}}))
    Image.new("RGB", (8, 8)).save(tmp_path / "女孩.png", pnginfo=info)
    env = {**os.environ, "PYTHONIOENCODING": "ascii"}
    res = subprocess.run([sys.executable, str(ROOT / "search_string_image_meta.py"),
                          str(tmp_path), "女孩", "-v", "--progress", "never"],
                         capture_output=True, env=env)
    assert res.returncode == 0, res.stderr.decode("utf-8", "replace")
    out = res.stdout.decode("utf-8")
    assert "女孩.png" in out and "女孩 on a beach" in out
