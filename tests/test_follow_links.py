"""--follow-links: searching the images that symlinks and Finder aliases point to."""

import os
import shutil
import sqlite3

import pytest

from helpers import run_cli, run_search, ssim

ALIAS_BYTES = b"book\x00\x00\x00\x00mark\x00\x00\x00\x00" + bytes(range(256)) * 5


def make_finder_alias(target, alias):
    """Write a real alias file with CoreFoundation, as Finder would (macOS)."""
    import ctypes
    import ctypes.util

    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    ref = ctypes.c_void_p
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
        ref, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
    cf.CFURLCreateFromFileSystemRepresentation.restype = ref
    cf.CFURLCreateBookmarkData.argtypes = [ref, ref, ctypes.c_ulong, ref, ref, ref]
    cf.CFURLCreateBookmarkData.restype = ref
    cf.CFURLWriteBookmarkDataToFile.argtypes = [ref, ref, ctypes.c_ulong, ref]
    cf.CFURLWriteBookmarkDataToFile.restype = ctypes.c_bool
    cf.CFRelease.argtypes = [ref]

    def url(path):
        raw = os.fsencode(str(path))
        return cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)

    target_url, alias_url = url(target), url(alias)
    data = cf.CFURLCreateBookmarkData(None, target_url, 1 << 10, None, None, None)
    ok = bool(data) and cf.CFURLWriteBookmarkDataToFile(data, alias_url, 0, None)
    for handle in (data, target_url, alias_url):
        if handle:
            cf.CFRelease(handle)
    assert ok, "could not write the alias"


def can_symlink(tmp_path):
    try:
        os.symlink(tmp_path, tmp_path / "probe-link")
        return True
    except OSError:
        return False


needs_aliases = pytest.mark.skipif(not ssim.CAN_RESOLVE_ALIASES,
                                   reason="following Finder aliases needs macOS")


@pytest.fixture
def library(sample_dir, tmp_path):
    """library/real.png and library/other.png, and an empty library/links."""
    lib = (tmp_path / "library").resolve()
    (lib / "links").mkdir(parents=True)
    shutil.copy(sample_dir / "01_stray_loader.png", lib / "real.png")
    shutil.copy(sample_dir / "02_stray_chain.png", lib / "other.png")
    return lib


@pytest.fixture(params=["symlink", "Finder alias"])
def link_kind(request, tmp_path):
    if request.param == "symlink" and not can_symlink(tmp_path):
        pytest.skip("symlinks need Developer Mode or admin rights here")
    if request.param == "Finder alias" and not ssim.CAN_RESOLVE_ALIASES:
        pytest.skip("following Finder aliases needs macOS")
    return request.param


def make_link(kind, target, link):
    if kind == "symlink":
        os.symlink(target, link)
    else:
        make_finder_alias(target, link)


def test_links_are_skipped_without_the_option(library, link_kind):
    make_link(link_kind, library / "real.png", library / "links" / "real.png")
    report = run_search(library / "links", "lighthouse")
    assert report["results"] == []
    assert report["skipped_links"] == {link_kind: 1}


def test_followed_links_lead_to_the_original(library, link_kind):
    make_link(link_kind, library / "real.png", library / "links" / "real.png")
    report = run_search(library / "links", "lighthouse", "--follow-links")
    [result] = report["results"]
    assert result["path"] == str(library / "links" / "real.png")
    assert result["original"] == str(library / "real.png")
    assert report["skipped_links"] == {}


def test_each_image_counts_once(library, link_kind):
    make_link(link_kind, library / "real.png", library / "links" / "real.png")
    # serial, so the walk meets the originals (top folder) before the link
    report = run_search(library, "lighthouse", "-r", "--follow-links", "-j", "1")
    assert {r["path"] for r in report["results"]} == {
        str(library / "real.png"), str(library / "other.png")}
    assert report["duplicates"] == 1


def test_link_dir_points_at_the_original(library, link_kind, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("the new link is a symlink")
    make_link(link_kind, library / "real.png", library / "links" / "real.png")
    out = tmp_path / "out"
    report = run_search(library / "links", "lighthouse", "--follow-links",
                        "--link-dir", out, "--link-type", "symlink")
    [result] = report["results"]
    assert os.readlink(result["link"]) == str(library / "real.png")


def test_dead_links(library, link_kind):
    doomed = library / "doomed.png"
    shutil.copy(library / "real.png", doomed)
    make_link(link_kind, doomed, library / "links" / "doomed.png")
    doomed.unlink()

    assert run_search(library / "links", "lighthouse")["skipped_links"] == {link_kind: 1}
    followed = run_search(library / "links", "lighthouse", "--follow-links")
    assert followed["skipped_links"] == {f"dead {link_kind}": 1}
    assert followed["unreadable"] == 0

    broken = run_search(library / "links", "--broken", "--follow-links")
    assert [os.path.basename(r["path"]) for r in broken["results"]] == ["doomed.png"]
    assert broken["broken_reasons"] == {"link to a missing file": 1}
    assert run_search(library / "links", "--broken")["results"] == []


def test_index_remembers_the_original(library, link_kind, tmp_path):
    make_link(link_kind, library / "real.png", library / "links" / "real.png")
    db = tmp_path / "meta.sqlite"
    run_search(library / "links", "lighthouse", "--follow-links", "--index", db)
    con = sqlite3.connect(db)
    paths = [row[0] for row in con.execute("SELECT path FROM files")]
    con.close()
    assert paths == [str(library / "real.png")]


def test_plain_and_csv_output_name_the_original(library, link_kind):
    make_link(link_kind, library / "real.png", library / "links" / "real.png")
    code, out, err = run_cli(library / "links", "lighthouse", "--follow-links",
                             "-v", "--progress", "never")
    assert code == 0, err
    assert f"    (original) {library / 'real.png'}" in out.splitlines()
    code, out, err = run_cli(library / "links", "lighthouse", "--follow-links",
                             "--csv", "--progress", "never")
    header, row = out.splitlines()[:2]
    assert header == "path,field,snippet,link,original"
    assert row.endswith("," + str(library / "real.png"))


def test_aliases_stay_skipped_where_they_cant_be_followed(library, monkeypatch):
    (library / "links" / "real.png").write_bytes(ALIAS_BYTES)
    monkeypatch.setattr(ssim, "CAN_RESOLVE_ALIASES", False)
    report = run_search(library / "links", "lighthouse", "--follow-links")
    assert report["skipped_links"] == {"Finder alias": 1}


@needs_aliases
def test_an_unreadable_alias_is_a_dead_link_on_macos(library):
    (library / "links" / "real.png").write_bytes(ALIAS_BYTES)  # not a real bookmark
    report = run_search(library / "links", "lighthouse", "--follow-links")
    assert report["skipped_links"] == {"dead Finder alias": 1}
