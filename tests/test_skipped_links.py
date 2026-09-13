"""Links inside a searched folder: Finder aliases and symlinks aren't images."""

import os
import shutil

import pytest

from helpers import hit_names, run_cli, run_search, ssim

# first bytes of a real Finder alias (a bookmark file), followed by filler
ALIAS_BYTES = b"book\x00\x00\x00\x00mark\x00\x00\x00\x00" + bytes(range(256)) * 5


def can_symlink(tmp_path):
    try:
        os.symlink(tmp_path, tmp_path / "probe-link")
        return True
    except OSError:
        return False


@pytest.fixture
def folder(sample_dir, tmp_path):
    d = tmp_path / "output"
    (d / "matches").mkdir(parents=True)
    shutil.copy(sample_dir / "01_stray_loader.png", d / "real.png")
    # a link folder like --link-dir makes, with the original's name
    (d / "matches" / "real.png").write_bytes(ALIAS_BYTES)
    # starts like an alias but isn't one: still a broken file
    (d / "bookish.png").write_bytes(b"bookmark? no, just text")
    return d


def test_finder_aliases_are_skipped_not_broken(folder):
    report = run_search(folder, "lighthouse", "-r", allow_errors=True)
    assert hit_names(report) == {"real.png"}
    assert report["skipped_links"] == {"Finder alias": 1}
    assert report["unreadable"] == 1  # only bookish.png
    assert report["scanned"] == 2     # links don't count as scanned images

    broken = run_search(folder, "-r", "--broken")
    assert hit_names(broken) == {"bookish.png"}
    assert broken["skipped_links"] == {"Finder alias": 1}


def test_summary_mentions_skipped_links(folder):
    code, _out, err = run_cli(folder, "lighthouse", "-r", "--progress", "never")
    assert code == 0
    assert "1 Finder alias(s) skipped (links, not images" in err


def test_symlinked_files_are_skipped_unless_followed(folder, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks need Developer Mode or admin rights here")
    os.symlink(folder / "real.png", folder / "matches" / "linked.png")

    report = run_search(folder, "lighthouse", "-r", allow_errors=True)
    assert hit_names(report) == {"real.png"}
    assert report["skipped_links"] == {"Finder alias": 1, "symlink": 1}

    # the old option name still works; serial, so the original is met first
    followed = run_search(folder, "lighthouse", "-r", "--follow-symlinks",
                          "-j", "1", allow_errors=True)
    assert hit_names(followed) == {"real.png"}  # the link leads to the same image
    assert followed["duplicates"] == 1
    # the fake alias can't be opened: a dead link where aliases can be followed
    fake_alias = "dead Finder alias" if ssim.CAN_RESOLVE_ALIASES else "Finder alias"
    assert followed["skipped_links"] == {fake_alias: 1}


def test_links_never_reach_the_index(folder, tmp_path):
    if not can_symlink(tmp_path):
        pytest.skip("symlinks need Developer Mode or admin rights here")
    os.symlink(folder / "real.png", folder / "matches" / "linked.png")
    report = run_search(folder, "lighthouse", "-r", "--index", tmp_path / "i.sqlite",
                        allow_errors=True)
    assert report["index"]["read"] == 1  # real.png; the broken file isn't cached
