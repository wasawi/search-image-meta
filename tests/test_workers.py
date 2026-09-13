"""Worker set-up: a worker starting must never disturb one that is scanning."""

from helpers import ssim


def options(*patterns):
    return {"patterns": list(patterns), "regex": False, "case_sensitive": False,
            "fields": None, "snippets": False, "deep": False, "mode": "all",
            "scope": "image", "connected": None}


def test_starting_a_worker_replaces_the_config_instead_of_clearing_it():
    # thread-pool workers share the module; each runs _worker_init as it
    # starts, possibly while another thread is halfway through scan_one
    ssim._worker_init(options("first"))
    in_use = ssim._CFG
    ssim._worker_init(options("second"))
    assert ssim._CFG is not in_use
    assert [rx.pattern for rx in in_use["rx"]] == ["first"]
    assert [rx.pattern for rx in ssim._CFG["rx"]] == ["second"]


def test_scan_uses_one_config_from_start_to_end(sample_dir, monkeypatch):
    ssim._worker_init(options("MAIN_ckpt_s01"))
    real_extract = ssim.extract_metadata

    def extract_while_another_worker_starts(*args, **kwargs):
        ssim._worker_init(options("something_else_entirely"))
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(ssim, "extract_metadata", extract_while_another_worker_starts)
    path, hits, error, _summary, _record, _original = ssim.scan_one(
        str(sample_dir / "01_stray_loader.png"))
    assert error is None and hits
