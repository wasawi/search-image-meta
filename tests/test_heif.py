"""HEIC / HEIF photos, readable when the optional pillow-heif is installed."""

import pytest
from PIL import Image

from helpers import hit_names, run_search

pytest.importorskip("pillow_heif")


def test_heic_exif(tmp_path):
    img = Image.new("RGB", (16, 16))
    exif = img.getexif()
    exif[0x010E] = "HEIC_marker holiday in Girona"  # ImageDescription
    img.save(tmp_path / "photo.heic", exif=exif)
    assert hit_names(run_search(tmp_path, "HEIC_marker", "girona")) == {"photo.heic"}
