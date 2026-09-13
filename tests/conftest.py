import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import samples  # noqa: E402


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    """The synthetic ComfyUI samples, built once per test run."""
    out = tmp_path_factory.mktemp("samples")
    samples.build(out, quiet=True)
    return out
