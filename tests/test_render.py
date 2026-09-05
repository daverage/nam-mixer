from pathlib import Path

import numpy as np
import pytest

from hybrid.nam_loader import load_nam
from hybrid.render import NamRenderError, find_nam_render_exe, render

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "assets" / "nam_models" / "FenderSuperReverb1977_Clean.nam"


def _exe_available() -> bool:
    try:
        find_nam_render_exe()
        return True
    except NamRenderError:
        return False


pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and _exe_available()),
    reason=(
        "requires a real .nam model in assets/nam_models/ (gitignored, "
        "user-provided) and a built native/nam_render tool -- see "
        "native/nam_render/README.md"
    ),
)


def test_render_real_model_length_and_sanity():
    model = load_nam(MODEL_PATH)
    sample_rate = int(model.sample_rate or 48000)
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.3, 0.3, sample_rate).astype(np.float32)

    out = render(model, audio, sample_rate)

    assert len(out) == len(audio)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


def test_render_missing_exe_raises_nam_render_error(monkeypatch):
    import hybrid.render as render_module

    def _raise():
        raise NamRenderError("not found")

    monkeypatch.setattr(render_module, "find_nam_render_exe", _raise)
    model = load_nam(MODEL_PATH)
    with pytest.raises(NamRenderError):
        render(model, np.zeros(100, dtype=np.float32), 48000)
