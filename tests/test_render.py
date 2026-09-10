from pathlib import Path
from types import SimpleNamespace

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


requires_native_model = pytest.mark.skipif(
    not (MODEL_PATH.is_file() and _exe_available()),
    reason=(
        "requires a real .nam model in assets/nam_models/ (gitignored, "
        "user-provided) and a built native/nam_render tool -- see "
        "native/nam_render/README.md"
    ),
)


@requires_native_model
def test_render_real_model_length_and_sanity():
    model = load_nam(MODEL_PATH)
    sample_rate = int(model.sample_rate or 48000)
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.3, 0.3, sample_rate).astype(np.float32)

    out = render(model, audio, sample_rate)

    assert len(out) == len(audio)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


@requires_native_model
def test_render_is_deterministic():
    """Same DI through the same model twice should produce identical (or
    extremely close) output -- establishing this before generating any real
    training targets."""
    model = load_nam(MODEL_PATH)
    sample_rate = int(model.sample_rate or 48000)
    rng = np.random.default_rng(1)
    audio = rng.uniform(-0.3, 0.3, sample_rate).astype(np.float32)

    out1 = render(model, audio, sample_rate)
    out2 = render(model, audio, sample_rate)

    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_render_rejects_stereo_input(monkeypatch):
    import hybrid.render as render_module

    monkeypatch.setattr(render_module, "find_nam_render_exe", lambda: Path("unused-renderer"))
    model = SimpleNamespace(path=Path("unused-model.nam"))
    sample_rate = 48000
    stereo = np.zeros((100, 2), dtype=np.float32)
    with pytest.raises(NamRenderError):
        render(model, stereo, sample_rate)


def test_render_missing_exe_raises_nam_render_error(monkeypatch):
    import hybrid.render as render_module

    def _raise():
        raise NamRenderError("not found")

    monkeypatch.setattr(render_module, "find_nam_render_exe", _raise)
    model = SimpleNamespace(path=Path("unused-model.nam"))
    with pytest.raises(NamRenderError):
        render(model, np.zeros(100, dtype=np.float32), 48000)


def test_renderer_env_override(monkeypatch, tmp_path):
    executable = tmp_path / "nam_render"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setenv("NAM_RENDER_EXE", str(executable))
    assert find_nam_render_exe() == executable


def test_renderer_env_override_rejects_non_executable(monkeypatch, tmp_path):
    configured = tmp_path / "missing-renderer"
    monkeypatch.setenv("NAM_RENDER_EXE", str(configured))
    with pytest.raises(NamRenderError, match="NAM_RENDER_EXE"):
        find_nam_render_exe()
