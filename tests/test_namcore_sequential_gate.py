"""Native compatibility gate for the experimental Sequential export path.

Run explicitly with the two renderers built from CMake:
NAM_RENDER_BASELINE=native/nam_render/build/nam_render \\
NAM_RENDER_SEQUENTIAL=native/nam_render/build-sequential/nam_render \\
python -m pytest tests/test_namcore_sequential_gate.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


BASELINE = os.environ.get("NAM_RENDER_BASELINE")
SEQUENTIAL = os.environ.get("NAM_RENDER_SEQUENTIAL")


def _require_renderers() -> tuple[Path, Path]:
    if not BASELINE or not SEQUENTIAL:
        pytest.skip("set NAM_RENDER_BASELINE and NAM_RENDER_SEQUENTIAL to run the native Sequential gate")
    return Path(BASELINE), Path(SEQUENTIAL)


def _nam(path: Path, architecture: str, config: dict, weights: list[float], *, sample_rate: int = 48000) -> Path:
    path.write_text(json.dumps({"version": "0.7.0", "architecture": architecture, "config": config, "weights": weights, "sample_rate": sample_rate}))
    return path


def _render(exe: Path, model: Path, audio: np.ndarray, tmp_path: Path, *, slim: float | None = None) -> np.ndarray:
    source, destination = tmp_path / f"in-{exe.parent.name}.wav", tmp_path / f"out-{exe.parent.name}.wav"
    sf.write(source, audio.astype(np.float32), 48000, subtype="FLOAT")
    command = [str(exe)] + ([] if slim is None else ["--slim", str(slim)]) + [str(model), str(source), str(destination)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr or result.stdout
    rendered, sample_rate = sf.read(destination, dtype="float32")
    assert sample_rate == 48000
    return rendered


def _linear(path: Path, taps: list[float]) -> Path:
    return _nam(path, "Linear", {"receptive_field": len(taps), "bias": False, "implementation": "auto"}, taps)


def test_existing_wavenet_and_slimmable_models_are_numerically_stable(tmp_path: Path):
    baseline, sequential = _require_renderers()
    source = Path("native/nam_render/build-sequential/_deps/namcore-src/example_models")
    audio = np.linspace(-0.2, 0.2, 1024, dtype=np.float32)
    for name, slim in (("wavenet.nam", None), ("slimmable_container.nam", 0.0), ("slimmable_container.nam", 1.0)):
        old = _render(baseline, source / name, audio, tmp_path, slim=slim)
        new = _render(sequential, source / name, audio, tmp_path, slim=slim)
        np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)


def test_linear_and_sequential_fir_are_exact(tmp_path: Path):
    _, sequential = _require_renderers()
    taps = [0.5, -0.25, 0.125]
    audio = np.zeros(128, dtype=np.float32); audio[0] = 1.0; audio[9] = -0.5
    linear = _linear(tmp_path / "linear.nam", taps)
    identity = _linear(tmp_path / "identity.nam", [1.0])
    sequential_model = _nam(tmp_path / "sequential.nam", "Sequential", {"models": [json.loads(identity.read_text()), json.loads(linear.read_text())]}, [])
    expected = np.convolve(audio, np.asarray(taps, dtype=np.float32))[:len(audio)]
    np.testing.assert_allclose(_render(sequential, linear, audio, tmp_path), expected, rtol=0, atol=2e-7)
    np.testing.assert_allclose(_render(sequential, sequential_model, audio, tmp_path), expected, rtol=0, atol=2e-7)


def test_sequential_a2_followed_by_linear_matches_head_then_fir(tmp_path: Path):
    _, sequential = _require_renderers()
    a2 = Path("native/nam_render/build-sequential/_deps/namcore-src/example_models/A2.nam")
    child = json.loads(a2.read_text())
    linear = _linear(tmp_path / "cab.nam", [0.75, 0.125, -0.0625])
    composite = _nam(tmp_path / "a2-plus-cab.nam", "Sequential", {"models": [child, json.loads(linear.read_text())]}, [])
    audio = np.random.default_rng(7).standard_normal(512).astype(np.float32) * 0.05
    head = _render(sequential, a2, audio, tmp_path)
    expected = np.convolve(head, [0.75, 0.125, -0.0625])[:len(head)]
    np.testing.assert_allclose(_render(sequential, composite, audio, tmp_path), expected, rtol=0, atol=3e-6)
