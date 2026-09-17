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

from hybrid.sequential_nam import build_embedded_sequential


BASELINE = os.environ.get("NAM_RENDER_BASELINE")
SEQUENTIAL = os.environ.get("NAM_RENDER_SEQUENTIAL")


def _require_renderers() -> tuple[Path, Path]:
    if not BASELINE or not SEQUENTIAL:
        pytest.skip("set NAM_RENDER_BASELINE and NAM_RENDER_SEQUENTIAL to run the native Sequential gate")
    return Path(BASELINE), Path(SEQUENTIAL)


def _nam(path: Path, architecture: str, config: dict, weights: list[float], *, sample_rate: int = 48000) -> Path:
    path.write_text(json.dumps({"version": "0.7.0", "architecture": architecture, "config": config, "weights": weights, "sample_rate": sample_rate}))
    return path


def _render(exe: Path, model: Path, audio: np.ndarray, tmp_path: Path, *, slim: float | None = None, block_size: int = 64) -> np.ndarray:
    source, destination = tmp_path / f"in-{exe.parent.name}.wav", tmp_path / f"out-{exe.parent.name}.wav"
    sf.write(source, audio.astype(np.float32), 48000, subtype="FLOAT")
    command = [str(exe)] + ([] if slim is None else ["--slim", str(slim)]) + ["--block-size", str(block_size), str(model), str(source), str(destination)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr or result.stdout
    rendered, sample_rate = sf.read(destination, dtype="float32")
    assert sample_rate == 48000
    return rendered


def _linear(path: Path, taps: list[float]) -> Path:
    return _nam(path, "Linear", {"receptive_field": len(taps), "bias": False, "implementation": "auto"}, taps)


def _wavenet_prewarm_samples(model: dict) -> int:
    """Mirror NAMCore WaveNet's documented prewarm calculation."""
    config = model["config"]
    history = 1  # no condition DSP in the shipped A2 fixture
    for layer in config["layers"]:
        history += sum((kernel - 1) * dilation for kernel, dilation in zip(layer["kernel_sizes"], layer["dilations"]))
        history += int(layer["head"]["kernel_size"]) - 1
    return history


def _sequential_warmup_samples(a2_child: dict, linear_taps: int) -> int:
    # Container returns the active child history; Sequential sums child
    # prewarm histories. Linear's buffer history is its receptive field.
    active_child = (a2_child["config"]["submodels"][0]["model"]
                    if a2_child.get("architecture") == "SlimmableContainer" else a2_child)
    return _wavenet_prewarm_samples(active_child) + linear_taps


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


def test_sequential_a2_followed_by_linear_matches_head_then_fir_after_stream_warmup(tmp_path: Path):
    _, sequential = _require_renderers()
    a2 = Path("native/nam_render/build-sequential/_deps/namcore-src/example_models/A2.nam")
    child = json.loads(a2.read_text())
    payload = np.zeros(1024, dtype=np.float32)
    payload[73:] = np.random.default_rng(7).standard_normal(len(payload) - 73).astype(np.float32) * 0.05
    # The standalone head and a child inside Sequential prewarm differently at
    # stream startup. Compare a continuous, silence-prefixed stream and only
    # assess the audible payload after the documented warm-up prefix.
    cases = (([1.0], .73), ([0.75, 0.125, -0.0625], 1.0),
             ([0.4, -0.2, 0.1, 0.05, -0.025, 0.0125], .91),
             ([0.001] * 512, .65), ([0.001] * 2048, 1.2), ([0.001] * 8192, .47))
    for index, (taps, final_scalar) in enumerate(cases):
        composite_json, package_record = build_embedded_sequential(
            child, np.asarray(taps), sample_rate=48000, final_scalar=final_scalar)
        composite = tmp_path / f"a2-plus-cab-{index}.nam"
        composite.write_text(json.dumps(composite_json))
        full_child = composite_json["config"]["models"][0]
        warmup = _sequential_warmup_samples(full_child, len(taps))
        audio = np.concatenate([np.zeros(warmup, dtype=np.float32), payload])
        for block_size in (1, 7, 64, 257):
            head = _render(sequential, a2, audio, tmp_path, slim=0.0, block_size=block_size)
            expected = np.convolve(head, np.asarray(taps) * package_record["final_linear_scalar"])[:len(head)]
            actual = _render(sequential, composite, audio, tmp_path, block_size=block_size)
            # Each CLI invocation is a fresh Reset; confirm that reset is
            # deterministic as well as comparing steady-state streams.
            repeated = _render(sequential, composite, audio, tmp_path, block_size=block_size)
            np.testing.assert_array_equal(repeated, actual)
            np.testing.assert_allclose(actual[warmup:], expected[warmup:], rtol=0, atol=3e-6)


def test_sequential_does_not_expose_nested_a2_slim_selection(tmp_path: Path):
    """Do not label a Sequential A2 child as Full or Lite without runtime support."""
    _, sequential = _require_renderers()
    a2 = Path("native/nam_render/build-sequential/_deps/namcore-src/example_models/A2.nam")
    linear = _linear(tmp_path / "cab.nam", [1.0])
    composite = _nam(tmp_path / "a2-plus-cab.nam", "Sequential", {"models": [json.loads(a2.read_text()), json.loads(linear.read_text())]}, [])
    source, destination = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(128, dtype=np.float32), 48000, subtype="FLOAT")
    result = subprocess.run([str(sequential), "--slim", "1.0", str(composite), str(source), str(destination)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "SlimmableModel" in (result.stderr + result.stdout)


def test_explicit_full_a2_child_matches_normal_container_full_selection(tmp_path: Path):
    """The embedded policy extracts Full rather than relying on a nested
    container. Prove this exact mapping against NAMCore's normal interface."""
    _, sequential = _require_renderers()
    a2 = Path("native/nam_render/build-sequential/_deps/namcore-src/example_models/A2.nam")
    container = json.loads(a2.read_text())
    full = min(container["config"]["submodels"], key=lambda item: float(item["max_value"]))["model"]
    full_path = tmp_path / "explicit-full.nam"
    full_path.write_text(json.dumps(full))
    identity = _linear(tmp_path / "identity.nam", [1.0])
    composite = _nam(tmp_path / "explicit-full-plus-identity.nam", "Sequential",
                     {"models": [full, json.loads(identity.read_text())]}, [])
    audio = np.random.default_rng(9).standard_normal(1024).astype(np.float32) * .03
    normal_full = _render(sequential, a2, audio, tmp_path, slim=0.0)
    extracted = _render(sequential, full_path, audio, tmp_path)
    embedded = _render(sequential, composite, audio, tmp_path)
    np.testing.assert_allclose(extracted, normal_full, rtol=0, atol=3e-6)
    np.testing.assert_allclose(embedded, extracted, rtol=0, atol=3e-6)
