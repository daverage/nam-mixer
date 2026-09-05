"""Tests for hybrid/validation.py -- docs/phase3.md sections 24-28.

Uses a fake identity render() (same pattern as other pipeline tests) so
these run without the native nam_render tool or real .nam captures.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import hybrid.validation as validation
from hybrid.design import HybridDesign
from hybrid.validation import compute_esr_metrics, render_reference_hybrid


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate, slim=None):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(validation, "render", fake_render)


def _write_nam(path, input_level_dbu=None):
    raw = {"architecture": "WaveNet", "sample_rate": 48000.0}
    if input_level_dbu is not None:
        raw["input_level_dbu"] = input_level_dbu
        raw["output_level_dbu"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _design(amp_a_path, amp_b_path, **overrides):
    fields = dict(
        amp_a_path=str(amp_a_path),
        amp_b_path=str(amp_b_path),
        crossover_dbfs=-20.0,
        transition_width_db=8.0,
        effective_b_trim_db=2.0,
        calibration_mode="raw",
        alignment_enabled=False,
    )
    fields.update(overrides)
    return HybridDesign(**fields)


def test_render_reference_hybrid_matches_frozen_trim_and_crossover(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    design = _design(amp_a, amp_b, crossover_dbfs=-100.0, transition_width_db=2.0, effective_b_trim_db=6.0)

    rng = np.random.default_rng(0)
    dry = (0.3 * rng.uniform(-1, 1, 48000)).astype(np.float32)

    result = render_reference_hybrid(design, dry, 48000)

    # Crossover set far below any real signal level -> should be fully Amp B,
    # scaled by the frozen +6dB trim (identity render means amp_b == dry).
    expected_gain = 10.0 ** (6.0 / 20.0)
    np.testing.assert_allclose(result.hybrid, dry * expected_gain, atol=1e-5)
    assert result.alignment_offset_samples == 0


def test_render_reference_hybrid_uses_per_model_calibration(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam", input_level_dbu=8.0)
    amp_b = _write_nam(tmp_path / "b.nam", input_level_dbu=12.0)
    design = _design(
        amp_a, amp_b, crossover_dbfs=-100.0, effective_b_trim_db=0.0,
        calibration_mode="auto", reference_input_level_dbu=12.0,
    )
    dry = np.full(4800, 0.2, dtype=np.float32)
    result = render_reference_hybrid(design, dry, 48000)
    # Amp B needs 0dB compensation (already at reference) -> amp_b render == dry.
    np.testing.assert_allclose(result.amp_b, dry, atol=1e-6)
    # Amp A needs +4dB compensation -> its render is louder than dry.
    assert np.max(np.abs(result.amp_a)) > np.max(np.abs(dry))


def test_compute_esr_metrics_zero_for_identical_signals():
    a = np.full(1000, 0.5, dtype=np.float32)
    metrics = compute_esr_metrics(a, a)
    assert metrics["raw_esr"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["gain_normalized_esr"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rms_difference"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["peak_difference"] == pytest.approx(0.0, abs=1e-9)


def test_compute_esr_metrics_distinguishes_gain_from_shape_error():
    """A pure gain difference should show up in raw ESR but vanish after
    gain normalization; a shape difference should survive normalization."""
    rng = np.random.default_rng(1)
    reference = rng.uniform(-0.5, 0.5, 2000).astype(np.float64)

    louder = reference * 2.0
    gain_metrics = compute_esr_metrics(louder, reference)
    assert gain_metrics["raw_esr"] > 0.1
    assert gain_metrics["gain_normalized_esr"] == pytest.approx(0.0, abs=1e-9)

    distorted = reference + rng.uniform(-0.1, 0.1, 2000)
    shape_metrics = compute_esr_metrics(distorted, reference)
    assert shape_metrics["gain_normalized_esr"] > 0.0
