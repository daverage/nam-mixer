"""Tests for hybrid/continuous_gain.py -- Continuous Gain Model research
harness, Phase 1 (ground truth) and Phase 2 (baseline interpolation) from
docs/CONTINUOUS_GAIN.md.

Uses a fake render() (no native nam_render tool required), same convention
as tests/test_pipeline_render.py: a per-model "gain" drives a tanh
saturation curve, standing in for a real amp's gain-dependent nonlinearity
without needing real captures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hybrid.continuous_gain as continuous_gain
from hybrid.continuous_gain import (
    GainCapture,
    GainCaptureSet,
    interpolate_output,
    leave_one_out_validation,
    run_ground_truth_harness,
)
from hybrid.nam_loader import NamModel


def _model(gain: float, input_level_dbu=None):
    raw = {"gain": gain}
    if input_level_dbu is not None:
        raw["input_level_dbu"] = input_level_dbu
    return NamModel(path=Path(f"fake_gain_{gain}.nam"), raw=raw)


@pytest.fixture(autouse=True)
def fake_saturating_render(monkeypatch):
    """A model's `raw["gain"]` (0..1) drives how hard it saturates -- higher
    gain compresses/distorts a fixed-amplitude input more, similar in shape
    (though not in physical accuracy) to a real amp gain sweep."""
    def fake_render(model, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        gain = model.raw.get("gain", 0.0)
        drive = 1.0 + 8.0 * gain
        return np.tanh(audio * drive) / np.tanh(drive)
    monkeypatch.setattr(continuous_gain, "render", fake_render)


def _dry(n=4000, sample_rate=48000, amplitude=0.4):
    rng = np.random.default_rng(0)
    return (amplitude * rng.uniform(-1.0, 1.0, n)).astype(np.float32)


def _training_set(hidden_gain=None):
    captures = [
        GainCapture(model=_model(0.0), control_position=1.0, label="g1"),
        GainCapture(model=_model(0.25), control_position=3.0, label="g3"),
        GainCapture(model=_model(0.75), control_position=7.0, label="g7"),
        GainCapture(model=_model(1.0), control_position=10.0, label="g10"),
    ]
    if hidden_gain is not None:
        captures.append(GainCapture(
            model=_model(hidden_gain[0]), control_position=hidden_gain[1],
            label="hidden", hidden=True,
        ))
    return GainCaptureSet(captures)


def test_capture_set_rejects_fewer_than_two_training_captures():
    with pytest.raises(ValueError):
        GainCaptureSet([GainCapture(model=_model(0.0), control_position=1.0)])


def test_capture_set_rejects_duplicate_positions():
    with pytest.raises(ValueError):
        GainCaptureSet([
            GainCapture(model=_model(0.0), control_position=5.0),
            GainCapture(model=_model(1.0), control_position=5.0),
        ])


def test_normalized_position_spans_zero_to_one_over_training_range():
    capture_set = _training_set()
    training = capture_set.training_captures()
    assert capture_set.normalized_position(training[0]) == pytest.approx(0.0)
    assert capture_set.normalized_position(training[-1]) == pytest.approx(1.0)
    assert capture_set.normalized_position(training[2]) == pytest.approx(6.0 / 9.0)


def test_neighbors_returns_bracketing_training_captures():
    capture_set = _training_set()
    lower, upper = capture_set.neighbors(0.42)
    assert (lower.label, upper.label) == ("g3", "g7")


def test_ground_truth_harness_renders_every_capture_including_hidden():
    capture_set = _training_set(hidden_gain=(0.5, 5.0))
    dry = _dry()
    harness = run_ground_truth_harness(capture_set, dry, 48000, calibration_mode="raw")
    labels = {r.capture.label for r in harness.rendered}
    assert labels == {"g1", "g3", "g7", "g10", "hidden"}


def test_partial_calibration_falls_back_to_raw_with_warning():
    captures = [
        GainCapture(model=_model(0.0, input_level_dbu=12.0), control_position=1.0, label="g1"),
        GainCapture(model=_model(1.0), control_position=10.0, label="g10"),  # no metadata
    ]
    capture_set = GainCaptureSet(captures)
    harness = run_ground_truth_harness(capture_set, _dry(), 48000, calibration_mode="auto")
    assert harness.calibration_warning is not None
    assert all(r.calibration_gain_db == 0.0 for r in harness.rendered)


def test_full_calibration_applies_when_every_capture_has_metadata():
    captures = [
        GainCapture(model=_model(0.0, input_level_dbu=6.0), control_position=1.0, label="g1"),
        GainCapture(model=_model(1.0, input_level_dbu=12.0), control_position=10.0, label="g10"),
    ]
    capture_set = GainCaptureSet(captures)
    harness = run_ground_truth_harness(
        capture_set, _dry(), 48000, calibration_mode="auto", reference_input_level_dbu=12.0,
    )
    assert harness.calibration_warning is None
    assert harness.by_label("g1").calibration_gain_db == pytest.approx(6.0)
    assert harness.by_label("g10").calibration_gain_db == pytest.approx(0.0)


def test_output_normalization_zero_preserves_real_level_differences():
    capture_set = _training_set()
    harness = run_ground_truth_harness(capture_set, _dry(), 48000, calibration_mode="raw", output_normalization_amount=0.0)
    # Higher-gain tanh saturation compresses harder -> different active RMS.
    assert harness.by_label("g1").active_rms_dbfs != pytest.approx(harness.by_label("g10").active_rms_dbfs, abs=0.05)


def test_output_normalization_one_matches_active_rms_to_reference():
    capture_set = _training_set()
    harness = run_ground_truth_harness(capture_set, _dry(), 48000, calibration_mode="raw", output_normalization_amount=1.0)
    reference_dbfs = harness.by_label("g1").active_rms_dbfs
    for r in harness.rendered:
        assert r.active_rms_dbfs == pytest.approx(reference_dbfs, abs=0.05)


def test_interpolate_output_at_zero_and_one_matches_endpoints():
    capture_set = _training_set()
    harness = run_ground_truth_harness(capture_set, _dry(), 48000, calibration_mode="raw")
    np.testing.assert_allclose(
        interpolate_output(harness, capture_set, 0.0), harness.by_label("g1").output, atol=1e-6,
    )
    np.testing.assert_allclose(
        interpolate_output(harness, capture_set, 1.0), harness.by_label("g10").output, atol=1e-6,
    )


def test_leave_one_out_validation_reports_esr_against_real_hidden_capture():
    # Withhold the true Gain-5 capture; ask the baseline to reconstruct it
    # from Gain-3/Gain-7 neighbours (matching the doc's "Test 1").
    capture_set = _training_set(hidden_gain=(0.5, 5.0))
    harness = run_ground_truth_harness(capture_set, _dry(), 48000, calibration_mode="raw")
    results = leave_one_out_validation(harness, capture_set)
    assert len(results) == 1
    result = results[0]
    assert result.capture_label == "hidden"
    assert result.normalized_position == pytest.approx(4.0 / 9.0)
    assert np.isfinite(result.metrics["raw_esr"])
    assert result.metrics["raw_esr"] >= 0.0


def test_leave_one_out_validation_rejects_extrapolation():
    """A hidden capture outside the training range is a different, harder
    question (extrapolation, not interpolation) -- doc: only interior
    withheld points are used for leave-one-out validation."""
    captures = [
        GainCapture(model=_model(0.0), control_position=3.0, label="g3"),
        GainCapture(model=_model(1.0), control_position=10.0, label="g10"),
        GainCapture(model=_model(0.0), control_position=1.0, label="outside", hidden=True),
    ]
    capture_set = GainCaptureSet(captures)
    harness = run_ground_truth_harness(capture_set, _dry(), 48000, calibration_mode="raw")
    with pytest.raises(ValueError):
        leave_one_out_validation(harness, capture_set)
