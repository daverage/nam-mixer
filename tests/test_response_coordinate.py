"""Tests for hybrid/response_coordinate.py -- see
docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hybrid.continuous_gain as continuous_gain
from hybrid.continuous_gain import GainCapture, GainCaptureSet, run_ground_truth_harness
from hybrid.nam_loader import NamModel
from hybrid.response_coordinate import (
    RMS_ONLY_WEIGHTS,
    build_response_axis,
    combined_step_scores,
    compute_adjacent_metrics,
    interpolate_output_response_coordinate,
    oracle_response_mismatch,
    response_coordinate_for_position,
)


def _model(gain: float):
    return NamModel(path=Path(f"fake_{gain}.nam"), raw={"gain": gain})


@pytest.fixture(autouse=True)
def fake_saturating_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        gain = model.raw.get("gain", 0.0)
        drive = 1.0 + 8.0 * gain
        return np.tanh(audio * drive) / np.tanh(drive)
    monkeypatch.setattr(continuous_gain, "render", fake_render)


def _dry(n=4000, amplitude=0.4):
    rng = np.random.default_rng(0)
    return (amplitude * rng.uniform(-1.0, 1.0, n)).astype(np.float32)


def _uneven_set():
    """A sweep with a deliberately uneven electrical response: the jump from
    gain 0.0->0.05 (positions 1->2) is tiny, but 0.05->0.9 (positions 2->3)
    is huge -- like a front-loaded control law/knee."""
    captures = [
        GainCapture(model=_model(0.0), control_position=1.0, label="g1"),
        GainCapture(model=_model(0.05), control_position=2.0, label="g2"),
        GainCapture(model=_model(0.9), control_position=3.0, label="g3"),
        GainCapture(model=_model(1.0), control_position=4.0, label="g4"),
    ]
    return GainCaptureSet(captures)


def test_compute_adjacent_metrics_returns_one_entry_per_step():
    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    assert len(steps) == 3
    assert [s.lower_label for s in steps] == ["g1", "g2", "g3"]
    assert [s.upper_label for s in steps] == ["g2", "g3", "g4"]


def test_adjacent_metrics_reflect_the_uneven_response():
    """The g2->g3 step (0.05->0.9 electrical gain) should show a much larger
    level change than g1->g2 (0.0->0.05)."""
    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    assert steps[1].level_change_db > steps[0].level_change_db * 2


def test_response_axis_places_most_change_in_the_big_step():
    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    axis = build_response_axis(steps, first_position=1.0)
    # g1->g2 (tiny electrical change) should cover much less of the 0..1
    # response range than g2->g3 (huge electrical change).
    span_g1_g2 = axis.coordinates[1] - axis.coordinates[0]
    span_g2_g3 = axis.coordinates[2] - axis.coordinates[1]
    assert span_g2_g3 > span_g1_g2 * 2
    assert axis.coordinates[0] == pytest.approx(0.0)
    assert axis.coordinates[-1] == pytest.approx(1.0)


def test_response_coordinate_for_position_is_monotonic():
    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    axis = build_response_axis(steps, first_position=1.0)
    values = [response_coordinate_for_position(axis, p) for p in np.linspace(1.0, 4.0, 20)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_rms_only_weights_ignore_spectral_component():
    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    scores_combined = combined_step_scores(steps)
    scores_rms_only = combined_step_scores(steps, weights=RMS_ONLY_WEIGHTS)
    # Both should still weight the big step (index 1) highest.
    assert np.argmax(scores_combined) == np.argmax(scores_rms_only) == 1


def test_response_coordinate_interpolation_differs_from_knob_linear_on_uneven_sweep():
    from hybrid.continuous_gain import interpolate_output

    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    axis = build_response_axis(steps, first_position=1.0)

    # Position 2.5 (knob-linear midpoint between g2/g3) should get pulled
    # toward g3 in response space, since almost all the "real" change in
    # that bracket happens close to g3.
    knob_linear = interpolate_output(harness, cs, cs.normalized_position(GainCapture(model=_model(0), control_position=2.5)))
    response_based = interpolate_output_response_coordinate(harness, cs, cs.normalized_position(GainCapture(model=_model(0), control_position=2.5)), axis)
    assert not np.allclose(knob_linear, response_based)


def test_response_coordinate_interpolation_is_close_to_knob_linear_on_near_even_sweep():
    """On a sweep where every step has a NEARLY equal response distance
    (small, evenly-spaced electrical gain values, so the residual
    nonlinearity of the fake saturator is the only source of unevenness),
    the response-coordinate blend weights should stay close to the
    knob-linear ones -- unlike the deliberately uneven sweep above, where
    they diverge sharply."""
    from hybrid.continuous_gain import interpolate_output

    captures = [
        GainCapture(model=_model(g / 10.0), control_position=float(g), label=f"g{g}")
        for g in range(1, 6)
    ]
    cs = GainCaptureSet(captures)
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    steps = compute_adjacent_metrics(harness, cs)
    axis = build_response_axis(steps, first_position=1.0)

    query = GainCapture(model=_model(0), control_position=2.5)
    pos = cs.normalized_position(query)
    knob_linear = interpolate_output(harness, cs, pos)
    response_based = interpolate_output_response_coordinate(harness, cs, pos, axis)
    correlation = np.corrcoef(knob_linear, response_based)[0, 1]
    assert correlation > 0.999


def test_oracle_response_mismatch_is_diagnostic_only_and_detects_offset_knee():
    cs = _uneven_set()
    harness = run_ground_truth_harness(cs, _dry(), 48000, calibration_mode="raw")
    lower, hidden, upper = cs.training_captures()[0], cs.training_captures()[1], cs.training_captures()[2]
    # Reference steps from the OTHER pair (g3->g4) give a normalization
    # range independent of the two local distances being measured, avoiding
    # the degenerate always-{0,1} collapse a same-sample min-max would give.
    reference_steps = compute_adjacent_metrics(harness, cs)
    result = oracle_response_mismatch(harness, lower, hidden, upper, reference_steps)
    assert result.hidden_label == "g2"
    assert result.knob_linear_fraction == pytest.approx(0.5)
    # g2 sits almost right at the START of the big response jump (g2->g3),
    # so its TRUE response fraction between g1 and g3 should be well below
    # the knob-linear midpoint of 0.5.
    assert result.true_response_fraction < 0.3
    assert result.mismatch > 0.2


def test_level_spectral_correlation_detects_redundant_dimensions():
    from hybrid.response_coordinate import AdjacentStepMetrics, level_spectral_correlation

    # level and spectral distance move in lockstep -- collapsing loses
    # nothing here, so correlation should be strongly positive.
    steps = [
        AdjacentStepMetrics("a", "b", 1.0, 2.0, level_change_db=1.0, peak_change_db=0.0, spectral_distance=0.01),
        AdjacentStepMetrics("b", "c", 2.0, 3.0, level_change_db=2.0, peak_change_db=0.0, spectral_distance=0.02),
        AdjacentStepMetrics("c", "d", 3.0, 4.0, level_change_db=4.0, peak_change_db=0.0, spectral_distance=0.04),
    ]
    assert level_spectral_correlation(steps) == pytest.approx(1.0, abs=1e-6)


def test_level_spectral_correlation_detects_independent_dimensions():
    from hybrid.response_coordinate import AdjacentStepMetrics, level_spectral_correlation

    # level is large exactly where spectral distance is small -- collapsing
    # these into one score would hide the real disagreement.
    steps = [
        AdjacentStepMetrics("a", "b", 1.0, 2.0, level_change_db=5.0, peak_change_db=0.0, spectral_distance=0.01),
        AdjacentStepMetrics("b", "c", 2.0, 3.0, level_change_db=1.0, peak_change_db=0.0, spectral_distance=0.05),
        AdjacentStepMetrics("c", "d", 3.0, 4.0, level_change_db=0.2, peak_change_db=0.0, spectral_distance=0.09),
    ]
    assert level_spectral_correlation(steps) < -0.9
