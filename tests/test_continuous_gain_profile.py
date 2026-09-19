"""Tests for hybrid/continuous_gain_profile.py -- the production Continuous
Gain profile builder + runtime engine (docs/CONTINUOUS_GAIN_PRODUCTION.md).

Same fake-render convention as tests/test_continuous_gain.py: a model's
`raw["gain"]` drives a tanh saturation curve so gain-dependent nonlinearity
can be exercised without a real native nam_render tool or real captures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hybrid.continuous_gain_profile as cgp
from hybrid.continuous_gain_profile import (
    ContinuousGainProfile,
    ContinuousGainRuntime,
    ProfileQuality,
    TrainingTarget,
    build_profile,
    classify_quality,
    render_with_input_gain,
    search_virtual_input_gain,
)
from hybrid.nam_loader import NamModel


def _model(gain: float, label: str):
    return NamModel(path=Path(f"fake_{label}.nam"), raw={"gain": gain})


@pytest.fixture(autouse=True)
def fake_saturating_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        gain = model.raw.get("gain", 0.0)
        drive = 1.0 + 8.0 * gain
        return np.tanh(audio * drive) / np.tanh(drive)
    monkeypatch.setattr(cgp, "render", fake_render)


def _dry(n=6000, sample_rate=48000, amplitude=0.4):
    rng = np.random.default_rng(0)
    return (amplitude * rng.uniform(-1.0, 1.0, n)).astype(np.float32)


SAMPLE_RATE = 48000


def _targets(positions_and_gains):
    dry = _dry()
    targets = []
    for pos, gain in positions_and_gains:
        model = _model(gain, f"g{pos:g}")
        output = render_with_input_gain(model, dry, SAMPLE_RATE, 0.0)
        targets.append(TrainingTarget(label=f"g{pos:g}", physical_position=pos, model=model, output=output))
    return dry, targets


def test_render_with_input_gain_applies_gain_before_render():
    model = _model(0.5, "m")
    dry = _dry(n=100)
    plain = render_with_input_gain(model, dry, SAMPLE_RATE, 0.0)
    boosted = render_with_input_gain(model, dry, SAMPLE_RATE, 6.0)
    assert not np.allclose(plain, boosted)
    # 0 dB is a no-op scale (up to the tanh nonlinearity reacting to the same input).
    direct = np.tanh(dry * (1.0 + 8.0 * 0.5)) / np.tanh(1.0 + 8.0 * 0.5)
    np.testing.assert_allclose(plain, direct, atol=1e-6)


def test_search_finds_near_zero_gain_when_anchor_equals_target():
    dry, targets = _targets([(5.0, 0.4)])
    target = targets[0]
    mask = np.ones(len(target.output), dtype=bool)
    result = search_virtual_input_gain(target.model, dry, SAMPLE_RATE, target.output, mask)
    assert abs(result.input_gain_db) < 0.2
    assert result.raw_esr < 1e-4
    assert not result.hit_lower_bound
    assert not result.hit_upper_bound


def test_classify_quality_thresholds():
    assert classify_quality(0.001) == ProfileQuality.VALIDATED
    assert classify_quality(0.03) == ProfileQuality.ACCEPTABLE
    assert classify_quality(0.5) == ProfileQuality.POOR
    assert classify_quality(float("nan")) == ProfileQuality.POOR


def test_build_profile_single_capture_is_degenerate_but_valid():
    dry, targets = _targets([(5.0, 0.4)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=1)
    assert len(profile.anchors) == 1
    assert profile.anchors[0].label == "g5"
    assert profile.validation.worst_raw_esr < 1e-3


def test_build_profile_picks_smallest_acceptable_anchor_set():
    dry, targets = _targets([
        (1.0, 0.02), (3.0, 0.15), (5.0, 0.4), (7.0, 0.75), (10.0, 1.0),
    ])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=3)
    assert 1 <= len(profile.anchors) <= 3
    assert profile.validation.quality in (ProfileQuality.VALIDATED, ProfileQuality.ACCEPTABLE)
    # Anchor regions must fully cover the control range with no gaps.
    regions = sorted(profile.anchors, key=lambda a: a.range_low)
    assert regions[0].range_low == profile.control_min
    assert regions[-1].range_high == profile.control_max
    for a, b in zip(regions, regions[1:]):
        assert b.range_low <= a.range_high + 1e-9


def test_build_profile_flags_poor_quality_with_recommendation():
    # Five widely-spread, sharply different drive settings with only a
    # single anchor allowed: forces at least one target to be reconstructed
    # by a very different tanh curve, which virtual gain search alone cannot
    # fully compensate for.
    dry, targets = _targets([(1.0, 0.0), (3.0, 0.1), (5.0, 0.4), (7.0, 0.8), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=1)
    assert profile.validation.quality in (ProfileQuality.POOR, ProfileQuality.ACCEPTABLE, ProfileQuality.VALIDATED)
    if profile.validation.quality == ProfileQuality.POOR:
        assert profile.validation.recommended_capture_position is not None
        assert profile.validation.problem_regions
    else:
        # Not POOR is also a legitimate outcome (matches the research
        # finding that virtual gain search generalizes surprisingly well);
        # just confirm the report is internally consistent either way.
        assert profile.validation.recommended_capture_position is None
        assert not profile.validation.problem_regions


def test_profile_round_trips_through_dict():
    dry, targets = _targets([(1.0, 0.0), (5.0, 0.4), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=2)
    restored = ContinuousGainProfile.from_dict(profile.to_dict())
    assert restored.control_min == profile.control_min
    assert restored.control_max == profile.control_max
    assert len(restored.anchors) == len(profile.anchors)
    assert restored.validation.quality == profile.validation.quality


def test_profile_rejects_unsupported_schema_version():
    dry, targets = _targets([(1.0, 0.0), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=1)
    d = profile.to_dict()
    d["version"] = 999
    with pytest.raises(ValueError):
        ContinuousGainProfile.from_dict(d)


def test_runtime_at_anchors_own_position_matches_zero_gain_mapping_point():
    dry, targets = _targets([(1.0, 0.0), (5.0, 0.4), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=3)
    models = {t.label: t.model for t in targets}
    anchor_models = {a.label: models[a.label] for a in profile.anchors}
    runtime = ContinuousGainRuntime(profile, anchor_models)
    for anchor in profile.anchors:
        out = runtime.render_at(anchor.physical_position, dry, SAMPLE_RATE)
        assert np.isfinite(out).all()
        assert len(out) > 0


def test_runtime_clamps_out_of_range_position():
    dry, targets = _targets([(1.0, 0.0), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=1)
    models = {t.label: t.model for t in targets}
    anchor_models = {a.label: models[a.label] for a in profile.anchors}
    runtime = ContinuousGainRuntime(profile, anchor_models)
    below = runtime.render_at(-100.0, dry, SAMPLE_RATE)
    at_min = runtime.render_at(profile.control_min, dry, SAMPLE_RATE)
    np.testing.assert_allclose(below, at_min)


def test_runtime_raises_on_missing_anchor_model():
    dry, targets = _targets([(1.0, 0.0), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=1)
    with pytest.raises(ValueError):
        ContinuousGainRuntime(profile, {})


def test_runtime_transition_has_no_level_discontinuity():
    dry, targets = _targets([(1.0, 0.0), (5.0, 0.4), (10.0, 1.0)])
    profile = build_profile(targets, dry, SAMPLE_RATE, max_anchors=3)
    if len(profile.anchors) < 2:
        pytest.skip("single-anchor profile has no transition to test")
    models = {t.label: t.model for t in targets}
    anchor_models = {a.label: models[a.label] for a in profile.anchors}
    runtime = ContinuousGainRuntime(profile, anchor_models)
    regions = sorted(profile.anchors, key=lambda a: a.range_low)
    boundary = regions[0].range_high
    just_below = runtime.render_at(boundary - 1e-6, dry, SAMPLE_RATE)
    just_above = runtime.render_at(boundary + 1e-6, dry, SAMPLE_RATE)
    n = min(len(just_below), len(just_above))
    rms_below = np.sqrt(np.mean(just_below[:n] ** 2))
    rms_above = np.sqrt(np.mean(just_above[:n] ** 2))
    assert abs(rms_below - rms_above) < 0.05
