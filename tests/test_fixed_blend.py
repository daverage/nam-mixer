"""Tests for hybrid/fixed_blend.py -- Fixed Blend design mode, see
docs/blend-mode.md "TAB 2 -- FIXED BLEND" / "BLEND LEVEL MATCHING".
"""
from __future__ import annotations

import numpy as np

from hybrid.fixed_blend import (
    BlendDesign,
    build_fixed_blend,
    compute_active_trim,
    freeze_blend_design,
)
from hybrid.pipeline import RenderedPair


def _pair(amp_a, amp_b, envelope_db=None, sample_rate=48000):
    n = len(amp_a)
    if envelope_db is None:
        envelope_db = np.full(n, -10.0, dtype=np.float32)
    return RenderedPair(
        dry=np.zeros(n, dtype=np.float32),
        profiled_dry=np.zeros(n, dtype=np.float32),
        amp_a=np.asarray(amp_a, dtype=np.float32),
        amp_b=np.asarray(amp_b, dtype=np.float32),
        envelope_db=np.asarray(envelope_db, dtype=np.float32),
        source_envelope_db=np.asarray(envelope_db, dtype=np.float32),
        sample_rate=sample_rate,
    )


def test_mix_zero_returns_amp_a_exactly():
    a = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    b = np.array([0.9, -0.9, 0.9, -0.9], dtype=np.float32)
    pair = _pair(a, b)
    result = build_fixed_blend(pair, mix_b=0.0, auto_level=False)
    np.testing.assert_allclose(result.blend, a, atol=1e-6)


def test_mix_one_returns_trimmed_amp_b_exactly():
    a = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float32)
    b = np.array([0.05, -0.05, 0.05, -0.05], dtype=np.float32)
    pair = _pair(a, b)
    result = build_fixed_blend(pair, mix_b=1.0, auto_level=False, manual_b_trim_db=6.0)
    expected = b * (10.0 ** (6.0 / 20.0))
    np.testing.assert_allclose(result.blend, expected, atol=1e-6)


def test_mix_half_gives_expected_linear_blend():
    a = np.full(100, 0.4, dtype=np.float32)
    b = np.full(100, -0.2, dtype=np.float32)
    pair = _pair(a, b)
    result = build_fixed_blend(pair, mix_b=0.5, auto_level=False)
    expected = a * 0.5 + b * 0.5
    np.testing.assert_allclose(result.blend, expected, atol=1e-6)


def test_mix_is_independent_of_input_envelope():
    """Unlike Hybrid, changing the envelope must not change the blend at
    a fixed mix_b -- there is no crossover in Fixed Blend mode."""
    a = np.linspace(-1, 1, 1000).astype(np.float32)
    b = np.linspace(1, -1, 1000).astype(np.float32)

    quiet_env = np.full(1000, -50.0, dtype=np.float32)
    loud_env = np.full(1000, 0.0, dtype=np.float32)

    result_quiet = build_fixed_blend(_pair(a, b, envelope_db=quiet_env), mix_b=0.3, auto_level=False)
    result_loud = build_fixed_blend(_pair(a, b, envelope_db=loud_env), mix_b=0.3, auto_level=False)
    np.testing.assert_allclose(result_quiet.blend, result_loud.blend, atol=1e-9)


def test_manual_trim_shifts_amp_b_before_mixing():
    a = np.zeros(100, dtype=np.float32)
    b = np.full(100, 0.1, dtype=np.float32)
    pair = _pair(a, b)
    result = build_fixed_blend(pair, mix_b=1.0, auto_level=False, manual_b_trim_db=20.0)
    expected = b * (10.0 ** (20.0 / 20.0))
    np.testing.assert_allclose(result.blend, expected, atol=1e-6)
    assert result.effective_b_trim_db == 20.0


def test_active_level_match_uses_active_playing_not_crossover_region():
    """compute_active_trim must ignore silent samples and NOT depend on any
    crossover config -- it only needs an active-signal mask over the whole
    clip (docs/blend-mode.md "BLEND LEVEL MATCHING")."""
    n = 1000
    envelope_db = np.concatenate([
        np.full(500, -80.0, dtype=np.float32),   # silence -- excluded
        np.full(500, -10.0, dtype=np.float32),   # active
    ])
    amp_a = np.concatenate([np.full(500, 0.0, dtype=np.float32), np.full(500, 0.5, dtype=np.float32)])
    amp_b = np.concatenate([np.full(500, 10.0, dtype=np.float32), np.full(500, 0.25, dtype=np.float32)])  # loud "silent" garbage that must be ignored

    result = compute_active_trim(envelope_db, amp_a, amp_b)
    # Only the active half matters: amp_a=0.5, amp_b=0.25 -> +6.02 dB trim.
    expected_trim = 20.0 * np.log10(0.5 / 0.25)
    assert abs(result.suggested_b_trim_db - expected_trim) < 0.01
    assert result.n_active_samples == 500


def test_build_fixed_blend_auto_level_matches_build_active_trim():
    n = 1000
    envelope_db = np.full(n, -10.0, dtype=np.float32)
    amp_a = np.full(n, 0.4, dtype=np.float32)
    amp_b = np.full(n, 0.1, dtype=np.float32)
    pair = _pair(amp_a, amp_b, envelope_db=envelope_db)
    result = build_fixed_blend(pair, mix_b=1.0, auto_level=True)
    expected_trim = 20.0 * np.log10(0.4 / 0.1)
    assert abs(result.auto_trim_db - expected_trim) < 0.01
    np.testing.assert_allclose(result.blend, amp_b * (10.0 ** (expected_trim / 20.0)), atol=1e-4)


def test_mix_b_clamped_to_unit_range():
    a = np.full(10, 1.0, dtype=np.float32)
    b = np.full(10, -1.0, dtype=np.float32)
    pair = _pair(a, b)
    result_high = build_fixed_blend(pair, mix_b=5.0, auto_level=False)
    result_low = build_fixed_blend(pair, mix_b=-5.0, auto_level=False)
    np.testing.assert_allclose(result_high.blend, b, atol=1e-6)
    np.testing.assert_allclose(result_low.blend, a, atol=1e-6)


def test_freeze_blend_design_records_frozen_effective_trim():
    a = np.full(100, 0.4, dtype=np.float32)
    b = np.full(100, 0.1, dtype=np.float32)
    pair = _pair(a, b)
    result = build_fixed_blend(pair, mix_b=0.7, auto_level=True, manual_b_trim_db=1.0)
    design = freeze_blend_design(pair, result, amp_a_path="a.nam", amp_b_path="b.nam", alignment_enabled=False)
    assert design.mode == "blend"
    assert design.mix_b == 0.7
    assert abs(design.mix_a - 0.3) < 1e-9
    assert design.effective_b_trim_db == result.effective_b_trim_db
    assert design.manual_b_trim_db == 1.0


def test_blend_design_round_trips_through_json(tmp_path):
    design = BlendDesign(amp_a_path="a.nam", amp_b_path="b.nam", mix_b=0.25)
    path = design.write_json(tmp_path / "blend_design.json")
    loaded = BlendDesign.read_json(path)
    assert loaded == design
