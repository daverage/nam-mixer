from types import SimpleNamespace

import numpy as np
import pytest

from hybrid.character_analysis import CharacterAnalysisConfig, analyse_rendered_audio
from hybrid.character_blend import (
    CharacterBlendDesign,
    LowLevelResponseCheck,
    _adjacent_level_weights,
    _select_donor,
    build_character_blend,
    evaluate_low_level_response,
    freeze_character_design,
)


def _pair(n=4096, sample_rate=48000):
    t = np.arange(n) / sample_rate
    dry = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    amp_a = dry.copy()
    amp_b = (0.45 * np.tanh(dry * 9.0)).astype(np.float32)
    return SimpleNamespace(
        dry=dry, amp_a=amp_a, amp_b=amp_b, sample_rate=sample_rate,
        instrument_type="guitar", input_profile_id="vintage_humbucker", input_profile_gain_db=0.0,
        calibration_mode="raw", reference_input_level_dbu=12.0, calibration_applied=False,
        amp_a_model_input_level_dbu=None, amp_b_model_input_level_dbu=None,
        amp_a_calibration_gain_db=0.0, amp_b_calibration_gain_db=0.0, calibration_warning=None,
        amp_a_input_gain_db=0.0, amp_b_input_gain_db=0.0,
    )


def test_analysis_has_requested_level_grid_and_broad_spectrum():
    pair = _pair()
    cfg = CharacterAnalysisConfig(levels_db=(-18.0, -6.0), frequencies_hz=(100.0, 500.0, 2000.0))
    analysis = analyse_rendered_audio(pair.dry, pair.amp_a, pair.sample_rate, cfg)
    assert [x.input_gain_db for x in analysis.levels] == [-18.0, -6.0]
    assert all(len(x.spectrum_db) == 3 for x in analysis.levels)


def test_character_teacher_is_deterministic_and_not_parallel_sum():
    pair = _pair()
    design = CharacterBlendDesign("a.nam", "b.nam", tone_mix_b=.5, feel_mix_b=.5, drive_mix_b=0.0)
    one = build_character_blend(pair, design).blend
    two = build_character_blend(pair, design).blend
    parallel = .5 * pair.amp_a + .5 * pair.amp_b
    assert np.array_equal(one, two)
    assert not np.allclose(one, parallel)
    assert np.isfinite(one).all()


def test_drive_curve_is_smooth_and_freeze_roundtrips(tmp_path):
    pair = _pair()
    design = CharacterBlendDesign("a.nam", "b.nam", drive_low_mix_b=.1, drive_mid_mix_b=.5, drive_high_mix_b=.9)
    result = build_character_blend(pair, design)
    assert np.max(np.abs(np.diff(result.drive_weight_b))) < .1
    frozen = freeze_character_design(pair, result, "a.nam", "b.nam", drive_mix_b=.5)
    path = frozen.write_json(tmp_path / "character.json")
    assert CharacterBlendDesign.read_json(path).analysis_a == frozen.analysis_a


def test_freeze_character_design_preserves_per_amp_input_gains():
    pair = _pair()
    pair.amp_a_input_gain_db, pair.amp_b_input_gain_db = -3.0, 2.5
    result = build_character_blend(pair, CharacterBlendDesign("a.nam", "b.nam"))

    frozen = freeze_character_design(pair, result, "a.nam", "b.nam")

    assert frozen.amp_a_input_gain_db == -3.0
    assert frozen.amp_b_input_gain_db == 2.5


# ---------------------------------------------------------------------------
# Phase 1/2 -- adjacent-level interpolation weights never zero out below/
# above the analysis grid (docs/blend-mode-fixes.md).
# ---------------------------------------------------------------------------

_ANALYSIS_LEVELS = np.array([-24.0, -18.0, -12.0, -6.0, 0.0, 6.0])
_WIDE_ENVELOPE_DB = [-120, -80, -60, -48, -36, -30, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 12, 24]


def test_adjacent_level_weights_never_collapse_outside_the_grid():
    envelope = np.array(_WIDE_ENVELOPE_DB, dtype=np.float64)
    weights = _adjacent_level_weights(_ANALYSIS_LEVELS, envelope)
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert not np.any(weights.sum(axis=0) == 0.0)
    assert np.allclose(weights.sum(axis=0), 1.0)
    below = envelope <= _ANALYSIS_LEVELS[0]
    above = envelope >= _ANALYSIS_LEVELS[-1]
    assert np.allclose(weights[0, below], 1.0) and np.allclose(weights[1:, below], 0.0)
    assert np.allclose(weights[-1, above], 1.0) and np.allclose(weights[:-1, above], 0.0)


def test_adjacent_level_weights_match_documented_examples():
    weights = _adjacent_level_weights(_ANALYSIS_LEVELS, np.array([-35.0, -21.0, -15.0, 12.0]))
    assert np.allclose(weights[:, 0], [1, 0, 0, 0, 0, 0])   # -35 dB: 100% -24 dB state
    assert np.allclose(weights[:, 1], [.5, .5, 0, 0, 0, 0])  # -21 dB: 50/50 -24/-18
    assert np.allclose(weights[:, 2], [0, .5, .5, 0, 0, 0])  # -15 dB: 50/50 -18/-12
    assert np.allclose(weights[:, 3], [0, 0, 0, 0, 0, 1])    # +12 dB: 100% +6 dB state


# ---------------------------------------------------------------------------
# Phase 3 -- synthetic low-level response regression: a real amp/blend must
# never collapse to digital silence just because playing got soft.
# ---------------------------------------------------------------------------

def _linear_pair(gain_db, n=8192, sample_rate=48000, gain_a=0.3, gain_b=1.2):
    """Guaranteed nonzero output for any nonzero input -- both donors are a
    plain linear gain on the dry signal, so any collapse to silence can only
    come from the interpolation, not amp/DI nonlinearity."""
    t = np.arange(n) / sample_rate
    amplitude = 0.2 * (10.0 ** (gain_db / 20.0))
    dry = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return SimpleNamespace(dry=dry, amp_a=(dry * gain_a).astype(np.float32), amp_b=(dry * gain_b).astype(np.float32), sample_rate=sample_rate)


def _rms_dbfs(audio):
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-10))


def test_low_level_sweep_never_cliffs_to_digital_silence():
    design = CharacterBlendDesign("a.nam", "b.nam", tone_mix_b=.5, feel_mix_b=.5, drive_mix_b=.5)
    sweep_db = [6, 0, -6, -12, -18, -24, -30, -36, -42, -48]
    outputs = [build_character_blend(_linear_pair(g), design).blend for g in sweep_db]
    assert all(np.isfinite(o).all() for o in outputs)
    rms = [_rms_dbfs(o) for o in outputs]
    assert all(r > -140.0 for r in rms)  # never digital silence
    for i in range(1, len(rms)):
        input_step = sweep_db[i] - sweep_db[i - 1]
        output_step = rms[i] - rms[i - 1]
        assert output_step < 1.0  # falls (or stays flat) with input, no spurious rise
        assert output_step > input_step - 10.0  # no near-infinite reduction for a moderate input step


# ---------------------------------------------------------------------------
# Phase 5 -- the export-gate LowLevelResponseCheck.
# ---------------------------------------------------------------------------

def test_evaluate_low_level_response_passes_for_healthy_linear_amps():
    design = CharacterBlendDesign("a.nam", "b.nam", tone_mix_b=.5, feel_mix_b=.5, drive_mix_b=.5)
    check = evaluate_low_level_response(_linear_pair, design)
    assert isinstance(check, LowLevelResponseCheck)
    assert check.ok
    assert not check.dead_zone_detected
    assert check.warning is None


def test_evaluate_low_level_response_flags_a_hard_gate(monkeypatch):
    import hybrid.character_blend as character_blend_module

    def fake_build(pair, design, **kwargs):
        # Simulates the pre-fix bug: flat output until the envelope falls
        # outside an implicit analysis grid, then a hard drop to silence.
        rms = 0.4 if float(np.max(np.abs(pair.dry))) > 0.01 else 0.0
        return SimpleNamespace(blend=np.full(256, rms, dtype=np.float32))

    monkeypatch.setattr(character_blend_module, "build_character_blend", fake_build)
    design = CharacterBlendDesign("a.nam", "b.nam")

    def build_pair_at_gain(gain_db):
        amplitude = 0.2 * (10.0 ** (gain_db / 20.0))
        return SimpleNamespace(dry=np.full(256, amplitude, dtype=np.float32), amp_a=None, amp_b=None, sample_rate=48000)

    check = evaluate_low_level_response(build_pair_at_gain, design)
    assert not check.ok
    assert check.dead_zone_detected
    assert check.warning is not None


# ---------------------------------------------------------------------------
# Phase 8 -- Drive donor selection is a threshold, not a continuous morph.
# Documented rather than changed by this fix.
# ---------------------------------------------------------------------------

def test_drive_donor_threshold_favours_amp_b_at_exactly_half():
    sample_rate = 48000
    a = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    below_half = _select_donor(a, b, np.full(4, 0.499999), sample_rate)
    at_half = _select_donor(a, b, np.full(4, 0.5), sample_rate)
    above_half = _select_donor(a, b, np.full(4, 0.500001), sample_rate)
    assert np.allclose(below_half, a)  # below 50% B: Amp A is the donor
    assert np.allclose(at_half, b)     # AT exactly 50% B: Amp B is already the donor
    assert np.allclose(above_half, b)  # above 50% B: still Amp A -> Amp B is unaffected
    assert not np.allclose(at_half, 0.5 * (a + b))  # not a literal 50/50 waveform blend
