import json
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid.modes.character_analysis import CharacterAnalysisConfig, analyse_rendered_audio
from hybrid.modes.character_blend import (
    CharacterBlendDesign,
    LowLevelResponseCheck,
    _adjacent_level_weights,
    _continuous_drive_donor,
    _causal_donor_weight,
    _soft_donor_weight,
    character_temporal_history_samples,
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


def test_full_character_teacher_is_prefix_invariant_with_frozen_analysis():
    """Offline full-clip analysis is frozen before testing runtime causality."""
    full = _pair(n=4096, sample_rate=48000)
    configured = CharacterBlendDesign(
        "a.nam", "b.nam", tone_mix_b=.35, feel_mix_b=.65,
        drive_low_mix_b=.2, drive_mid_mix_b=.8, drive_high_mix_b=.3,
        envelope_smoothing_ms=4.0,
    )
    analysed = build_character_blend(full, configured)
    frozen = freeze_character_design(
        full, analysed, "a.nam", "b.nam", tone_mix_b=.35,
        feel_mix_b=.65, drive_low_mix_b=.2, drive_mid_mix_b=.8,
        drive_high_mix_b=.3, envelope_smoothing_ms=4.0,
    )
    boundary = 2048
    prefix_pair = SimpleNamespace(
        **{**full.__dict__, "dry": full.dry[:boundary], "amp_a": full.amp_a[:boundary], "amp_b": full.amp_b[:boundary]}
    )
    changed = _pair(n=4096, sample_rate=48000)
    changed.dry[:boundary] = full.dry[:boundary]
    changed.amp_a[:boundary] = full.amp_a[:boundary]
    changed.amp_b[:boundary] = full.amp_b[:boundary]
    changed.dry[boundary:] *= -0.7
    changed.amp_a[boundary:] = 0.75
    changed.amp_b[boundary:] = -0.5

    short = build_character_blend(prefix_pair, frozen).blend
    long = build_character_blend(changed, frozen).blend[:boundary]

    np.testing.assert_allclose(short, long, rtol=2e-6, atol=2e-7)
    assert np.array_equal(
        build_character_blend(changed, frozen).blend,
        build_character_blend(changed, frozen).blend,
    )


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
# above the analysis grid (docs/history/blend-mode-fixes.md).
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
    import hybrid.modes.character_blend as character_blend_module

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
# Character teacher v3 -- Drive is continuous through a soft donor region.
# ---------------------------------------------------------------------------

def test_drive_donor_v3_is_continuous_and_preserves_exact_endpoints():
    sample_rate = 48000
    a = np.full(1001, 1.0, dtype=np.float64)
    b = np.full(1001, 4.0, dtype=np.float64)
    drive = np.linspace(0.0, 1.0, len(a))
    output = _select_donor(a, b, drive, sample_rate)
    assert output[0] == pytest.approx(1.0)
    assert output[-1] == pytest.approx(4.0)
    assert np.max(np.abs(np.diff(output))) < 0.02
    assert output[500] != pytest.approx(0.5 * (a[500] + b[500]))


def test_soft_donor_weight_has_flat_exact_regions_and_smooth_midpoint():
    weights = _soft_donor_weight(np.array([0.0, 0.35, 0.5, 0.65, 1.0]))
    np.testing.assert_allclose(weights, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_continuous_drive_residual_is_bounded_for_dissimilar_amps():
    a = np.ones(4096)
    b = np.full(4096, 10.0)
    midpoint = _continuous_drive_donor(a, b, np.full(4096, 0.5), 48000)
    assert np.isfinite(midpoint).all()
    assert not np.allclose(midpoint, 0.5 * (a + b))


def test_continuous_drive_residual_is_prefix_invariant():
    rng = np.random.default_rng(42)
    a = rng.normal(0.0, 0.1, 4096)
    b = np.tanh(a * 8.0)
    changed_a, changed_b = a.copy(), b.copy()
    changed_a[2048:] = 0.8
    changed_b[2048:] = -0.8
    weight = np.full(4096, 0.5)
    short = _continuous_drive_donor(a[:2048], b[:2048], weight[:2048], 48000)
    long = _continuous_drive_donor(changed_a, changed_b, weight, 48000)
    np.testing.assert_allclose(short, long[:2048], rtol=0.0, atol=0.0)


def test_v2_threshold_donor_remains_reproducible():
    a = np.ones(4)
    b = np.full(4, 2.0)
    below = _select_donor(a, b, np.full(4, 0.499999), 48000, semantics_version=2)
    at = _select_donor(a, b, np.full(4, 0.5), 48000, semantics_version=2)
    assert np.allclose(below, a)
    assert np.allclose(at, b)


def test_drive_donor_transition_is_prefix_invariant_and_causal():
    """A future switch must never rewrite already-emitted teacher samples."""
    sample_rate = 1000
    a = np.zeros(100, dtype=np.float64)
    b = np.ones(100, dtype=np.float64)
    before_switch = np.zeros(100, dtype=np.float64)
    switches_at_50 = before_switch.copy()
    switches_at_50[50:] = 1.0
    assert np.array_equal(
        _select_donor(a, b, before_switch, sample_rate, semantics_version=2)[:50],
        _select_donor(a, b, switches_at_50, sample_rate, semantics_version=2)[:50],
    )
    # The first affected sample is the switch itself, never a centred pre-fade.
    assert _select_donor(a, b, switches_at_50, sample_rate, semantics_version=2)[49] == 0.0
    assert _select_donor(a, b, switches_at_50, sample_rate, semantics_version=2)[50] > 0.0


def test_drive_donor_causal_ramp_handles_reversal_without_a_jump():
    drive = np.zeros(40)
    drive[5:12] = 1.0
    weight = _causal_donor_weight(drive, 1000)
    assert np.all((0.0 <= weight) & (weight <= 1.0))
    assert weight[0] == 0.0
    assert weight[5] > 0.0
    assert 0.0 < weight[11] < 1.0
    # The new ramp starts from the current weight, rather than snapping to A.
    assert abs(weight[12] - weight[11]) <= 0.1
    assert weight[-1] == 0.0


@pytest.mark.parametrize("initial,target", [(0.0, 1.0), (1.0, 0.0)])
def test_drive_donor_settles_within_documented_bound(initial, target):
    sample_rate = 1000
    ramp_samples = 10
    drive = np.full(2 + ramp_samples + 2, initial)
    drive[2:] = target
    weight = _causal_donor_weight(drive, sample_rate)
    assert weight[1] == initial
    assert weight[2 + ramp_samples - 1] == target
    assert np.all(weight[2 + ramp_samples - 1:] == target)
    assert np.all((weight >= 0.0) & (weight <= 1.0))


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_drive_donor_constant_and_short_clip_boundaries(value, length):
    drive = np.full(length, value)
    weight = _causal_donor_weight(drive, 1000)
    assert len(weight) == length
    assert np.all(weight == value)


def test_unknown_teacher_semantics_are_rejected():
    with pytest.raises(ValueError, match="unsupported Character Blend teacher semantics"):
        _select_donor(np.zeros(4), np.ones(4), np.zeros(4), 1000, semantics_version=99)


def test_character_design_without_semantics_version_loads_as_legacy(tmp_path):
    path = tmp_path / "legacy-character.json"
    path.write_text('{"amp_a_path":"a.nam","amp_b_path":"b.nam"}')
    assert CharacterBlendDesign.read_json(path).teacher_semantics_version == 1


def test_character_temporal_history_reports_serial_dependencies():
    history = character_temporal_history_samples(1000, 40.0, semantics_version=2)
    assert history == {
        "teacher_semantics_version": 2,
        "drive_smoothing_serial_samples": 39,
        "compensation_smoothing_serial_samples": 39,
        "donor_transition_serial_samples": 9,
        "correction_fir_serial_samples": 64,
        "donor_transition_exact_history_bounded": False,
    }


def test_character_v3_temporal_history_has_no_donor_transition_and_is_bounded():
    assert character_temporal_history_samples(1000, 40.0, semantics_version=3) == {
        "teacher_semantics_version": 3,
        "drive_smoothing_serial_samples": 39,
        "compensation_smoothing_serial_samples": 39,
        "correction_fir_serial_samples": 64,
        "donor_transition_exact_history_bounded": True,
    }


def test_preview_pair_is_measured_against_profiled_dry_not_raw_di():
    """A preview RenderedPair's amps were driven by profiled_dry; the teacher
    must match a generation-style pair whose `dry` IS that profiled signal."""
    base = _pair()
    profiled = (base.dry * 2.0).astype(np.float32)  # +6 dB pickup profile
    preview = SimpleNamespace(**{**vars(base), "dry": base.dry, "profiled_dry": profiled})
    generation_style = SimpleNamespace(**{**vars(base), "dry": profiled})
    design = CharacterBlendDesign("a.nam", "b.nam", tone_mix_b=.4, feel_mix_b=.6, drive_mix_b=.5)
    assert np.array_equal(build_character_blend(preview, design).blend,
                          build_character_blend(generation_style, design).blend)


def test_analysis_cache_key_changes_with_the_measured_audio():
    from hybrid.modes.character_analysis import analysis_cache_key

    pair = _pair()
    key = analysis_cache_key("nam", pair.dry, pair.amp_a)
    assert key == analysis_cache_key("nam", pair.dry.copy(), pair.amp_a.copy())
    assert key != analysis_cache_key("nam", pair.dry * 2.0, pair.amp_a)   # different DI level/profile
    assert key != analysis_cache_key("nam", pair.dry, pair.amp_a * 0.5)   # different render (e.g. input trim)
    assert key != analysis_cache_key("other", pair.dry, pair.amp_a)


@pytest.mark.parametrize("window_db", [3.0, 2.0])
def test_frozen_design_records_and_restores_the_analysis_level_window(tmp_path, window_db):
    """Default and non-default windows survive freeze -> JSON -> load, and the
    restored config reproduces the analyses' own config hash exactly."""
    pair = _pair()
    config = CharacterAnalysisConfig(level_window_db=window_db)
    analysis_a = analyse_rendered_audio(pair.dry, pair.amp_a, pair.sample_rate, config)
    analysis_b = analyse_rendered_audio(pair.dry, pair.amp_b, pair.sample_rate, config)
    result = build_character_blend(pair, CharacterBlendDesign("a.nam", "b.nam"), analysis_a=analysis_a, analysis_b=analysis_b)

    frozen = freeze_character_design(pair, result, "a.nam", "b.nam")
    loaded = CharacterBlendDesign.read_json(frozen.write_json(tmp_path / "character.json"))

    restored = CharacterAnalysisConfig(**loaded.analysis_config)
    assert restored.level_window_db == window_db
    assert restored.cache_key() == analysis_a.config_hash == analysis_b.config_hash


def test_legacy_frozen_design_without_level_window_keeps_its_behaviour():
    """Designs frozen before the window was recorded: analyses without the
    field load as the default window, and the teacher is unchanged."""
    pair = _pair()
    result = build_character_blend(pair, CharacterBlendDesign("a.nam", "b.nam"))
    current = freeze_character_design(pair, result, "a.nam", "b.nam").to_dict()

    legacy = json.loads(json.dumps(current))
    for key in ("analysis_a", "analysis_b"):
        del legacy[key]["level_window_db"]
    del legacy["analysis_config"]["level_window_db"]
    legacy_design = CharacterBlendDesign(**legacy)

    assert CharacterAnalysisConfig(**legacy_design.analysis_config).level_window_db == 3.0
    assert np.array_equal(build_character_blend(pair, legacy_design).blend,
                          build_character_blend(pair, CharacterBlendDesign(**current)).blend)


def _dense_reference_mix(signal, firs, levels, envelope):
    """The per-level mix build_character_blend used before streaming: every
    filtered path stacked and summed against the dense weight matrix."""
    from scipy.signal import fftconvolve

    n = len(signal)
    filtered = [fftconvolve(signal, fir, mode="full")[:n] for fir in firs]
    return np.sum(np.vstack(filtered) * _adjacent_level_weights(levels, envelope), axis=0)


def _level_firs(n_levels, sample_rate=48000):
    from hybrid.modes.character_blend import _minimum_phase_correction

    freqs = np.geomspace(80.0, 10_000.0, 24)
    rng = np.random.default_rng(7)
    return [_minimum_phase_correction(freqs, rng.uniform(-4, 4, len(freqs)), sample_rate) for _ in range(n_levels)]


@pytest.mark.parametrize("case", ["sweep", "on_grid_points", "below_and_above_grid", "constant_inside", "tiny"])
def test_streamed_level_mix_is_bit_identical_to_the_dense_sum(case):
    from hybrid.modes.character_blend import _mix_adjacent_filtered_levels

    levels = np.array([-24.0, -18.0, -12.0, -6.0, 0.0, 6.0])
    rng = np.random.default_rng(3)
    n = 5 if case == "tiny" else 20_000
    t = np.arange(n) / 48000
    signal = 0.3 * np.sin(2 * np.pi * 196 * t) + 0.05 * rng.standard_normal(n)
    envelope = {
        "sweep": np.linspace(-40.0, 15.0, n),                        # crosses every level and both edges
        "on_grid_points": np.resize(levels, n),                     # exactly on each grid point
        "below_and_above_grid": np.where(np.arange(n) % 2, -80.0, 30.0),  # clamped at both ends
        "constant_inside": np.full(n, -9.5),                         # never touches most levels
        "tiny": np.array([-30.0, -18.0, -3.0, 6.0, 10.0]),
    }[case]
    firs = _level_firs(len(levels))

    streamed = _mix_adjacent_filtered_levels(signal, firs, levels, envelope)
    dense = _dense_reference_mix(signal, firs, levels, envelope)
    assert np.array_equal(streamed, dense)  # exact float64 equality, not approximate


def _alternating_level_sine(seconds=6, sample_rate=48000):
    """A pure 220 Hz tone alternating between -12 and -30 dB RMS every 150 ms:
    the samples near either level form separate runs, so stitching them
    creates seams, while any contiguous stretch has almost no HF energy."""
    t = np.arange(sample_rate * seconds) / sample_rate
    gain = np.where((np.arange(len(t)) // int(0.150 * sample_rate)) % 2 == 0, 10 ** (-9 / 20), 10 ** (-27 / 20))
    return np.sin(2 * np.pi * 220 * t) * gain


def _hf_below_peak_db(analysis, level_db=-12.0):
    freqs = np.array(analysis.frequencies_hz)
    spectrum = np.array(next(x for x in analysis.levels if x.input_gain_db == level_db).spectrum_db)
    return float(np.mean(spectrum[freqs >= 2000]) - spectrum.max())


def test_v1_spectrum_shows_the_stitching_artefact_and_v2_does_not():
    x = _alternating_level_sine()
    stitched = analyse_rendered_audio(x, x, 48000, CharacterAnalysisConfig(version=1))
    contiguous = analyse_rendered_audio(x, x, 48000, CharacterAnalysisConfig(version=2))
    assert _hf_below_peak_db(stitched) > -50.0      # seams put HF energy into a pure tone
    assert _hf_below_peak_db(contiguous) < -75.0    # contiguous frames do not
    assert contiguous.version == 2 and stitched.version == 1


def test_default_analysis_is_still_the_version_1_method():
    x = _alternating_level_sine(seconds=2)
    y = np.tanh(3.0 * x)
    default = analyse_rendered_audio(x, y, 48000)
    assert CharacterAnalysisConfig().version == 1
    assert default == analyse_rendered_audio(x, y, 48000, CharacterAnalysisConfig(version=1))


def test_analyses_of_different_versions_cannot_be_mixed():
    pair = _pair()
    a = analyse_rendered_audio(pair.dry, pair.amp_a, pair.sample_rate, CharacterAnalysisConfig(version=1))
    b = analyse_rendered_audio(pair.dry, pair.amp_b, pair.sample_rate, CharacterAnalysisConfig(version=2))
    with pytest.raises(ValueError, match="different Character analysis versions"):
        build_character_blend(pair, CharacterBlendDesign("a.nam", "b.nam"), analysis_a=a, analysis_b=b)


def test_version_2_design_records_and_restores_its_analysis_version(tmp_path):
    pair = _pair(n=48000)
    config = CharacterAnalysisConfig(version=2)
    a = analyse_rendered_audio(pair.dry, pair.amp_a, pair.sample_rate, config)
    b = analyse_rendered_audio(pair.dry, pair.amp_b, pair.sample_rate, config)
    result = build_character_blend(pair, CharacterBlendDesign("a.nam", "b.nam"), analysis_a=a, analysis_b=b)
    loaded = CharacterBlendDesign.read_json(freeze_character_design(pair, result, "a.nam", "b.nam").write_json(tmp_path / "d.json"))
    assert loaded.analysis_config["version"] == 2 and loaded.analysis_a["version"] == 2
    assert CharacterAnalysisConfig(**loaded.analysis_config).cache_key() == a.config_hash
    assert np.array_equal(build_character_blend(pair, loaded).blend, result.blend)
