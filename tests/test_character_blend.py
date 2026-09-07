from types import SimpleNamespace

import numpy as np

from hybrid.character_analysis import CharacterAnalysisConfig, analyse_rendered_audio
from hybrid.character_blend import CharacterBlendDesign, build_character_blend, freeze_character_design


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
