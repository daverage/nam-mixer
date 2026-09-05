from hybrid.calibration import input_calibration_gain_db, resolve_calibration


def test_official_compensation_formula():
    assert input_calibration_gain_db(reference_input_level_dbu=12.0, model_input_level_dbu=8.0) == 4.0


def test_auto_mode_both_calibrated_applies_per_model_compensation():
    result = resolve_calibration("auto", 12.0, amp_a_input_level_dbu=8.0, amp_b_input_level_dbu=12.0)
    assert result.applied is True
    assert result.amp_a_gain_db == 4.0
    assert result.amp_b_gain_db == 0.0
    assert result.warning is None


def test_auto_mode_both_uncalibrated_is_raw_with_no_warning():
    result = resolve_calibration("auto", 12.0, amp_a_input_level_dbu=None, amp_b_input_level_dbu=None)
    assert result.applied is False
    assert result.amp_a_gain_db == 0.0
    assert result.amp_b_gain_db == 0.0
    assert result.warning is None


def test_auto_mode_only_one_calibrated_falls_back_to_raw_with_warning():
    result = resolve_calibration("auto", 12.0, amp_a_input_level_dbu=8.0, amp_b_input_level_dbu=None)
    assert result.applied is False
    assert result.amp_a_gain_db == 0.0
    assert result.amp_b_gain_db == 0.0
    assert result.warning is not None
    assert "one or both" in result.warning.lower()


def test_explicit_raw_mode_never_compensates_even_if_both_calibrated():
    result = resolve_calibration("raw", 12.0, amp_a_input_level_dbu=8.0, amp_b_input_level_dbu=12.0)
    assert result.mode == "raw"
    assert result.applied is False
    assert result.amp_a_gain_db == 0.0
    assert result.amp_b_gain_db == 0.0
