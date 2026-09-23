import json


from hybrid.core.nam_loader import load_nam


def test_calibrated_nam(tmp_path):
    p = tmp_path / "calibrated.nam"
    p.write_text(json.dumps({
        "architecture": "WaveNet",
        "sample_rate": 48000,
        "input_level_dbu": 12.0,
        "output_level_dbu": -6.0,
    }))
    model = load_nam(p)
    assert model.is_calibrated
    assert model.calibration_status == "Calibrated NAM"
    assert model.input_level_dbu == 12.0


def test_uncalibrated_nam_not_rejected(tmp_path):
    p = tmp_path / "old.nam"
    p.write_text(json.dumps({
        "architecture": "LSTM",
        "sample_rate": 44100,
    }))
    model = load_nam(p)
    assert not model.is_calibrated
    assert model.calibration_status == "Calibration metadata unavailable"


def test_nested_metadata_calibration(tmp_path):
    p = tmp_path / "nested.nam"
    p.write_text(json.dumps({
        "architecture": "ConvNet",
        "metadata": {"input_level_dbu": 10.0, "output_level_dbu": -2.0},
    }))
    model = load_nam(p)
    assert model.is_calibrated


def test_descriptive_metadata_accessors(tmp_path):
    p = tmp_path / "described.nam"
    p.write_text(json.dumps({
        "architecture": "WaveNet",
        "metadata": {
            "name": "British American High Gain", "modeled_by": "Someone",
            "gear_type": "amp", "gear_make": "Marshall", "gear_model": "JCM800",
            "tone_type": "hi_gain",
        },
    }))
    model = load_nam(p)
    assert model.name == "British American High Gain"
    assert model.modeled_by == "Someone"
    assert model.gear_type == "amp"
    assert model.gear_make == "Marshall"
    assert model.gear_model == "JCM800"
    assert model.tone_type == "hi_gain"


def test_descriptive_metadata_accessors_default_to_none(tmp_path):
    p = tmp_path / "bare.nam"
    p.write_text(json.dumps({"architecture": "WaveNet"}))
    model = load_nam(p)
    assert model.name is None
    assert model.modeled_by is None
    assert model.gear_type is None
    assert model.gear_make is None
    assert model.gear_model is None
    assert model.tone_type is None


def test_input_only_calibration_is_reported_as_usable(tmp_path):
    """Auto calibration only needs input_level_dbu, so an input-only file must
    not be labelled as having no calibration metadata."""
    p = tmp_path / "input_only.nam"
    p.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000, "input_level_dbu": 12.0}))
    model = load_nam(p)
    assert not model.is_calibrated
    assert model.calibration_status == "Input level only (enough for Auto calibration)"

