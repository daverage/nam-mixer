import json

import pytest

from hybrid.nam_loader import load_nam


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
