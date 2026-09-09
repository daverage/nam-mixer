import pytest

from hybrid.nam_tools import NamToolError, apply_metadata_changes, apply_volume_change, calculate_gain_multiplier, compare_changes, describe_nam_tools


def slimmable(count=2):
    return {"architecture": "SlimmableContainer", "config": {"submodels": [
        {"max_value": i + 1, "model": {"config": {"head_scale": 0.0051461088670930214, "weights": [1, 2]}, "metadata": {"loudness": -22.8 - i}}}
        for i in range(count)]}, "metadata": {"loudness": -22.7, "gain": 4.0}}


def test_slimmable_volume_edits_only_approved_paths():
    original = slimmable()
    edited, paths, multiplier = apply_volume_change(original, 6)
    assert multiplier == pytest.approx(1.9952623149688795)
    assert edited["config"]["submodels"][0]["model"]["config"]["head_scale"] == pytest.approx(0.01026783)
    assert edited["metadata"]["gain"] == 4.0
    assert compare_changes(original, edited) == paths


@pytest.mark.parametrize(("db", "expected"), [(6, 1.9952623149688795), (-6, 0.5011872336272722), (3.5, 1.4962356560944334)])
def test_gain_multiplier(db, expected):
    assert calculate_gain_multiplier(db) == pytest.approx(expected)


def test_single_model_and_missing_loudness():
    source = {"architecture": "WaveNet", "config": {"head_scale": 2.0}, "metadata": {"gain": 1}}
    edited, paths, _ = apply_volume_change(source, -6)
    assert edited["config"]["head_scale"] == pytest.approx(1.0023744672545444)
    assert paths == ["config.head_scale"]


def test_unknown_model_is_refused():
    with pytest.raises(NamToolError, match="ambiguous"):
        apply_volume_change({"architecture": "Unknown", "config": {"nested": {"head_scale": 1}}}, 6)


def test_metadata_editor_cannot_change_model_content():
    source = slimmable(1)
    edited, paths = apply_metadata_changes(source, {"name": "Battery", "modeled_by": "NAM user", "gear_type": "amp", "gear_make": "Mesa"})
    assert paths == ["metadata.name", "metadata.modeled_by", "metadata.gear_type", "metadata.gear_make"]
    assert compare_changes(source, edited) == paths


def test_metadata_editor_refuses_calibration_fields():
    with pytest.raises(NamToolError, match="only permits"):
        apply_metadata_changes(slimmable(1), {"input_level_dbu": 12.0})


def test_editor_reports_current_loudness_and_standard_metadata():
    source = slimmable(1)
    source["metadata"].update({"name": "Battery", "gear_type": "amp"})
    source.update({"input_level_dbu": 12.0, "output_level_dbu": -3.0})
    details = describe_nam_tools(source)
    assert details["loudness_db"] == -22.7
    assert details["metadata"] == {"name": "Battery", "gear_type": "amp"}
    assert details["calibration"] == {"input_level_dbu": 12.0, "output_level_dbu": -3.0, "status": "Calibrated NAM"}
