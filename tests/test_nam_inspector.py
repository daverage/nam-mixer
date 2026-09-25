import json

import numpy as np
import pytest

from hybrid.services import nam_inspector


def _model(architecture="WaveNet", *, sample_rate=48000, metadata=None, config=None):
    return {
        "version": "0.5.0", "architecture": architecture,
        "config": config if config is not None else {"layers": []},
        "weights": [], "sample_rate": sample_rate,
        "metadata": metadata or {},
    }


def _write(path, raw):
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_inspects_standard_wavenet_and_reports_calibration(monkeypatch, tmp_path):
    source = _write(tmp_path / "amp.nam", _model(metadata={
        "name": "Amp", "modeled_by": "Maker", "gear_type": "amp",
        "input_level_dbu": 12, "output_level_dbu": -6, "loudness": -18,
    }))
    monkeypatch.setattr(nam_inspector, "compute_source_nam_receptive_field", lambda model: 128)
    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.ones_like(audio) * 0.1)

    result = nam_inspector.inspect_nam(source)

    assert result["identity"]["name"] == "Amp"
    assert result["architecture"]["name"] == "WaveNet"
    assert result["architecture"]["sample_rate"] == 48000
    assert result["calibration"]["status"] == "complete"
    assert result["cabinet"]["status"] == "amp_only"
    assert result["temporal"]["neural_receptive_field_ms"] == pytest.approx(2.667)
    assert result["validation"]["status"] == "passed"
    assert result["validation"]["branches"]["render"]["peak_dbfs"] == pytest.approx(-20, abs=0.01)


def test_reports_input_only_and_unknown_cabinet_without_inference(monkeypatch, tmp_path):
    source = _write(tmp_path / "unknown.nam", _model(metadata={"input_level_dbu": 0}))
    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.ones_like(audio))
    result = nam_inspector.inspect_nam(source)
    assert result["calibration"]["status"] == "input-only"
    assert result["cabinet"]["status"] == "unknown"


def test_a2_checks_full_and_lite_independently(monkeypatch, tmp_path):
    source = _write(tmp_path / "a2.nam", _model("SlimmableContainer", config={"submodels": [
        {"max_value": 0.5, "model": {}}, {"max_value": 1.0, "model": {}}
    ]}))
    calls = []
    def fake_render(model, audio, sample_rate, slim=None):
        calls.append(slim)
        return np.ones_like(audio) * (0.2 if slim is None else 0.1)
    monkeypatch.setattr(nam_inspector, "render", fake_render)

    result = nam_inspector.inspect_nam(source)

    assert result["architecture"]["a2_packed"] is True
    assert result["architecture"]["full_lite_supported"] is True
    assert result["validation"]["branches"]["full"]["status"] == "passed"
    assert result["validation"]["branches"]["lite"]["status"] == "passed"
    assert calls == [None, 0.0]


def test_sequential_linear_stage_reports_ir_duration_and_metadata_evidence(monkeypatch, tmp_path):
    linear = {"architecture": "Linear", "sample_rate": 48000,
              "config": {"receptive_field": 240}, "weights": [0.1] * 240}
    source = _write(tmp_path / "sequential.nam", _model("Sequential", config={"models": [
        _model(), linear,
    ]}, metadata={"gear_type": "amp_cab"}))
    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.ones_like(audio))
    result = nam_inspector.inspect_nam(source)
    assert result["architecture"]["sequential"] is True
    assert result["architecture"]["stages"][1]["ir_length_samples"] == 240
    assert result["architecture"]["stages"][1]["duration_ms"] == 5
    assert result["cabinet"]["status"] == "embedded_linear"
    assert result["temporal"]["fir_history_samples"] == 239
    assert result["temporal"]["total_formal_dependency_samples"] == 240


def test_missing_sample_rate_skips_renderer_and_renderer_failure_is_distinct(monkeypatch, tmp_path):
    missing_rate = _write(tmp_path / "missing-rate.nam", _model(sample_rate=None))
    result = nam_inspector.inspect_nam(missing_rate)
    assert result["validation"]["branches"]["render"]["status"] == "not_run"
    assert "sample rate" in result["validation"]["branches"]["render"]["detail"]

    source = _write(tmp_path / "renderer-fail.nam", _model())
    def fail(*args, **kwargs):
        raise nam_inspector.NamRenderError("could not load model")
    monkeypatch.setattr(nam_inspector, "render", fail)
    result = nam_inspector.inspect_nam(source)
    assert result["validation"]["metadata_status"] == "passed"
    assert result["validation"]["status"] == "failed"
    assert result["validation"]["branches"]["render"]["detail"] == "could not load model"


@pytest.mark.parametrize("architecture", ["LSTM", "UnsupportedArchitecture"])
def test_architecture_is_reported_even_when_renderer_cannot_load(monkeypatch, tmp_path, architecture):
    source = _write(tmp_path / "architecture.nam", _model(architecture))
    def fail(*args, **kwargs):
        raise nam_inspector.NamRenderError("unsupported architecture")
    monkeypatch.setattr(nam_inspector, "render", fail)
    result = nam_inspector.inspect_nam(source)
    assert result["architecture"]["name"] == architecture
    assert result["validation"]["metadata_status"] == "passed"
    assert result["validation"]["branches"]["render"]["status"] == "failed"


def test_silent_nonfinite_and_invalid_json_cases(monkeypatch, tmp_path):
    silent = _write(tmp_path / "silent.nam", _model())
    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.zeros_like(audio))
    result = nam_inspector.inspect_nam(silent)
    assert result["validation"]["status"] == "warning"
    assert result["validation"]["branches"]["render"]["detail"] == "Silent output"

    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.full_like(audio, np.nan))
    result = nam_inspector.inspect_nam(silent)
    assert result["validation"]["branches"]["render"]["detail"] == "Non-finite output"
    assert result["validation"]["status"] == "failed"

    invalid = tmp_path / "bad.nam"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        nam_inspector.inspect_nam(invalid)

    non_nam = _write(tmp_path / "not-nam.nam", {"hello": "world"})
    result = nam_inspector.inspect_nam(non_nam)
    assert result["validation"]["metadata_status"] == "failed"
    assert result["validation"]["status"] == "failed"
    assert "incomplete" in result["validation"]["metadata_detail"]


def test_partial_stage_histories_never_produce_a_total(monkeypatch, tmp_path):
    """One malformed Linear stage (or an unknown stage) makes the history
    unknown rather than silently understated."""
    good = {"architecture": "Linear", "config": {"receptive_field": 100}, "weights": [0.0] * 100}
    bad = {"architecture": "Linear", "config": {"receptive_field": "bad"}, "weights": []}
    source = _write(tmp_path / "partial.nam", _model("Sequential", config={"models": [_model(), good, bad]}))
    monkeypatch.setattr(nam_inspector, "compute_source_nam_receptive_field", lambda model: 128)
    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.ones_like(audio))
    result = nam_inspector.inspect_nam(source)
    assert result["cabinet"]["status"] == "embedded_linear"
    assert result["temporal"]["fir_history_samples"] is None
    assert result["temporal"]["neural_receptive_field_samples"] is None
    assert result["temporal"]["total_formal_dependency_samples"] is None

    unknown_stage = _write(tmp_path / "unknown-stage.nam", _model("Sequential", config={"models": [
        _model(), {"architecture": "MysteryNet", "config": {}}]}))
    result = nam_inspector.inspect_nam(unknown_stage)
    assert result["temporal"]["total_formal_dependency_samples"] is None


def test_amp_pedal_cab_metadata_is_cabinet_evidence(monkeypatch, tmp_path):
    source = _write(tmp_path / "rig.nam", _model(metadata={"gear_type": "amp_pedal_cab"}))
    monkeypatch.setattr(nam_inspector, "render", lambda model, audio, sample_rate, slim=None: np.ones_like(audio))
    assert nam_inspector.inspect_nam(source)["cabinet"]["status"] == "metadata_indicates_cabinet"
