"""Continuous Gain Stage 4 validation (hybrid/continuous_gain/validation.py) with the renderer faked.

A "perfect" trained model here renders exactly the capture's output times the training scale c, so the
comparison must report no difference. With a learned cab the trained model contains the cabinet, so the
comparison is like-with-like only when the captures go through the same cabinet.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest
import soundfile as sf

import hybrid.continuous_gain.validation as cgv
from hybrid.core.render import SLIM_FULL, SLIM_LITE, NamRenderError
from tests.cg_synth import SR, amp_render, synth_di

SCALE_C = 0.5
CAB = np.array([0.5, 0.3, 0.2], dtype=np.float32)          # a dull "cabinet": clearly changes the HF balance


def cab_fn(y):
    return np.convolve(y, CAB)[: len(y)].astype(np.float32)


def capture(gain):
    return types.SimpleNamespace(kind="capture", gain=gain)


def trained_model(with_cab=False):
    return types.SimpleNamespace(kind="model", with_cab=with_cab)


@pytest.fixture()
def fake_render(monkeypatch):
    """Captures are memoryless amps; the trained model reproduces capture 2 (the plan maps every position
    to 0 dB Input gain below) scaled by c, optionally through the learned cabinet."""
    calls = []

    def render(model, x, sr, **kwargs):
        calls.append((model.kind, kwargs.get("slim")))
        if model.kind == "capture":
            return amp_render(model.gain)(x)
        y = amp_render(2.0)(x) * SCALE_C
        return cab_fn(y) if model.with_cab else y
    monkeypatch.setattr(cgv, "render", render)
    return calls


def short_di(name):
    return synth_di(name, 2.0)


PLAN = [{"position": 2.0, "input_gain_db": 0.0}]


def test_progression_reports_no_difference_for_a_model_that_matches_the_capture(fake_render):
    rows = cgv.check_progression(trained_model(), SCALE_C, {2.0: capture(2.0)}, {2.0: 0}, PLAN, {2.0}, load_di=short_di)
    row = rows["positions"][0]
    assert row["role"] == "training"
    for key in ("level_db", "hf_db", "crest_db", "eq_max_db"):
        assert abs(row[key]) < 1e-6, (key, row[key])
    assert row["lm_esr"] < 1e-9 and rows["reversals"] == []


def test_learned_cab_model_is_compared_with_the_captures_through_the_same_cabinet(fake_render):
    model = trained_model(with_cab=True)
    like_for_like = cgv.check_progression(model, SCALE_C, {2.0: capture(2.0)}, {2.0: 0}, PLAN, {2.0},
                                          load_di=short_di, cab_fn=cab_fn)["positions"][0]
    without_cab = cgv.check_progression(model, SCALE_C, {2.0: capture(2.0)}, {2.0: 0}, PLAN, {2.0},
                                        load_di=short_di)["positions"][0]
    assert abs(like_for_like["hf_db"]) < 1e-6 and like_for_like["lm_esr"] < 1e-9
    # Comparing a cab model with cabless captures would report the cabinet itself as model error.
    assert abs(without_cab["hf_db"]) > 0.2 and without_cab["lm_esr"] > 1e-5


def test_reference_positions_are_reported_separately_from_training_anchors(fake_render):
    plan = [{"position": 2.0, "input_gain_db": 0.0}, {"position": 3.0, "input_gain_db": 0.0}]
    result = cgv.check_progression(trained_model(), SCALE_C, {2.0: capture(2.0), 3.0: capture(3.0)}, {}, plan, {2.0},
                                   load_di=short_di)
    roles = {r["position"]: r["role"] for r in result["positions"]}
    assert roles == {2.0: "training", 3.0: "reference"}
    assert set(result["summary"]) == {"training", "reference"}
    assert abs(result["summary"]["reference"]["level_db"]["mean_abs"]) > 0.1      # G3 is louder than the model of G2


def test_audition_writes_matching_model_and_original_clips(fake_render, tmp_path):
    out = cgv.write_audition(tmp_path, trained_model(with_cab=True), SCALE_C, {2.0: capture(2.0)}, {2.0: 0}, PLAN,
                             load_di=short_di, seconds=1, cab_fn=cab_fn)
    assert (tmp_path / "sweep.wav").is_file()
    assert out["sweep"]["input_gains_db"] == list(cgv.SWEEP_GAINS_DB)
    [clip] = out["comparisons"]
    model_clip, _ = sf.read(tmp_path / clip["model"], dtype="float32")
    original_clip, _ = sf.read(tmp_path / clip["original"], dtype="float32")
    # Same attenuation on both sides, and the original went through the same learned cabinet (24-bit files).
    np.testing.assert_allclose(model_clip, original_clip, atol=1e-5)


def test_audition_attenuates_a_hot_sweep_instead_of_limiting_it(fake_render, tmp_path, monkeypatch):
    out = cgv.write_audition(tmp_path, trained_model(), 0.05, {2.0: capture(2.0)}, {}, PLAN, load_di=short_di, seconds=1)
    assert out["sweep"]["attenuated_db"] > 0
    sweep, _ = sf.read(tmp_path / "sweep.wav", dtype="float32")
    assert 20 * np.log10(np.max(np.abs(sweep))) == pytest.approx(-1.0, abs=0.01)


def test_safety_sweep_renders_each_input_gain_once_and_reports_the_output_gain(fake_render):
    x = short_di("safety")
    result = cgv.check_safety(trained_model(), SCALE_C, x, gains_db=(-6, 0, 6))
    assert len(fake_render) == 3 and [r["input_gain_db"] for r in result["sweep"]] == [-6, 0, 6]
    assert result["recommended_output_gain_db"] == pytest.approx(-20 * np.log10(SCALE_C))
    assert result["finite"] is True


def _nam_file(path, **overrides):
    raw = {"architecture": "WaveNet", "version": "0.5.4", "sample_rate": SR, "config": {}, "weights": [0.0],
           "metadata": {"gear_type": "amp"}, **overrides}
    path.write_text(json.dumps(raw))
    return path


def test_compatibility_checks_full_and_lite_and_reports_a_lite_failure(tmp_path, monkeypatch):
    def render(model, x, sr, slim=None, **_):
        if slim == SLIM_LITE:
            raise NamRenderError("no Lite submodel")
        return np.zeros_like(x)
    monkeypatch.setattr(cgv, "render", render)
    monkeypatch.setattr(cgv, "load_nam", lambda p: object())
    result = cgv.check_compatibility(_nam_file(tmp_path / "m.nam"), np.zeros(4800, np.float32))
    assert result["full"]["rendered_ok"] is True and result["lite"]["rendered_ok"] is False
    assert "no Lite submodel" in result["lite"]["error"]
    assert result["standard_nam"] is True                       # judged on the Full render
    assert SLIM_FULL != SLIM_LITE


@pytest.mark.parametrize("render_output, overrides, standard", [
    (lambda x: np.full_like(x, np.nan), {}, False),             # non-finite output
    (lambda x: x[:-1], {}, False),                              # wrong length
    (lambda x: np.zeros_like(x), {"sample_rate": 44100}, False),  # not 48 kHz
    (lambda x: np.zeros_like(x), {"weights": None}, False),     # no weights
])
def test_compatibility_rejects_broken_models(tmp_path, monkeypatch, render_output, overrides, standard):
    monkeypatch.setattr(cgv, "render", lambda model, x, sr, **_: render_output(x))
    monkeypatch.setattr(cgv, "load_nam", lambda p: object())
    result = cgv.check_compatibility(_nam_file(tmp_path / "m.nam", **overrides), np.zeros(4800, np.float32))
    assert result["standard_nam"] is standard
