"""Tests for hybrid/validation.py -- docs/phase3.md sections 24-28.

Uses a fake identity render() (same pattern as other pipeline tests) so
these run without the native nam_render tool or real .nam captures.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import hybrid.validation as validation
from hybrid.design import HybridDesign
from hybrid.validation import (
    compute_esr_metrics, load_frozen_design, render_processed_reference,
    render_reference_hybrid, render_reference_blend, render_reference_character,
)
from hybrid.fixed_blend import BlendDesign
from hybrid.character_blend import CharacterBlendDesign
from hybrid.validation_report import build_validation_report


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate, slim=None):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(validation, "render", fake_render)


def _write_nam(path, input_level_dbu=None):
    raw = {"architecture": "WaveNet", "sample_rate": 48000.0}
    if input_level_dbu is not None:
        raw["input_level_dbu"] = input_level_dbu
        raw["output_level_dbu"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _design(amp_a_path, amp_b_path, **overrides):
    fields = dict(
        amp_a_path=str(amp_a_path),
        amp_b_path=str(amp_b_path),
        crossover_dbfs=-20.0,
        transition_width_db=8.0,
        effective_b_trim_db=2.0,
        calibration_mode="raw",
        alignment_enabled=False,
    )
    fields.update(overrides)
    return HybridDesign(**fields)


def test_render_reference_hybrid_matches_frozen_trim_and_crossover(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    design = _design(amp_a, amp_b, crossover_dbfs=-100.0, transition_width_db=2.0, effective_b_trim_db=6.0)

    rng = np.random.default_rng(0)
    dry = (0.3 * rng.uniform(-1, 1, 48000)).astype(np.float32)

    result = render_reference_hybrid(design, dry, 48000)

    # Crossover set far below any real signal level -> should be fully Amp B,
    # scaled by the frozen +6dB trim (identity render means amp_b == dry).
    expected_gain = 10.0 ** (6.0 / 20.0)
    np.testing.assert_allclose(result.hybrid, dry * expected_gain, atol=1e-5)
    assert result.alignment_offset_samples == 0


def test_render_reference_hybrid_uses_per_model_calibration(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam", input_level_dbu=8.0)
    amp_b = _write_nam(tmp_path / "b.nam", input_level_dbu=12.0)
    design = _design(
        amp_a, amp_b, crossover_dbfs=-100.0, effective_b_trim_db=0.0,
        calibration_mode="auto", reference_input_level_dbu=12.0,
    )
    dry = np.full(4800, 0.2, dtype=np.float32)
    result = render_reference_hybrid(design, dry, 48000)
    # Amp B needs 0dB compensation (already at reference) -> amp_b render == dry.
    np.testing.assert_allclose(result.amp_b, dry, atol=1e-6)
    # Amp A needs +4dB compensation -> its render is louder than dry.
    assert np.max(np.abs(result.amp_a)) > np.max(np.abs(dry))


def test_frozen_blend_and_character_teachers_render_without_current_ui_state(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    dry = np.full(4800, .1, dtype=np.float32)
    blend = render_reference_blend(BlendDesign(str(amp_a), str(amp_b), mix_b=.25, calibration_mode="raw"), dry, 48000)
    character = render_reference_character(CharacterBlendDesign(str(amp_a), str(amp_b), calibration_mode="raw"), dry, 48000)
    assert len(blend.hybrid) == len(character.hybrid) == len(dry)
    assert np.isfinite(blend.hybrid).all() and np.isfinite(character.hybrid).all()


def test_compute_esr_metrics_zero_for_identical_signals():
    a = np.full(1000, 0.5, dtype=np.float32)
    metrics = compute_esr_metrics(a, a)
    assert metrics["raw_esr"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["gain_normalized_esr"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rms_difference"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["peak_difference"] == pytest.approx(0.0, abs=1e-9)


def test_compute_esr_metrics_distinguishes_gain_from_shape_error():
    """A pure gain difference should show up in raw ESR but vanish after
    gain normalization; a shape difference should survive normalization."""
    rng = np.random.default_rng(1)
    reference = rng.uniform(-0.5, 0.5, 2000).astype(np.float64)

    louder = reference * 2.0
    gain_metrics = compute_esr_metrics(louder, reference)
    assert gain_metrics["raw_esr"] > 0.1
    assert gain_metrics["gain_normalized_esr"] == pytest.approx(0.0, abs=1e-9)

    distorted = reference + rng.uniform(-0.1, 0.1, 2000)
    shape_metrics = compute_esr_metrics(distorted, reference)
    assert shape_metrics["gain_normalized_esr"] > 0.0


def test_processed_reference_replays_fixed_output_and_safety_gains_once(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    design = _design(
        amp_a, amp_b, crossover_dbfs=-100.0, transition_width_db=2.0,
        effective_b_trim_db=0.0,
    )
    design.write_json(tmp_path / "hybrid_design.json")
    manifest = {
        "mode": "hybrid",
        "output_gain": {"applied_gain_db": 6.0},
        "target": {"global_safety_gain_reduction_db": 2.0},
    }
    frozen = load_frozen_design(tmp_path, manifest)
    dry = np.full(1000, 0.1, dtype=np.float32)

    result = render_processed_reference(frozen, manifest, dry, 48000)

    np.testing.assert_allclose(result.hybrid, dry * (10.0 ** (4.0 / 20.0)), atol=1e-6)


def test_processed_reference_applies_baked_cab_only_in_teacher_post_stage(monkeypatch, tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    from hybrid.cab_ir import CabDesign
    design = _design(
        amp_a, amp_b,
        cab=CabDesign(selected=True, baked=True, ir_working_path=str(tmp_path / "cab.wav")),
    )
    dry = np.full(16, 0.1, dtype=np.float32)
    base = validation.ReferenceHybridResult(dry.copy(), dry, dry, dry, 0)
    calls = []
    monkeypatch.setattr(validation, "render_reference_hybrid", lambda *args: base)
    monkeypatch.setattr(validation, "get_prepared_cab_ir", lambda path, sr: (path, sr))
    def fake_cab(audio, prepared):
        calls.append(prepared)
        return audio * 3.0
    monkeypatch.setattr(validation, "apply_cab_ir", fake_cab)

    result = render_processed_reference(design, {
        "mode": "hybrid", "output_gain": {"applied_gain_db": 6.0},
        "target": {"global_safety_gain_reduction_db": 2.0},
    }, dry, 48000)

    assert calls == [(str(tmp_path / "cab.wav"), 48000)]
    np.testing.assert_allclose(result.hybrid, dry * 3.0 * (10.0 ** (4.0 / 20.0)), atol=1e-6)


def test_load_frozen_design_requires_mode_specific_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="blend design"):
        load_frozen_design(tmp_path, {"mode": "blend"})


@pytest.mark.parametrize("mode,filename,design", [
    ("hybrid", "hybrid_design.json", HybridDesign("a.nam", "b.nam", crossover_dbfs=-20.0)),
    ("blend", "blend_design.json", BlendDesign("a.nam", "b.nam", mix_b=0.4)),
    ("character", "character_design.json", CharacterBlendDesign("a.nam", "b.nam")),
])
def test_load_frozen_design_supports_every_saved_mode(tmp_path, mode, filename, design):
    design.write_json(tmp_path / filename)
    loaded = load_frozen_design(tmp_path, {"mode": mode})
    assert type(loaded) is type(design)
    assert loaded.mode == mode


def test_validation_report_separates_completion_quality_and_unavailable_checks():
    report = build_validation_report(
        "model-hash",
        {"full": {"rendered_ok": True, "metrics": {"raw_esr": 0.1}}, "lite": {"rendered_ok": False, "error": "Lite render failed"}},
    )
    assert report["state"] == "needs_attention"
    states = {check["id"]: check["state"] for check in report["checks"]}
    assert states == {
        "full_render": "passed", "full_quality": "unavailable",
        "lite_render": "failed", "lite_quality": "unavailable",
        "full_quiet_playing": "unavailable", "lite_quiet_playing": "unavailable",
    }
    assert report["schema_version"] == 2
    assert report["policy"]["max_raw_esr"] == 0.25
