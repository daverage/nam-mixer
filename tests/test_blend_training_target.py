"""Tests for hybrid/blend_training_target.py -- Fixed Blend A2 target
generation, see docs/blend-mode.md "FIXED BLEND TRAINING TARGET".

Mirrors tests/test_training_target.py's fixture pattern (fake identity
render(), bypassed official-V3 MD5 check) since these run without the
native nam_render tool or real .nam captures.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

import hybrid.blend_training_target as blend_training_target
import hybrid.training_target as training_target
from hybrid.blend_training_target import generate_blend_training_bundle
from hybrid.cab_ir import cab_design_from_prepared, load_and_prepare_cab_ir
from hybrid.fixed_blend import BlendDesign
from hybrid.training_target import TrainingInputError


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(blend_training_target, "render", fake_render)
    monkeypatch.setattr(training_target, "_md5_file", lambda path: training_target.OFFICIAL_V3_INPUT_MD5)


def _write_nam(path, input_level_dbu=None):
    raw = {"architecture": "WaveNet", "sample_rate": 48000.0}
    if input_level_dbu is not None:
        raw["input_level_dbu"] = input_level_dbu
        raw["output_level_dbu"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _write_training_input(path, n=48000, sample_rate=48000):
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.uniform(-1.0, 1.0, n)).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path


def _design(amp_a_path, amp_b_path, **overrides):
    fields = dict(
        amp_a_path=str(amp_a_path),
        amp_b_path=str(amp_b_path),
        mix_b=0.5,
        auto_trim_db=-1.0,
        manual_b_trim_db=0.5,
        effective_b_trim_db=-0.5,
        instrument_type="guitar",
        design_reference_profile_id="p90",
        design_reference_profile_gain_db=1.0,
        calibration_mode="raw",
    )
    fields.update(overrides)
    return BlendDesign(**fields)


def test_generate_blend_bundle_writes_expected_filenames(tmp_path):
    amp_a, amp_b = _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    bundle = generate_blend_training_bundle(design, training_input, tmp_path / "bundle")

    assert bundle.input_path.name == "input.wav"
    assert bundle.hybrid_target_path.name == "hybrid_target.wav"  # legacy filename kept for Blend too
    assert bundle.training_manifest_path.name == "training_manifest.json"
    assert bundle.input_path.is_file()
    assert bundle.hybrid_target_path.is_file()
    assert bundle.training_manifest_path.is_file()


def test_manifest_records_mode_blend_and_mix(tmp_path):
    amp_a, amp_b = _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b, mix_b=0.35)

    bundle = generate_blend_training_bundle(design, training_input, tmp_path / "bundle")
    assert bundle.manifest["mode"] == "blend"
    assert bundle.manifest["design"]["mix_b"] == 0.35
    assert abs(bundle.manifest["design"]["mix_a"] - 0.65) < 1e-9


def test_frozen_effective_trim_is_reused_not_recomputed(tmp_path):
    """Target generation must use design.effective_b_trim_db verbatim, never
    recompute an active-playing trim against the official training input."""
    amp_a, amp_b = _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    frozen_trim_db = 12.0
    design = _design(amp_a, amp_b, mix_b=1.0, auto_trim_db=0.0, manual_b_trim_db=frozen_trim_db, effective_b_trim_db=frozen_trim_db)

    bundle = generate_blend_training_bundle(design, training_input, tmp_path / "bundle")
    official_input, _ = sf.read(training_input, dtype="float32")
    target, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")

    expected = official_input * (10.0 ** (frozen_trim_db / 20.0))
    np.testing.assert_allclose(target, expected, atol=1e-4)


def test_never_applies_pickup_profile_gain_to_official_input(tmp_path):
    amp_a, amp_b = _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b, mix_b=0.0, design_reference_profile_gain_db=12.0, auto_trim_db=0.0, manual_b_trim_db=0.0, effective_b_trim_db=0.0)

    bundle = generate_blend_training_bundle(design, training_input, tmp_path / "bundle")
    official_input, _ = sf.read(training_input, dtype="float32")
    target, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")
    # mix_b=0.0 -> pure Amp A, and Amp A is an identity render of the
    # official input -- the 12 dB reference profile gain must NOT appear.
    np.testing.assert_allclose(target, official_input, atol=1e-4)


def test_length_mismatch_raises_training_input_error(tmp_path, monkeypatch):
    amp_a, amp_b = _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    def bad_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32)[:-10]
    monkeypatch.setattr(blend_training_target, "render", bad_render)

    with pytest.raises(TrainingInputError):
        generate_blend_training_bundle(design, training_input, tmp_path / "bundle")


def test_baked_cab_alters_target_but_preview_only_would_not(tmp_path):
    amp_a, amp_b = _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")

    ir_path = tmp_path / "ir.wav"
    ir = np.concatenate([[1.0], np.zeros(9)]).astype(np.float32) * 0.0
    ir[0] = 0.6
    ir[1] = 0.4
    sf.write(ir_path, ir, 48000, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)

    baked_cab = cab_design_from_prepared(prepared, original_filename="ir.wav", preview_enabled=True, baked=True)
    preview_only_cab = cab_design_from_prepared(prepared, original_filename="ir.wav", preview_enabled=True, baked=False)

    design_no_cab = _design(amp_a, amp_b, mix_b=0.0, auto_trim_db=0.0, manual_b_trim_db=0.0, effective_b_trim_db=0.0)
    design_baked = _design(amp_a, amp_b, mix_b=0.0, auto_trim_db=0.0, manual_b_trim_db=0.0, effective_b_trim_db=0.0, cab=baked_cab)
    design_preview_only = _design(amp_a, amp_b, mix_b=0.0, auto_trim_db=0.0, manual_b_trim_db=0.0, effective_b_trim_db=0.0, cab=preview_only_cab)

    bundle_no_cab = generate_blend_training_bundle(design_no_cab, training_input, tmp_path / "no_cab")
    bundle_baked = generate_blend_training_bundle(design_baked, training_input, tmp_path / "baked")
    bundle_preview_only = generate_blend_training_bundle(design_preview_only, training_input, tmp_path / "preview_only")

    target_no_cab, _ = sf.read(bundle_no_cab.hybrid_target_raw_path, dtype="float32")
    target_baked, _ = sf.read(bundle_baked.hybrid_target_raw_path, dtype="float32")
    target_preview_only, _ = sf.read(bundle_preview_only.hybrid_target_raw_path, dtype="float32")

    assert not np.allclose(target_no_cab, target_baked, atol=1e-6)
    np.testing.assert_allclose(target_no_cab, target_preview_only, atol=1e-6)

    assert bundle_baked.manifest["cab"]["baked"] is True
    assert bundle_baked.manifest["receptive_field"]["cab_fir_serial_samples"] == prepared.prepared_frame_count - 1
    assert bundle_preview_only.manifest["cab"]["baked"] is False
    assert bundle_preview_only.manifest["receptive_field"]["cab_fir_serial_samples"] == 0
