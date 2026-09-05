"""Tests for hybrid/training_target.py -- docs/phase3.md sections 7-12.

Uses a fake identity render() (same pattern as test_pipeline_render.py) so
these run without the native nam_render tool or real .nam captures.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

import hybrid.training_target as training_target
from hybrid.design import HybridDesign
from hybrid.training_target import (
    TrainingInputError,
    generate_training_bundle,
    validate_training_input,
)


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(training_target, "render", fake_render)
    # Bypass the official-V3-file MD5 check for these synthetic fixtures --
    # dedicated tests below exercise the real check against
    # OFFICIAL_V3_INPUT_MD5 directly. We don't ship the real ~27MB official
    # file as a test fixture.
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
        crossover_dbfs=-20.0,
        transition_width_db=8.0,
        auto_trim_db=-1.0,
        manual_b_trim_db=0.5,
        effective_b_trim_db=-0.5,
        instrument_type="guitar",
        design_reference_profile_id="p90",
        design_reference_profile_gain_db=1.0,
        calibration_mode="raw",
    )
    fields.update(overrides)
    return HybridDesign(**fields)


def test_validate_training_input_rejects_wrong_sample_rate(tmp_path):
    path = _write_training_input(tmp_path / "in.wav", sample_rate=44100)
    with pytest.raises(TrainingInputError):
        validate_training_input(path)


def test_validate_training_input_rejects_stereo(tmp_path):
    path = tmp_path / "in.wav"
    audio = np.zeros((1000, 2), dtype=np.float32)
    sf.write(path, audio, 48000, subtype="FLOAT")
    with pytest.raises(TrainingInputError):
        validate_training_input(path)


def test_validate_training_input_rejects_missing_file(tmp_path):
    with pytest.raises(TrainingInputError):
        validate_training_input(tmp_path / "nope.wav")


def test_validate_training_input_accepts_good_file(tmp_path):
    path = _write_training_input(tmp_path / "in.wav")
    audio, info = validate_training_input(path)
    assert info.sample_rate == 48000
    assert info.frame_count == len(audio)


def test_validate_training_input_rejects_non_v3_file(tmp_path, monkeypatch):
    """With the real MD5 check active (undoing the autouse bypass), a
    correctly-formatted (mono, 48kHz) but non-official file must still be
    rejected -- docs/phase3.md review: 'any recognized input' is not enough,
    it must be V3 specifically."""
    monkeypatch.undo()  # remove the autouse identity_render/_md5_file bypass for this test
    import hybrid.training_target as training_target
    monkeypatch.setattr(training_target, "render", lambda model, audio, sr: np.asarray(audio, dtype=np.float32).copy())

    path = _write_training_input(tmp_path / "in.wav")
    with pytest.raises(TrainingInputError, match="official NAM v3.0.0"):
        validate_training_input(path)


def test_validate_training_input_accepts_real_md5_match(tmp_path, monkeypatch):
    """The acceptance path with the real check active: a file whose actual
    MD5 equals OFFICIAL_V3_INPUT_MD5 (simulated here via monkeypatching the
    constant to this fixture's real hash, since we don't ship the real
    27MB official file) is accepted."""
    monkeypatch.undo()
    import hybrid.training_target as training_target
    monkeypatch.setattr(training_target, "render", lambda model, audio, sr: np.asarray(audio, dtype=np.float32).copy())

    path = _write_training_input(tmp_path / "in.wav")
    real_md5 = training_target._md5_file(path)
    monkeypatch.setattr(training_target, "OFFICIAL_V3_INPUT_MD5", real_md5)

    audio, info = validate_training_input(path)
    assert info.md5 == real_md5
    assert len(info.sha256) == 64


def test_generate_training_bundle_output_files(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    out_dir = tmp_path / "bundle"
    bundle = generate_training_bundle(design, training_input, out_dir)

    assert bundle.input_path.is_file()
    assert bundle.hybrid_target_raw_path.is_file()
    assert bundle.hybrid_target_path.is_file()
    assert bundle.hybrid_metadata_path.is_file()
    assert bundle.training_manifest_path.is_file()

    target_audio, sr = sf.read(bundle.hybrid_target_path, dtype="float32")
    input_audio, _ = sf.read(bundle.input_path, dtype="float32")
    assert sr == 48000
    assert len(target_audio) == len(input_audio)


def test_generate_training_bundle_target_is_float32_not_pcm16(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    bundle = generate_training_bundle(design, training_input, tmp_path / "bundle")
    info = sf.info(bundle.hybrid_target_path)
    assert info.subtype == "FLOAT"


def test_generate_training_bundle_never_applies_pickup_profile_gain(tmp_path):
    """The design was auditioned with a +6dB profile -- the actual training
    target's amp inputs must NOT reflect that gain (docs/phase3.md section 1)."""
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input_path = tmp_path / "input.wav"
    training_input = _write_training_input(training_input_path)
    design = _design(amp_a, amp_b, design_reference_profile_gain_db=6.0, calibration_mode="raw")

    bundle = generate_training_bundle(design, training_input, tmp_path / "bundle_a")

    raw_input, _ = sf.read(training_input_path, dtype="float32")
    target_raw, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")

    # With raw calibration (no compensation) and identity render, the hybrid
    # blend of two identical unity-gain renders should track the ORIGINAL
    # input's peak, not a +6dB-boosted one.
    assert np.max(np.abs(target_raw)) == pytest.approx(np.max(np.abs(raw_input)), rel=0.05)

    with open(bundle.hybrid_metadata_path) as f:
        meta = json.load(f)
    assert meta["hybrid"]["pickup_profile_applied_to_training_input"] is False


def test_generate_training_bundle_uses_frozen_trim_not_recomputed(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b, auto_trim_db=-99.0, manual_b_trim_db=0.0, effective_b_trim_db=-99.0)

    bundle = generate_training_bundle(design, training_input, tmp_path / "bundle")
    with open(bundle.training_manifest_path) as f:
        manifest = json.load(f)
    assert manifest["design"]["frozen_effective_b_trim_db"] == -99.0
    assert manifest["design"]["original_auto_trim_db"] == -99.0


def test_generate_training_bundle_calibration_both_calibrated(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam", input_level_dbu=8.0)
    amp_b = _write_nam(tmp_path / "b.nam", input_level_dbu=12.0)
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b, calibration_mode="auto", reference_input_level_dbu=12.0)

    bundle = generate_training_bundle(design, training_input, tmp_path / "bundle")
    with open(bundle.training_manifest_path) as f:
        manifest = json.load(f)
    assert manifest["calibration"]["applied"] is True
    assert manifest["calibration"]["amp_a_compensation_db"] == pytest.approx(4.0)
    assert manifest["calibration"]["amp_b_compensation_db"] == pytest.approx(0.0)


def test_generate_training_bundle_calibration_one_sided_falls_back_to_raw(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam", input_level_dbu=8.0)
    amp_b = _write_nam(tmp_path / "b.nam")  # no calibration metadata
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b, calibration_mode="auto")

    bundle = generate_training_bundle(design, training_input, tmp_path / "bundle")
    with open(bundle.training_manifest_path) as f:
        manifest = json.load(f)
    assert manifest["calibration"]["applied"] is False
    assert manifest["calibration"]["warning"] is not None


def test_generate_training_bundle_no_nan_inf_and_records_hashes(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    bundle = generate_training_bundle(design, training_input, tmp_path / "bundle")
    target, _ = sf.read(bundle.hybrid_target_path, dtype="float32")
    assert np.all(np.isfinite(target))

    with open(bundle.training_manifest_path) as f:
        manifest = json.load(f)
    assert len(manifest["target"]["raw_sha256"]) == 64
    assert len(manifest["target"]["final_sha256"]) == 64
    assert manifest["target"]["preview_limiter_used"] is False
    assert manifest["target"]["synthetic_latency_samples"] == 0
    assert manifest["training_input"]["sha256"] == bundle.training_input_info.sha256


def test_generate_training_bundle_applies_only_fixed_peak_ceiling_when_hot(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")

    training_input_path = tmp_path / "input.wav"
    audio = np.full(48000, 0.99, dtype=np.float32)  # very hot after any blending
    sf.write(training_input_path, audio, 48000, subtype="FLOAT")

    design = _design(amp_a, amp_b)
    bundle = generate_training_bundle(design, training_input_path, tmp_path / "bundle")

    assert bundle.safety.gain_reduction_db > 0
    target, _ = sf.read(bundle.hybrid_target_path, dtype="float32")
    assert np.max(np.abs(target)) <= 10 ** (training_target.A2_TARGET_PEAK_CEILING_DBFS / 20.0) + 1e-6


def test_generate_training_bundle_leaves_target_untouched_when_already_safe(tmp_path):
    """docs/phase3.md review section 4: a target that never reaches 0 dBFS
    must be left COMPLETELY unchanged, not massaged down to some arbitrary
    fixed ceiling like the old -3 dBFS default."""
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")

    training_input_path = tmp_path / "input.wav"
    audio = np.full(48000, 0.1, dtype=np.float32)  # peak ~ -20 dBFS, nowhere near 0
    sf.write(training_input_path, audio, 48000, subtype="FLOAT")

    design = _design(amp_a, amp_b)
    bundle = generate_training_bundle(design, training_input_path, tmp_path / "bundle")

    assert bundle.safety.gain_reduction_db == 0.0
    raw, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")
    final, _ = sf.read(bundle.hybrid_target_path, dtype="float32")
    np.testing.assert_array_equal(raw, final)


def test_generate_training_bundle_rejects_mismatched_sample_rate(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav", sample_rate=44100)
    design = _design(amp_a, amp_b)

    with pytest.raises(TrainingInputError):
        generate_training_bundle(design, training_input, tmp_path / "bundle")
