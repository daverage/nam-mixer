"""Tests for hybrid/character_training_target.py's low-level response gate
(docs/blend-mode-fixes.md, Phases 5/7/11) -- uses a fake identity render()
(same pattern as test_training_target.py) so these run without the native
nam_render tool or real .nam captures.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

import hybrid.character_training_target as character_training_target
import hybrid.training_target as training_target
from hybrid.character_blend import CharacterBlendDesign, LowLevelResponseCheck
from hybrid.character_training_target import generate_character_training_bundle
from hybrid.training_target import TrainingInputError


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    def fake_render(model, audio, sample_rate, **kwargs):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(character_training_target, "render", fake_render)
    # validate_training_input() (imported from hybrid.training_target) hashes
    # against the real official V3 excitation file -- bypass for this
    # synthetic fixture, same as test_training_target.py.
    monkeypatch.setattr(training_target, "_md5_file", lambda path: training_target.OFFICIAL_V3_INPUT_MD5)


def _write_nam(path):
    path.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000.0}), encoding="utf-8")
    return path


def _write_training_input(path, n=48000, sample_rate=48000):
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.uniform(-1.0, 1.0, n)).astype(np.float32)
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path


def _design(amp_a_path, amp_b_path, **overrides):
    fields = dict(
        amp_a_path=str(amp_a_path), amp_b_path=str(amp_b_path),
        tone_mix_b=0.5, feel_mix_b=0.5, drive_mix_b=0.5,
        calibration_mode="raw",
        # See test_training_target.py's _design() for why this defaults to a
        # no-op rather than the production "auto" default.
        output_gain_mode="manual",
        manual_output_gain_db=0.0,
    )
    fields.update(overrides)
    return CharacterBlendDesign(**fields)


def test_generate_character_training_bundle_records_a_healthy_low_level_response(tmp_path):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    bundle = generate_character_training_bundle(design, training_input, tmp_path / "out")

    assert bundle.manifest["low_level_response"]["ok"] is True
    assert bundle.manifest["low_level_response"]["dead_zone_detected"] is False
    assert bundle.manifest["low_level_response"]["levels_db"][0] == 0.0
    reference = bundle.manifest["export_validation_reference"]
    assert reference["schema_version"] == 2
    assert reference["score_frame_start"] > 0
    assert reference["score_frame_count"] > 0
    assert reference["input_excerpt_sha256"] == training_target._sha256_file(
            bundle.bundle_dir / reference["input_excerpt_path"]
    )


def test_generate_character_training_bundle_aborts_on_low_level_collapse(tmp_path, monkeypatch):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    training_input = _write_training_input(tmp_path / "input.wav")
    design = _design(amp_a, amp_b)

    failing_check = LowLevelResponseCheck(
        ok=False, levels_db=[0.0, -6.0], output_rms_dbfs=[-10.0, -90.0],
        max_step_error_db=74.0, dead_zone_detected=True, warning="simulated collapse",
    )
    monkeypatch.setattr(character_training_target, "evaluate_bundle_low_level_response", lambda *a, **k: failing_check)

    out_dir = tmp_path / "out"
    with pytest.raises(TrainingInputError, match="low-level response check failed"):
        generate_character_training_bundle(design, training_input, out_dir)

    # No bundle files/manifest should be left behind by an aborted generation.
    assert not (out_dir / "training_manifest.json").exists()


# ---------------------------------------------------------------------------
# Phases 10/11 -- post-training validation: the trained Full A2 must track
# the teacher's own recorded low-level response, shared with
# hybrid.kaggle_training.validate_downloaded_model and scripts/train_a2.py.
# ---------------------------------------------------------------------------

def test_check_full_low_level_response_returns_none_for_non_character_or_missing_section(tmp_path):
    assert character_training_target.check_full_low_level_response({"mode": "hybrid"}, tmp_path / "x.nam", tmp_path / "in.wav", 48000) is None
    legacy = character_training_target.check_full_low_level_response({"mode": "character"}, tmp_path / "x.nam", tmp_path / "in.wav", 48000)
    assert legacy["state"] == "unavailable"
    assert "legacy" in legacy["reason"]


def _equivalent_reference_manifest(tmp_path, audio, levels_db, teacher_rms):
    input_path = tmp_path / "input.wav"
    excerpt_path = tmp_path / "export_validation_reference_input.wav"
    sf.write(input_path, audio, 48000, subtype="FLOAT")
    sf.write(excerpt_path, audio, 48000, subtype="FLOAT")
    return input_path, {
        "mode": "character",
        "export_validation_reference": {
            "schema_version": 2,
            "levels_db": levels_db,
            "teacher_output_rms_dbfs": teacher_rms,
            "sample_rate": 48000,
            "frame_start": 0,
            "frame_count": len(audio),
            "score_frame_start": 0,
            "score_frame_count": len(audio),
            "input_excerpt_path": excerpt_path.name,
            "input_excerpt_sha256": training_target._sha256_file(excerpt_path),
            "source_training_input_sha256": training_target._sha256_file(input_path),
            "amp_a_sha256": "a" * 64,
            "amp_b_sha256": "b" * 64,
            "cab_baked": False,
            "teacher_semantics_version": 2,
            "processing_version": "character-export-reference-v2",
        },
    }


def test_check_full_low_level_response_passes_when_full_tracks_teacher(tmp_path):
    from hybrid.character_training_target import LOW_LEVEL_CHECK_REFERENCE_SECONDS
    from hybrid.input_profiles import db_to_amplitude

    nam_path = _write_nam(tmp_path / "model.nam")
    levels_db = [0.0, -6.0, -12.0]
    rng = np.random.default_rng(1)
    audio = (0.3 * rng.uniform(-1, 1, 48000)).astype(np.float32)
    reference = audio[: int(48000 * LOW_LEVEL_CHECK_REFERENCE_SECONDS)]
    teacher_rms = []
    for gain_db in levels_db:
        scaled = reference * db_to_amplitude(gain_db)
        rms = float(np.sqrt(np.mean(np.square(scaled, dtype=np.float64))))
        teacher_rms.append(float(max(20.0 * np.log10(max(rms, 1e-10)), -90.0)))

    input_path, manifest = _equivalent_reference_manifest(tmp_path, reference, levels_db, teacher_rms)
    result = character_training_target.check_full_low_level_response(manifest, nam_path, input_path, 48000)
    assert result["pass"] is True
    assert result["max_error_db"] < 1e-6
    assert result["dead_zone_detected"] is False
    assert json.dumps(result)  # must be JSON-serializable for training_manifest.json


def test_check_full_low_level_response_flags_a_new_dead_zone(tmp_path, monkeypatch):
    def gating_render(model, audio, sample_rate, **kwargs):
        # Simulates a NEWLY trained hard gate the teacher itself did not have.
        return np.zeros_like(np.asarray(audio, dtype=np.float32))
    monkeypatch.setattr(character_training_target, "render", gating_render)

    nam_path = _write_nam(tmp_path / "model.nam")
    levels_db = [0.0, -6.0, -12.0]
    input_path, manifest = _equivalent_reference_manifest(
        tmp_path, 0.3 * np.ones(48000, dtype=np.float32), levels_db, [-10.0, -16.0, -22.0],
    )

    result = character_training_target.check_full_low_level_response(manifest, nam_path, input_path, 48000)
    assert result["pass"] is False
    assert result["dead_zone_detected"] is True
    assert json.dumps(result)


def test_fixed_level_offset_is_separate_from_response_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(
        character_training_target, "render",
        lambda model, audio, sample_rate, **kwargs: np.asarray(audio, dtype=np.float32) * 0.5,
    )
    audio = np.full(48000, 0.25, dtype=np.float32)
    levels = [0.0, -6.0, -12.0]
    teacher = [-12.0412, -18.0412, -24.0412]
    input_path, manifest = _equivalent_reference_manifest(tmp_path, audio, levels, teacher)
    result = character_training_target.check_export_low_level_response(
        manifest, _write_nam(tmp_path / "model.nam"), input_path, 48000, variant="lite", slim=1.0,
    )
    assert result["level_offset_db"] == pytest.approx(-6.0206, abs=0.01)
    assert result["max_shape_error_db"] < 0.01
    assert result["dead_zone_detected"] is False
    assert result["pass"] is False  # absolute level policy, not a collapse verdict


def test_modified_reference_excerpt_hash_cannot_pass(tmp_path):
    audio = np.full(48000, 0.25, dtype=np.float32)
    input_path, manifest = _equivalent_reference_manifest(tmp_path, audio, [0.0], [-12.0])
    (tmp_path / "export_validation_reference_input.wav").write_bytes(b"modified")
    result = character_training_target.check_full_low_level_response(
        manifest, _write_nam(tmp_path / "model.nam"), input_path, 48000,
    )
    assert result["state"] == "unavailable"
    assert result["pass"] is None
