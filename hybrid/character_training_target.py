"""Official-input A2 bundle generation for deterministic Character Blend."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from .calibration import resolve_calibration
from .character_blend import CharacterBlendDesign, LowLevelResponseCheck, build_character_blend, evaluate_low_level_response
from .envelope import bounded_envelope_max_history_ms
from .input_profiles import db_to_amplitude
from .nam_loader import load_nam
from .render import render
from .safety import apply_peak_ceiling, check_audio
from .training_target import (
    A2_TARGET_PEAK_CEILING_DBFS, TargetSafetyReport, TrainingBundle, TrainingInputError,
    _git_commit, _sha256_file, compute_receptive_field_record, maybe_bake_cab, validate_training_input,
)

HYBRID_BUILDER_VERSION = "character-blend-v1"
# Bounded so the sweep (Phases 4-5, docs/blend-mode-fixes.md) costs a fixed
# handful of extra NAM renders regardless of how long the official training
# input is -- it is a sanity check on the teacher's low-level response, not
# a re-render of the whole excitation file.
LOW_LEVEL_CHECK_REFERENCE_SECONDS = 5.0


def evaluate_bundle_low_level_response(design, amp_a, amp_b, calibration, official_input, sample_rate) -> LowLevelResponseCheck:
    reference = official_input[: int(sample_rate * LOW_LEVEL_CHECK_REFERENCE_SECONDS)]
    if len(reference) == 0:
        reference = official_input

    def build_pair_at_gain(gain_db: float):
        scaled = (reference * db_to_amplitude(gain_db)).astype(np.float32)
        a = render(amp_a, (scaled * db_to_amplitude(calibration.amp_a_gain_db)).astype(np.float32), sample_rate)
        b = render(amp_b, (scaled * db_to_amplitude(calibration.amp_b_gain_db)).astype(np.float32), sample_rate)
        return SimpleNamespace(dry=scaled, amp_a=a, amp_b=b, sample_rate=sample_rate)

    return evaluate_low_level_response(build_pair_at_gain, design)


def check_full_low_level_response(manifest: dict, nam_path, input_path, sample_rate: int) -> "dict | None":
    """Re-run the teacher's low-level response sweep (docs/blend-mode-fixes.md,
    Phases 10-11) through an exported Full A2 and compare its output RMS at
    each level against the teacher's OWN recorded RMS (the `low_level_response`
    section a Character Blend manifest carries after `evaluate_bundle_low_level_
    response`). A trained A2 must track the teacher across the sweep -- it
    must not develop a NEW hard low-level gate the corrected teacher did not
    have. Shared by `scripts/train_a2.py` (local trainer) and
    `hybrid.kaggle_training.validate_downloaded_model` (Kaggle's local
    re-validation of a downloaded model) so both hold trained models to the
    identical bar. Returns None for non-Character-Blend or older manifests
    without a recorded `low_level_response`.
    """
    teacher = manifest.get("low_level_response")
    if not teacher or manifest.get("mode") != "character":
        return None
    input_audio, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    # Same bounded excerpt used to build the teacher's own sweep -- both
    # fixed-length, so comparing per-level RMS is apples-to-apples regardless
    # of the full excitation file's length.
    reference = input_audio[: int(sample_rate * LOW_LEVEL_CHECK_REFERENCE_SECONDS)]
    if len(reference) == 0:
        reference = input_audio
    model = load_nam(nam_path)

    full_rms_dbfs = []
    for gain_db in teacher["levels_db"]:
        scaled = (reference * db_to_amplitude(gain_db)).astype(np.float32)
        rendered = render(model, scaled, sr)
        rms = float(np.sqrt(np.mean(np.square(rendered, dtype=np.float64))))
        full_rms_dbfs.append(float(max(20.0 * np.log10(max(rms, 1e-10)), -90.0)))

    errors_db = [float(abs(full - teach)) for full, teach in zip(full_rms_dbfs, teacher["output_rms_dbfs"])]
    max_error_db = max(errors_db) if errors_db else 0.0
    # The teacher's own dead-zone status is the baseline: only flag the
    # TRAINED model going silent when the teacher itself did not.
    dead_zone_detected = bool(any(rms <= -89.999 for rms in full_rms_dbfs) and not teacher.get("dead_zone_detected"))
    passed = bool(not dead_zone_detected and max_error_db < 20.0)
    return {
        "levels_db": teacher["levels_db"], "teacher_output_rms_dbfs": teacher["output_rms_dbfs"],
        "full_output_rms_dbfs": full_rms_dbfs, "full_error_db": errors_db,
        "max_error_db": max_error_db, "dead_zone_detected": dead_zone_detected, "pass": passed,
    }


def build_character_training_manifest(design, amp_a, amp_b, amp_a_sha256, amp_b_sha256, calibration, training_input, safety, receptive_field, warnings, low_level_response: LowLevelResponseCheck):
    return {
        "hybrid_builder_version": HYBRID_BUILDER_VERSION, "git_commit": _git_commit(), "mode": "character",
        "amp_a": {"filename": Path(design.amp_a_path).name, "path": design.amp_a_path, "sha256": amp_a_sha256, "architecture": amp_a.architecture, "sample_rate": amp_a.sample_rate, "input_level_dbu": amp_a.input_level_dbu},
        "amp_b": {"filename": Path(design.amp_b_path).name, "path": design.amp_b_path, "sha256": amp_b_sha256, "architecture": amp_b.architecture, "sample_rate": amp_b.sample_rate, "input_level_dbu": amp_b.input_level_dbu},
        "design": design.to_dict(),
        "character_analysis": {"version": 1, "amp_a_sha256": design.amp_a_sha256 or amp_a_sha256, "amp_b_sha256": design.amp_b_sha256 or amp_b_sha256, "analysis_a": design.analysis_a, "analysis_b": design.analysis_b, "tone_mix_b": design.tone_mix_b, "feel_mix_b": design.feel_mix_b, "drive_mix_b": design.drive_mix_b, "drive_curve": {"low": design.drive_low_mix_b, "mid": design.drive_mid_mix_b, "high": design.drive_high_mix_b}},
        "calibration": {"requested_mode": design.calibration_mode, "effective_mode": "auto" if calibration.applied else "raw", "reference_input_level_dbu": calibration.reference_input_level_dbu, "amp_a_compensation_db": calibration.amp_a_gain_db, "amp_b_compensation_db": calibration.amp_b_gain_db, "applied": calibration.applied, "warning": calibration.warning},
        "training_input": {"path": training_input.path, "sample_rate": training_input.sample_rate, "frame_count": training_input.frame_count, "sha256": training_input.sha256, "md5": training_input.md5, "detected_version": training_input.detected_version},
        "target": {"raw_sha256": safety.raw_sha256, "final_sha256": safety.final_sha256, "peak_before_safety_dbfs": safety.raw_peak_dbfs, "peak_after_safety_dbfs": safety.final_peak_dbfs, "global_safety_gain_reduction_db": safety.gain_reduction_db, "preview_limiter_used": False, "synthetic_latency_samples": 0},
        "training": {"status": "not yet run -- Character Blend defaults to high_def (120 epochs)", "recommended_epoch_preset": "high_def"},
        "cab": design.cab.to_dict() if design.cab else {"selected": False}, "receptive_field": receptive_field, "warnings": warnings,
        "low_level_response": low_level_response.to_dict(),
    }


def generate_character_training_bundle(design: CharacterBlendDesign, official_input_path: str | Path, output_directory: str | Path, target_peak_dbfs: float = A2_TARGET_PEAK_CEILING_DBFS) -> TrainingBundle:
    official_input_path, output_directory = Path(official_input_path), Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    official_input, input_info = validate_training_input(official_input_path)
    amp_a, amp_b = load_nam(design.amp_a_path), load_nam(design.amp_b_path)
    a_sha, b_sha = _sha256_file(design.amp_a_path), _sha256_file(design.amp_b_path)
    calibration = resolve_calibration(design.calibration_mode, design.reference_input_level_dbu, amp_a.input_level_dbu, amp_b.input_level_dbu)
    warnings = [calibration.warning] if calibration.warning else []
    a = render(amp_a, (official_input * db_to_amplitude(calibration.amp_a_gain_db)).astype(np.float32), input_info.sample_rate)
    b = render(amp_b, (official_input * db_to_amplitude(calibration.amp_b_gain_db)).astype(np.float32), input_info.sample_rate)
    pair = SimpleNamespace(dry=official_input, amp_a=a, amp_b=b, sample_rate=input_info.sample_rate)
    target_raw = build_character_blend(pair, design).blend
    if len(target_raw) != len(official_input): raise TrainingInputError("generated Character Blend target is not sample-aligned with training input")
    low_level_response = evaluate_bundle_low_level_response(design, amp_a, amp_b, calibration, official_input, input_info.sample_rate)
    if not low_level_response.ok:
        raise TrainingInputError(
            f"Character Blend low-level response check failed -- {low_level_response.warning} "
            "-- refusing to generate a training bundle with a hard low-level gate baked in (see docs/blend-mode-fixes.md)"
        )
    target_raw = maybe_bake_cab(target_raw, design.cab, input_info.sample_rate)
    check = check_audio(target_raw)
    if check.has_nan_or_inf: raise TrainingInputError("generated Character Blend target contains NaN/Inf -- aborting")
    if check.is_silent: warnings.append("generated Character Blend target is silent -- check amp/DI/design selection")
    target_final, reduction = apply_peak_ceiling(target_raw, target_peak_dbfs)
    input_out, raw_out, final_out, manifest_out = output_directory / "input.wav", output_directory / "hybrid_target_raw.wav", output_directory / "hybrid_target.wav", output_directory / "training_manifest.json"
    shutil.copyfile(official_input_path, input_out); sf.write(raw_out, target_raw.astype(np.float32), input_info.sample_rate, subtype="FLOAT"); sf.write(final_out, target_final.astype(np.float32), input_info.sample_rate, subtype="FLOAT")
    safety = TargetSafetyReport(check.peak_dbfs, check.peak_dbfs - reduction, reduction, _sha256_file(raw_out), _sha256_file(final_out))
    receptive = compute_receptive_field_record("character", amp_a, amp_b, input_info.sample_rate, design.cab, bounded_envelope_max_history_ms())
    manifest = build_character_training_manifest(design, amp_a, amp_b, a_sha, b_sha, calibration, input_info, safety, receptive, warnings, low_level_response)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TrainingBundle(output_directory, input_out, raw_out, final_out, None, manifest_out, design, safety, input_info, warnings, manifest)
