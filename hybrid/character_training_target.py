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
from .safety import apply_output_gain, apply_peak_ceiling, check_audio, compute_auto_output_gain_db
from .training_target import (
    A2_TARGET_PEAK_CEILING_DBFS, TargetSafetyReport, TrainingBundle, TrainingInputError,
    _git_commit, _sha256_file, compute_receptive_field_record, maybe_bake_cab, validate_training_input,
)
from .validation_report import DEFAULT_VALIDATION_POLICY

HYBRID_BUILDER_VERSION = "character-blend-v2"
# Bounded so the sweep (Phases 4-5, docs/blend-mode-fixes.md) costs a fixed
# handful of extra NAM renders regardless of how long the official training
# input is -- it is a sanity check on the teacher's low-level response, not
# a re-render of the whole excitation file.
LOW_LEVEL_CHECK_REFERENCE_SECONDS = 5.0
EXPORT_VALIDATION_REFERENCE_SCHEMA_VERSION = 2
EXPORT_VALIDATION_WARMUP_SECONDS = 0.25
EXPORT_VALIDATION_SCORE_SECONDS = 4.0
EXPORT_VALIDATION_MIN_SCORE_SECONDS = 0.5


def evaluate_bundle_low_level_response(design, amp_a, amp_b, calibration, official_input, sample_rate) -> LowLevelResponseCheck:
    reference = official_input[: int(sample_rate * LOW_LEVEL_CHECK_REFERENCE_SECONDS)]
    if len(reference) == 0:
        reference = official_input

    def build_pair_at_gain(gain_db: float):
        scaled = (reference * db_to_amplitude(gain_db)).astype(np.float32)
        a = render(amp_a, (scaled * db_to_amplitude(calibration.amp_a_gain_db + design.amp_a_input_gain_db)).astype(np.float32), sample_rate)
        b = render(amp_b, (scaled * db_to_amplitude(calibration.amp_b_gain_db + design.amp_b_input_gain_db)).astype(np.float32), sample_rate)
        return SimpleNamespace(dry=scaled, amp_a=a, amp_b=b, sample_rate=sample_rate)

    return evaluate_low_level_response(build_pair_at_gain, design)


def _select_reference_excerpt(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int, int]:
    """Choose a deterministic energetic excerpt instead of blindly using t=0.

    The official excitation may begin with silence.  Score one-second windows
    by RMS and select the earliest highest-energy contiguous five-second
    region; reject material that contains no useful excitation.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) <= 0 or not np.all(np.isfinite(audio)):
        raise TrainingInputError("cannot build validation reference from an empty training input")
    min_score = max(1, int(round(sample_rate * EXPORT_VALIDATION_MIN_SCORE_SECONDS)))
    if len(audio) < min_score:
        raise TrainingInputError(
            f"training input is too short for exported-model validation; need at least {min_score} frames"
        )
    warmup = min(int(round(sample_rate * EXPORT_VALIDATION_WARMUP_SECONDS)), len(audio) - min_score)
    score_frames = min(int(round(sample_rate * EXPORT_VALIDATION_SCORE_SECONDS)), len(audio) - warmup)
    last_start = len(audio) - score_frames
    step = max(1, int(sample_rate))
    starts = list(range(warmup, last_start + 1, step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    score_start = max(
        starts,
        key=lambda offset: float(np.mean(np.square(audio[offset:offset + score_frames], dtype=np.float64))),
    )
    excerpt_start = score_start - warmup
    excerpt = np.asarray(audio[excerpt_start:score_start + score_frames], dtype=np.float32)
    scored = excerpt[warmup:warmup + score_frames]
    rms = float(np.sqrt(np.mean(np.square(scored, dtype=np.float64))))
    active_frames = int(np.count_nonzero(np.abs(scored) >= 1e-6))
    if rms < 1e-7 or active_frames < min(64, len(scored)):
        raise TrainingInputError("training input has no useful excitation for exported-model validation")
    return excerpt, int(excerpt_start), int(warmup), int(score_frames)


def build_export_validation_reference(design, amp_a, amp_b, calibration, official_input, sample_rate: int, *, output_gain_db: float, peak_reduction_db: float) -> tuple[dict, np.ndarray]:
    """Build the *processed* teacher reference used for exported-model checks.

    Unlike the pre-processing teacher-health sweep, this applies the baked
    cabinet (when selected), frozen output gain, and the one fixed target peak
    reduction.  None of these values are recomputed per level.
    """
    excerpt, start, score_start, score_count = _select_reference_excerpt(official_input, sample_rate)
    teacher_rms = []
    peak_scale = db_to_amplitude(-peak_reduction_db)
    for gain_db in (0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0):
        scaled = (excerpt * db_to_amplitude(gain_db)).astype(np.float32)
        a = render(amp_a, (scaled * db_to_amplitude(calibration.amp_a_gain_db + design.amp_a_input_gain_db)).astype(np.float32), sample_rate)
        b = render(amp_b, (scaled * db_to_amplitude(calibration.amp_b_gain_db + design.amp_b_input_gain_db)).astype(np.float32), sample_rate)
        teacher = build_character_blend(SimpleNamespace(dry=scaled, amp_a=a, amp_b=b, sample_rate=sample_rate), design).blend
        teacher = maybe_bake_cab(teacher, design.cab, sample_rate)
        teacher = apply_output_gain(teacher, output_gain_db) * peak_scale
        scored_teacher = teacher[score_start:score_start + score_count]
        rms = float(np.sqrt(np.mean(np.square(scored_teacher, dtype=np.float64))))
        teacher_rms.append(float(max(20.0 * np.log10(max(rms, 1e-10)), -90.0)))
    return {
        "schema_version": EXPORT_VALIDATION_REFERENCE_SCHEMA_VERSION,
        "levels_db": [0.0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0],
        "teacher_output_rms_dbfs": teacher_rms,
        "sample_rate": sample_rate,
        "frame_start": start,
        "frame_count": len(excerpt),
        "score_frame_start": score_start,
        "score_frame_count": score_count,
        "warmup_policy": "render the complete saved excerpt, score only score_frame_start:score_frame_count",
        "input_gain_db": {"amp_a": calibration.amp_a_gain_db + design.amp_a_input_gain_db, "amp_b": calibration.amp_b_gain_db + design.amp_b_input_gain_db},
        "output_gain_db": output_gain_db,
        "peak_reduction_db": peak_reduction_db,
        "cab_baked": bool(design.cab and design.cab.baked),
        "teacher_semantics_version": design.teacher_semantics_version,
        "processing_version": "character-export-reference-v2",
        "validation_policy": dict(DEFAULT_VALIDATION_POLICY),
    }, excerpt


def check_export_low_level_response(manifest: dict, nam_path, input_path, sample_rate: int, *, variant: str, slim: float) -> "dict | None":
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
    if manifest.get("mode") != "character":
        return None
    teacher = manifest.get("export_validation_reference")
    if not teacher:
        return {"state": "unavailable", "reason": "legacy bundle has no equivalent validation reference", "pass": None, "variant": variant}
    if teacher.get("schema_version") != EXPORT_VALIDATION_REFERENCE_SCHEMA_VERSION:
        return {"state": "unavailable", "reason": f"unsupported or legacy validation reference schema {teacher.get('schema_version')!r}", "pass": None, "variant": variant}
    excerpt_name = teacher.get("input_excerpt_path")
    if not isinstance(excerpt_name, str) or Path(excerpt_name).name != excerpt_name:
        return {"state": "unavailable", "reason": "validation reference excerpt path is invalid", "pass": None, "variant": variant}
    excerpt_path = Path(input_path).parent / excerpt_name
    if not excerpt_path.is_file() or _sha256_file(excerpt_path) != teacher.get("input_excerpt_sha256"):
        return {"state": "unavailable", "reason": "validation reference excerpt is missing or its hash does not match", "pass": None, "variant": variant}
    reference, sr = sf.read(str(excerpt_path), dtype="float32", always_2d=False)
    frame_count = int(teacher.get("frame_count", 0))
    score_start = int(teacher.get("score_frame_start", -1))
    score_count = int(teacher.get("score_frame_count", 0))
    levels = teacher.get("levels_db") or []
    teacher_rms = teacher.get("teacher_output_rms_dbfs") or []
    valid = (
        reference.ndim == 1 and np.all(np.isfinite(reference)) and len(reference) == frame_count
        and sr == sample_rate == teacher.get("sample_rate") and score_start >= 0 and score_count > 0
        and score_start + score_count <= len(reference) and len(levels) == len(teacher_rms) > 0
        and np.all(np.isfinite(np.asarray(levels, dtype=float)))
        and np.all(np.isfinite(np.asarray(teacher_rms, dtype=float)))
        and teacher.get("processing_version") == "character-export-reference-v2"
        and teacher.get("teacher_semantics_version") in (1, 2)
        and all(isinstance(teacher.get(name), str) and len(teacher[name]) == 64 for name in ("amp_a_sha256", "amp_b_sha256", "source_training_input_sha256"))
    )
    if teacher.get("source_training_input_sha256") != _sha256_file(input_path):
        valid = False
    if teacher.get("cab_baked") and not (
        isinstance(teacher.get("cab_sha256"), str) and len(teacher["cab_sha256"]) == 64
    ):
        valid = False
    if not valid:
        return {"state": "unavailable", "reason": "validation reference metadata or audio is invalid", "pass": None, "variant": variant}
    model = load_nam(nam_path)

    output_rms_dbfs = []
    policy = {**DEFAULT_VALIDATION_POLICY, **(teacher.get("validation_policy") or {})}
    floor_dbfs = float(policy["silence_floor_dbfs"])
    for gain_db in levels:
        scaled = (reference * db_to_amplitude(gain_db)).astype(np.float32)
        rendered = render(model, scaled, sr, slim=slim)
        scored = rendered[score_start:score_start + score_count]
        rms = float(np.sqrt(np.mean(np.square(scored, dtype=np.float64))))
        output_rms_dbfs.append(float(max(20.0 * np.log10(max(rms, 1e-12)), floor_dbfs)))

    signed_errors = np.asarray(output_rms_dbfs) - np.asarray(teacher_rms)
    level_offset_db = float(np.median(signed_errors))
    shape_errors = signed_errors - level_offset_db
    absolute_errors = np.abs(signed_errors)
    max_error_db = float(np.max(absolute_errors))
    max_shape_error_db = float(np.max(np.abs(shape_errors)))
    relative_errors = signed_errors - signed_errors[0]
    max_extra_quiet_attenuation_db = float(max(0.0, -float(np.min(relative_errors))))
    dead_zone_detected = bool(
        any(rms <= floor_dbfs + 1e-6 and teach > floor_dbfs + 1e-6 for rms, teach in zip(output_rms_dbfs, teacher_rms))
    )
    failures = []
    if max_error_db > policy["max_absolute_level_error_db"]:
        failures.append(f"absolute level error {max_error_db:.2f} dB exceeds {policy['max_absolute_level_error_db']:.2f} dB")
    if max_shape_error_db > policy["max_response_shape_error_db"]:
        failures.append(f"response-shape error {max_shape_error_db:.2f} dB exceeds {policy['max_response_shape_error_db']:.2f} dB")
    if max_extra_quiet_attenuation_db > policy["max_extra_quiet_attenuation_db"]:
        failures.append(f"extra quiet attenuation {max_extra_quiet_attenuation_db:.2f} dB exceeds {policy['max_extra_quiet_attenuation_db']:.2f} dB")
    if dead_zone_detected:
        failures.append("export becomes silent at a level where the teacher remains active")
    passed = not failures
    return {
        "state": "passed" if passed else "failed",
        "variant": variant, "slim": slim, "levels_db": levels,
        "teacher_output_rms_dbfs": teacher_rms, "output_rms_dbfs": output_rms_dbfs,
        "signed_level_errors_db": signed_errors.tolist(), "absolute_level_errors_db": absolute_errors.tolist(),
        "level_offset_db": level_offset_db, "response_shape_errors_db": shape_errors.tolist(),
        "max_error_db": max_error_db, "max_shape_error_db": max_shape_error_db,
        "max_extra_quiet_attenuation_db": max_extra_quiet_attenuation_db,
        "dead_zone_detected": dead_zone_detected, "policy": policy,
        "reason": "; ".join(failures) if failures else "level and quiet-response metrics are within policy",
        "pass": passed,
    }


def check_full_low_level_response(manifest: dict, nam_path, input_path, sample_rate: int) -> "dict | None":
    """Backward-compatible Full-variant entry point for existing callers."""
    return check_export_low_level_response(
        manifest, nam_path, input_path, sample_rate, variant="full", slim=0.0,
    )


def build_character_training_manifest(design, amp_a, amp_b, amp_a_sha256, amp_b_sha256, calibration, training_input, safety, receptive_field, warnings, low_level_response: LowLevelResponseCheck, output_gain: "dict | None" = None):
    return {
        "hybrid_builder_version": HYBRID_BUILDER_VERSION, "git_commit": _git_commit(), "mode": "character",
        "amp_a": {"filename": Path(design.amp_a_path).name, "path": design.amp_a_path, "sha256": amp_a_sha256, "architecture": amp_a.architecture, "sample_rate": amp_a.sample_rate, "input_level_dbu": amp_a.input_level_dbu},
        "amp_b": {"filename": Path(design.amp_b_path).name, "path": design.amp_b_path, "sha256": amp_b_sha256, "architecture": amp_b.architecture, "sample_rate": amp_b.sample_rate, "input_level_dbu": amp_b.input_level_dbu},
        "design": design.to_dict(),
        "character_analysis": {"version": 2, "teacher_semantics_version": design.teacher_semantics_version, "amp_a_sha256": design.amp_a_sha256 or amp_a_sha256, "amp_b_sha256": design.amp_b_sha256 or amp_b_sha256, "analysis_a": design.analysis_a, "analysis_b": design.analysis_b, "tone_mix_b": design.tone_mix_b, "feel_mix_b": design.feel_mix_b, "drive_mix_b": design.drive_mix_b, "drive_curve": {"low": design.drive_low_mix_b, "mid": design.drive_mid_mix_b, "high": design.drive_high_mix_b}},
        "calibration": {"requested_mode": design.calibration_mode, "effective_mode": "auto" if calibration.applied else "raw", "reference_input_level_dbu": calibration.reference_input_level_dbu, "amp_a_compensation_db": calibration.amp_a_gain_db, "amp_b_compensation_db": calibration.amp_b_gain_db, "applied": calibration.applied, "warning": calibration.warning},
        "training_input": {"path": training_input.path, "sample_rate": training_input.sample_rate, "frame_count": training_input.frame_count, "sha256": training_input.sha256, "md5": training_input.md5, "detected_version": training_input.detected_version},
        "target": {"raw_sha256": safety.raw_sha256, "final_sha256": safety.final_sha256, "peak_before_safety_dbfs": safety.raw_peak_dbfs, "peak_after_safety_dbfs": safety.final_peak_dbfs, "global_safety_gain_reduction_db": safety.gain_reduction_db, "preview_limiter_used": False, "synthetic_latency_samples": 0},
        "training": {"status": "not yet run -- Character Blend defaults to high_def (120 epochs)", "recommended_epoch_preset": "high_def"},
        "cab": design.cab.to_dict() if design.cab else {"selected": False},
        "output_gain": output_gain or {"mode": design.output_gain_mode, "applied_gain_db": 0.0},
        "receptive_field": receptive_field, "warnings": warnings,
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
    a = render(amp_a, (official_input * db_to_amplitude(calibration.amp_a_gain_db + design.amp_a_input_gain_db)).astype(np.float32), input_info.sample_rate)
    b = render(amp_b, (official_input * db_to_amplitude(calibration.amp_b_gain_db + design.amp_b_input_gain_db)).astype(np.float32), input_info.sample_rate)
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

    # Shared post-combination output gain -- see hybrid.design.HybridDesign's
    # output_gain_mode/manual_output_gain_db docstring.
    if design.output_gain_mode == "manual":
        output_gain_db = design.manual_output_gain_db
        output_gain_peak_before_dbfs = check_audio(target_raw).peak_dbfs
    else:
        output_gain_db, output_gain_peak_before_dbfs = compute_auto_output_gain_db(target_raw, target_peak_dbfs)
    target_raw = apply_output_gain(target_raw, output_gain_db)

    check = check_audio(target_raw)
    if check.has_nan_or_inf: raise TrainingInputError("generated Character Blend target contains NaN/Inf -- aborting")
    if check.is_silent: warnings.append("generated Character Blend target is silent -- check amp/DI/design selection")
    target_final, reduction = apply_peak_ceiling(target_raw, target_peak_dbfs)
    validation_reference, validation_excerpt = build_export_validation_reference(
        design, amp_a, amp_b, calibration, official_input, input_info.sample_rate,
        output_gain_db=output_gain_db, peak_reduction_db=reduction,
    )
    input_out, raw_out, final_out, manifest_out = output_directory / "input.wav", output_directory / "hybrid_target_raw.wav", output_directory / "hybrid_target.wav", output_directory / "training_manifest.json"
    reference_out = output_directory / "export_validation_reference_input.wav"
    shutil.copyfile(official_input_path, input_out); sf.write(raw_out, target_raw.astype(np.float32), input_info.sample_rate, subtype="FLOAT"); sf.write(final_out, target_final.astype(np.float32), input_info.sample_rate, subtype="FLOAT"); sf.write(reference_out, validation_excerpt, input_info.sample_rate, subtype="FLOAT")
    safety = TargetSafetyReport(check.peak_dbfs, check.peak_dbfs - reduction, reduction, _sha256_file(raw_out), _sha256_file(final_out))
    receptive = compute_receptive_field_record(
        "character", amp_a, amp_b, input_info.sample_rate, design.cab,
        bounded_envelope_max_history_ms(), design.envelope_smoothing_ms,
    )
    available_rf = receptive.get("a2_receptive_field_samples_at_generation_time")
    if available_rf is not None and receptive.get(
        "formal_character_required_samples", receptive.get("hard_required_samples", 0)
    ) > available_rf:
        warnings.append(
            "Character processing extends beyond the standard A2 receptive field. Training will continue as an "
            "approximation; validate the Full and Lite exports against the frozen teacher and by listening."
        )
    output_gain_record = {
        "mode": design.output_gain_mode,
        "requested_manual_gain_db": design.manual_output_gain_db if design.output_gain_mode == "manual" else None,
        "peak_before_output_gain_dbfs": output_gain_peak_before_dbfs,
        "applied_gain_db": output_gain_db,
    }
    manifest = build_character_training_manifest(design, amp_a, amp_b, a_sha, b_sha, calibration, input_info, safety, receptive, warnings, low_level_response, output_gain=output_gain_record)
    validation_reference.update({
        "input_excerpt_path": reference_out.name,
        "input_excerpt_sha256": _sha256_file(reference_out),
        "source_training_input_sha256": input_info.sha256,
        "amp_a_sha256": a_sha,
        "amp_b_sha256": b_sha,
        "cab_sha256": design.cab.sha256 if design.cab else None,
    })
    manifest["export_validation_reference"] = validation_reference
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TrainingBundle(output_directory, input_out, raw_out, final_out, None, manifest_out, design, safety, input_info, warnings, manifest)
