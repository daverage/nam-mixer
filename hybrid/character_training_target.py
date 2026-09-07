"""Official-input A2 bundle generation for deterministic Character Blend."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from .calibration import resolve_calibration
from .character_blend import CharacterBlendDesign, build_character_blend
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


def build_character_training_manifest(design, amp_a, amp_b, amp_a_sha256, amp_b_sha256, calibration, training_input, safety, receptive_field, warnings):
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
    target_raw = maybe_bake_cab(target_raw, design.cab, input_info.sample_rate)
    check = check_audio(target_raw)
    if check.has_nan_or_inf: raise TrainingInputError("generated Character Blend target contains NaN/Inf -- aborting")
    if check.is_silent: warnings.append("generated Character Blend target is silent -- check amp/DI/design selection")
    target_final, reduction = apply_peak_ceiling(target_raw, target_peak_dbfs)
    input_out, raw_out, final_out, manifest_out = output_directory / "input.wav", output_directory / "hybrid_target_raw.wav", output_directory / "hybrid_target.wav", output_directory / "training_manifest.json"
    shutil.copyfile(official_input_path, input_out); sf.write(raw_out, target_raw.astype(np.float32), input_info.sample_rate, subtype="FLOAT"); sf.write(final_out, target_final.astype(np.float32), input_info.sample_rate, subtype="FLOAT")
    safety = TargetSafetyReport(check.peak_dbfs, check.peak_dbfs - reduction, reduction, _sha256_file(raw_out), _sha256_file(final_out))
    receptive = compute_receptive_field_record("character", amp_a, amp_b, input_info.sample_rate, design.cab, bounded_envelope_max_history_ms())
    manifest = build_character_training_manifest(design, amp_a, amp_b, a_sha, b_sha, calibration, input_info, safety, receptive, warnings)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return TrainingBundle(output_directory, input_out, raw_out, final_out, None, manifest_out, design, safety, input_info, warnings, manifest)
