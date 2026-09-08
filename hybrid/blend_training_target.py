"""Real A2 training-target generation for FIXED BLEND design mode -- the
Blend-mode counterpart to `hybrid.training_target.generate_training_bundle`.
See docs/blend-mode.md "FIXED BLEND TRAINING TARGET".

Deliberately NOT a rewrite of the Hybrid generator: this reuses the shared
bundle-writing/safety/provenance infrastructure from `hybrid.training_target`
(`validate_training_input`, `TrainingInputError`, `TargetSafetyReport`,
`maybe_bake_cab`, `compute_receptive_field_record`, hashing helpers) so the
two design modes' A2 bundles are byte-for-byte comparable in structure and
never silently drift apart -- only the actual combination step (fixed mix
instead of level-driven crossfade) and the manifest's `design`/`mode` section
differ.

Same rules as Hybrid target generation:
- Uses a FROZEN `hybrid.fixed_blend.BlendDesign` (never recomputes the
  active-playing auto trim against the official training input).
- Feeds the OFFICIAL NAM training excitation to both source models with only
  per-model NAM calibration compensation applied -- no input-profile gain,
  no test gain.
- Never applies `preview_safety_limiter`, only `apply_peak_ceiling`.
- Writes 32-bit float WAVs, preserves the existing bundle filenames
  (input.wav / hybrid_target.wav / training_manifest.json) so the same
  local/Kaggle training infrastructure works unchanged for both modes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .align import align_to_reference
from .calibration import resolve_calibration
from .fixed_blend import BlendDesign
from .input_profiles import db_to_amplitude
from .nam_loader import load_nam
from .render import render
from .safety import apply_output_gain, apply_peak_ceiling, check_audio, compute_auto_output_gain_db
from .training_target import (
    A2_TARGET_PEAK_CEILING_DBFS,
    TargetSafetyReport,
    TrainingBundle,
    TrainingInputError,
    _git_commit,
    _sha256_file,
    compute_receptive_field_record,
    maybe_bake_cab,
    validate_training_input,
)

HYBRID_BUILDER_VERSION = "phase3-a2-v1"


def build_blend_training_manifest(
    design: BlendDesign,
    amp_a,
    amp_b,
    amp_a_sha256: str,
    amp_b_sha256: str,
    calibration,
    training_input,
    safety: TargetSafetyReport,
    alignment_offset_samples: int,
    warnings: list[str],
    receptive_field: dict,
    training_env: Optional[dict] = None,
    output_gain: Optional[dict] = None,
) -> dict:
    return {
        "hybrid_builder_version": HYBRID_BUILDER_VERSION,
        "git_commit": _git_commit(),
        "mode": "blend",
        "amp_a": {
            "filename": Path(design.amp_a_path).name,
            "path": design.amp_a_path,
            "sha256": amp_a_sha256,
            "architecture": amp_a.architecture,
            "sample_rate": amp_a.sample_rate,
            "input_level_dbu": amp_a.input_level_dbu,
            "output_level_dbu": amp_a.output_level_dbu,
        },
        "amp_b": {
            "filename": Path(design.amp_b_path).name,
            "path": design.amp_b_path,
            "sha256": amp_b_sha256,
            "architecture": amp_b.architecture,
            "sample_rate": amp_b.sample_rate,
            "input_level_dbu": amp_b.input_level_dbu,
            "output_level_dbu": amp_b.output_level_dbu,
        },
        "design": {
            "instrument_type": design.instrument_type,
            "design_reference_profile_id": design.design_reference_profile_id,
            "design_reference_profile_gain_db": design.design_reference_profile_gain_db,
            "pickup_profile_applied_to_training_input": False,
            "mix_b": design.mix_b,
            "mix_a": design.mix_a,
            "original_auto_trim_db": design.auto_trim_db,
            "manual_trim_db": design.manual_b_trim_db,
            "frozen_effective_b_trim_db": design.effective_b_trim_db,
            "alignment_enabled": design.alignment_enabled,
            "alignment_offset_samples": alignment_offset_samples,
            "design_di_file": design.design_di_file,
            "amp_a_input_gain_db": design.amp_a_input_gain_db,
            "amp_b_input_gain_db": design.amp_b_input_gain_db,
        },
        "calibration": {
            "requested_mode": design.calibration_mode,
            "effective_mode": "auto" if calibration.applied else "raw",
            "reference_input_level_dbu": calibration.reference_input_level_dbu,
            "amp_a_compensation_db": calibration.amp_a_gain_db,
            "amp_b_compensation_db": calibration.amp_b_gain_db,
            "applied": calibration.applied,
            "warning": calibration.warning,
        },
        "training_input": {
            "path": training_input.path,
            "sample_rate": training_input.sample_rate,
            "frame_count": training_input.frame_count,
            "sha256": training_input.sha256,
            "md5": training_input.md5,
            "detected_version": training_input.detected_version,
        },
        "target": {
            "raw_sha256": safety.raw_sha256,
            "final_sha256": safety.final_sha256,
            "peak_before_safety_dbfs": safety.raw_peak_dbfs,
            "peak_after_safety_dbfs": safety.final_peak_dbfs,
            "global_safety_gain_reduction_db": safety.gain_reduction_db,
            "preview_limiter_used": False,
            "synthetic_latency_samples": 0,
        },
        "training": training_env or {
            "status": "not yet run -- see scripts/train_a2.py",
            "neural_amp_modeler_version": None,
            "torch_version": None,
            "pytorch_lightning_version": None,
            "python_version": None,
            "a2_config_identifier": None,
            "training_settings": None,
            "device": None,
        },
        "cab": design.cab.to_dict() if design.cab else {"selected": False},
        "output_gain": output_gain or {"mode": design.output_gain_mode, "applied_gain_db": 0.0},
        "receptive_field": receptive_field,
        "warnings": warnings,
    }


def generate_blend_training_bundle(
    design: BlendDesign,
    official_input_path: str | Path,
    output_directory: str | Path,
    target_peak_dbfs: float = A2_TARGET_PEAK_CEILING_DBFS,
) -> TrainingBundle:
    """Generate a self-contained, reproducible A2 training bundle from a
    FROZEN `BlendDesign` and the official NAM training excitation. Pure
    function of its arguments -- no Flask dependency, directly testable.
    """
    official_input_path = Path(official_input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    official_input, input_info = validate_training_input(official_input_path)

    amp_a = load_nam(design.amp_a_path)
    amp_b = load_nam(design.amp_b_path)
    amp_a_sha = _sha256_file(design.amp_a_path)
    amp_b_sha = _sha256_file(design.amp_b_path)

    warnings: list[str] = []

    calib = resolve_calibration(
        design.calibration_mode, design.reference_input_level_dbu,
        amp_a.input_level_dbu, amp_b.input_level_dbu,
    )
    if calib.warning:
        warnings.append(calib.warning)

    # design.design_reference_profile_gain_db is DESIGN CONTEXT ONLY --
    # never re-applied to the official training excitation, exactly like
    # Hybrid mode (see hybrid/training_target.py module docstring).
    # amp_a_input_gain_db/amp_b_input_gain_db ARE applied, same as Hybrid
    # mode -- see hybrid.pipeline.RenderedPair's docstring.
    amp_a_input = (official_input * db_to_amplitude(calib.amp_a_gain_db + design.amp_a_input_gain_db)).astype(np.float32)
    amp_b_input = (official_input * db_to_amplitude(calib.amp_b_gain_db + design.amp_b_input_gain_db)).astype(np.float32)

    amp_a_render = render(amp_a, amp_a_input, input_info.sample_rate)
    amp_b_render = render(amp_b, amp_b_input, input_info.sample_rate)

    amp_b_aligned, alignment_offset = align_to_reference(
        amp_a_render, amp_b_render, enabled=design.alignment_enabled
    )

    n = min(len(amp_a_render), len(amp_b_aligned))
    a = amp_a_render[:n]
    b = amp_b_aligned[:n] * (10.0 ** (design.effective_b_trim_db / 20.0))  # frozen -- not recomputed
    mix_b = design.mix_b
    blend_raw = (a * (1.0 - mix_b) + b * mix_b).astype(np.float32)

    if len(blend_raw) != len(official_input):
        raise TrainingInputError(
            f"generated target length {len(blend_raw)} != training input length "
            f"{len(official_input)} -- input/target must be exactly sample-aligned"
        )

    # Baked cab (if any) runs AFTER the fixed-mix combination, BEFORE safety
    # -- see docs/blend-mode.md "SHARED CABINET IR STAGE".
    blend_raw = maybe_bake_cab(blend_raw, design.cab, input_info.sample_rate)

    # Shared post-combination output gain -- see hybrid.design.HybridDesign's
    # output_gain_mode/manual_output_gain_db docstring and the mirror-image
    # comment in hybrid/training_target.py's generate_training_bundle.
    if design.output_gain_mode == "manual":
        output_gain_db = design.manual_output_gain_db
        output_gain_peak_before_dbfs = check_audio(blend_raw).peak_dbfs
    else:
        output_gain_db, output_gain_peak_before_dbfs = compute_auto_output_gain_db(blend_raw, target_peak_dbfs)
    blend_raw = apply_output_gain(blend_raw, output_gain_db)

    receptive_field = compute_receptive_field_record(
        "blend", amp_a, amp_b, input_info.sample_rate, design.cab,
    )

    safety_check = check_audio(blend_raw)
    if safety_check.has_nan_or_inf:
        raise TrainingInputError("generated blend target contains NaN/Inf -- aborting")
    if safety_check.is_silent:
        warnings.append("generated blend target is silent -- check amp/DI/design selection")

    raw_peak_dbfs = safety_check.peak_dbfs
    blend_final, gain_reduction_db = apply_peak_ceiling(blend_raw, target_peak_dbfs)
    final_peak_dbfs = raw_peak_dbfs - gain_reduction_db if np.isfinite(raw_peak_dbfs) else raw_peak_dbfs

    input_out = output_directory / "input.wav"
    raw_out = output_directory / "hybrid_target_raw.wav"
    final_out = output_directory / "hybrid_target.wav"  # legacy/internal filename, see docs/blend-mode.md
    manifest_out = output_directory / "training_manifest.json"

    shutil.copyfile(official_input_path, input_out)
    sf.write(raw_out, blend_raw.astype(np.float32), input_info.sample_rate, subtype="FLOAT")
    sf.write(final_out, blend_final.astype(np.float32), input_info.sample_rate, subtype="FLOAT")

    safety_report = TargetSafetyReport(
        raw_peak_dbfs=raw_peak_dbfs,
        final_peak_dbfs=final_peak_dbfs,
        gain_reduction_db=gain_reduction_db,
        raw_sha256=_sha256_file(raw_out),
        final_sha256=_sha256_file(final_out),
    )

    output_gain_record = {
        "mode": design.output_gain_mode,
        "requested_manual_gain_db": design.manual_output_gain_db if design.output_gain_mode == "manual" else None,
        "peak_before_output_gain_dbfs": output_gain_peak_before_dbfs,
        "applied_gain_db": output_gain_db,
    }

    manifest = build_blend_training_manifest(
        design=design, amp_a=amp_a, amp_b=amp_b,
        amp_a_sha256=amp_a_sha, amp_b_sha256=amp_b_sha,
        calibration=calib, training_input=input_info, safety=safety_report,
        alignment_offset_samples=alignment_offset,
        warnings=warnings, receptive_field=receptive_field, output_gain=output_gain_record,
    )
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return TrainingBundle(
        bundle_dir=output_directory,
        input_path=input_out,
        hybrid_target_raw_path=raw_out,
        hybrid_target_path=final_out,
        hybrid_metadata_path=None,  # no NAM-metadata-shaped sidecar for Blend -- see module docstring
        training_manifest_path=manifest_out,
        design=design,
        safety=safety_report,
        training_input_info=input_info,
        warnings=warnings,
        manifest=manifest,
    )
