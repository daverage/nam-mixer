"""Training-target generation for the NAM Tools single-capture cab embed flow.

This is the one-source counterpart to ``training_target`` and
``blend_training_target``.  It deliberately writes the same bundle filenames
and manifest sections so the existing local and Kaggle A2 trainers can consume
it without a separate training implementation.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core.cab_ir import CabDesign
from ..core.nam_loader import load_nam
from ..core.receptive_field import ReceptiveFieldUnavailable, compute_source_nam_receptive_field
from ..core.render import render
from ..core.safety import apply_peak_ceiling, check_audio
from .training_target import (
    A2_TARGET_PEAK_CEILING_DBFS,
    HYBRID_BUILDER_VERSION,
    TargetSafetyReport,
    TrainingBundle,
    TrainingInputError,
    _git_commit,
    _sha256_file,
    manifest_amp_record,
    manifest_target_record,
    manifest_training_input_record,
    manifest_untrained_record,
    maybe_bake_cab,
    validate_training_input,
)


@dataclass(frozen=True)
class CabEmbedDesign:
    source_nam_path: str
    cab: CabDesign
    model_name: str


def _receptive_field_record(source, cab: CabDesign, sample_rate: int) -> dict:
    source_samples = None
    unavailable = None
    try:
        source_samples = compute_source_nam_receptive_field(source)
    except ReceptiveFieldUnavailable as exc:
        unavailable = str(exc)

    from ..core.cab_ir import get_frozen_prepared_cab_ir

    # The convolution immediately before this record uses the same frozen IR;
    # failure here is therefore a real provenance error, not optional metadata.
    prepared = get_frozen_prepared_cab_ir(cab, sample_rate)
    fir_length = prepared.prepared_frame_count
    fir_history = max(0, fir_length - 1) if fir_length else (cab.fir_history_samples or 0)
    formal_total = source_samples + fir_history if source_samples is not None else None
    return {
        "mode": "cab_embed",
        "branch_samples": {"source": source_samples},
        "hard_required_samples": source_samples,
        "cab": {
            "export_mode": cab.export_mode,
            "baked": True,
            "fir_length_samples": fir_length,
            "fir_history_samples": fir_history,
            "preparation_mode": cab.preparation_mode,
            "leading_silence_threshold_db": cab.leading_silence_threshold_db,
            "sha256": cab.sha256,
        },
        "formal_total_required_samples": formal_total,
        "cab_requires_approximation": None,
        "unavailable": unavailable,
        "base_required_samples": source_samples,
        "cab_fir_serial_samples": fir_history,
        "total_required_samples": formal_total,
    }


def generate_cab_embed_training_bundle(
    design: CabEmbedDesign,
    official_input_path: str | Path,
    output_directory: str | Path,
    target_peak_dbfs: float = A2_TARGET_PEAK_CEILING_DBFS,
) -> TrainingBundle:
    """Render one NAM followed by its frozen cabinet and write an A2 bundle."""
    official_input_path = Path(official_input_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    official_input, input_info = validate_training_input(official_input_path)
    source = load_nam(design.source_nam_path)
    if source.sample_rate is not None and int(source.sample_rate) != input_info.sample_rate:
        raise TrainingInputError(
            f"source NAM sample rate {int(source.sample_rate)} does not match "
            f"the {input_info.sample_rate} Hz training input"
        )

    rendered = render(source, official_input, input_info.sample_rate)
    if len(rendered) != len(official_input):
        raise TrainingInputError(
            f"source render length {len(rendered)} != training input length {len(official_input)}"
        )
    target_raw = maybe_bake_cab(rendered, design.cab, input_info.sample_rate)
    safety_check = check_audio(target_raw)
    if safety_check.has_nan_or_inf:
        raise TrainingInputError("cab-embedded target contains NaN/Inf -- aborting")
    if safety_check.is_silent:
        raise TrainingInputError("cab-embedded target is silent -- check the source NAM and cabinet IR")

    raw_peak_dbfs = safety_check.peak_dbfs
    target_final, gain_reduction_db = apply_peak_ceiling(target_raw, target_peak_dbfs)
    final_peak_dbfs = raw_peak_dbfs - gain_reduction_db if np.isfinite(raw_peak_dbfs) else raw_peak_dbfs

    input_out = output_directory / "input.wav"
    raw_out = output_directory / "hybrid_target_raw.wav"
    final_out = output_directory / "hybrid_target.wav"
    design_out = output_directory / "cab_embed_design.json"
    manifest_out = output_directory / "training_manifest.json"
    shutil.copyfile(official_input_path, input_out)
    sf.write(raw_out, np.asarray(target_raw, dtype=np.float32), input_info.sample_rate, subtype="FLOAT")
    sf.write(final_out, np.asarray(target_final, dtype=np.float32), input_info.sample_rate, subtype="FLOAT")

    safety = TargetSafetyReport(
        raw_peak_dbfs=raw_peak_dbfs,
        final_peak_dbfs=final_peak_dbfs,
        gain_reduction_db=gain_reduction_db,
        raw_sha256=_sha256_file(raw_out),
        final_sha256=_sha256_file(final_out),
    )
    source_sha = _sha256_file(design.source_nam_path)
    manifest = {
        "hybrid_builder_version": HYBRID_BUILDER_VERSION,
        "git_commit": _git_commit(),
        "mode": "cab_embed",
        "model_name": design.model_name,
        "amp_a": manifest_amp_record(design.source_nam_path, source, source_sha),
        "design": {
            "source_nam_path": design.source_nam_path,
            "source_nam_sha256": source_sha,
            "operation": "single_capture_learned_cab",
        },
        "calibration": {
            "requested_mode": "preserve_source",
            "effective_mode": "source_native",
            "reference_input_level_dbu": source.input_level_dbu,
            "preserved_source_input_level_dbu": source.input_level_dbu,
            "applied": False,
            "warning": None,
        },
        "training_input": manifest_training_input_record(input_info),
        "target": manifest_target_record(safety),
        "training": manifest_untrained_record(),
        "cab": design.cab.to_dict(),
        "output_gain": {
            "mode": "safety_ceiling_only",
            "applied_gain_db": -gain_reduction_db,
        },
        "receptive_field": _receptive_field_record(source, design.cab, input_info.sample_rate),
        "warnings": [],
    }
    design_out.write_text(json.dumps({
        "source_nam_path": design.source_nam_path,
        "source_nam_sha256": source_sha,
        "model_name": design.model_name,
        "cab": design.cab.to_dict(),
    }, indent=2), encoding="utf-8")
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return TrainingBundle(
        bundle_dir=output_directory,
        input_path=input_out,
        hybrid_target_raw_path=raw_out,
        hybrid_target_path=final_out,
        hybrid_metadata_path=design_out,
        training_manifest_path=manifest_out,
        design=design,
        safety=safety,
        training_input_info=input_info,
        warnings=[],
        manifest=manifest,
    )
