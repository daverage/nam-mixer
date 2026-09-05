"""Real A2 training-target generation -- the actual implementation behind
`/api/generate` and `scripts/train_a2.py`'s input. See docs/phase3.md,
particularly sections 1 and 7-12, for the full rationale; the short version:

- Uses a FROZEN `hybrid.design.HybridDesign` (never recomputes auto-trim or
  crossover against the training input -- see hybrid/design.py).
- Feeds the OFFICIAL NAM training excitation to both source models at the
  design's reference (0 dB) profile level -- `design.design_reference_profile_gain_db`
  is recorded in the manifest for provenance but is NEVER applied to the
  actual training signal (docs/phase3.md section 1/23). Only per-model NAM
  calibration compensation (the same `resolve_calibration` rules
  `hybrid.pipeline.render_pair` uses) is applied.
- The crossover envelope is computed from the common, un-calibrated official
  input, mirroring render_pair()'s architecture exactly.
- Never applies `preview_safety_limiter` -- only `apply_peak_ceiling` (a
  single whole-file gain), and never normalizes/clips.
- Writes 32-bit float WAVs, never PCM16.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from .align import align_to_reference
from .blend import CrossoverConfig, blend
from .calibration import CalibrationResult, resolve_calibration
from .design import HybridDesign
from .envelope import (
    DEFAULT_BOUNDED_ENVELOPE_CONFIG,
    BoundedEnvelopeConfig,
    bounded_causal_envelope_db,
    bounded_envelope_max_history_ms,
)
from .input_profiles import db_to_amplitude
from .metadata import HybridMetadata
from .nam_loader import NamModel, load_nam
from .render import render
from .safety import apply_peak_ceiling, check_audio

REQUIRED_TRAINING_INPUT_SAMPLE_RATE = 48000

# The A2 trainer's own hard requirement is only that the target not clip
# (max(|y|) < 1.0, i.e. < 0 dBFS) -- and it already normalizes training
# output internally (to -18 dBFS RMS) before reversing that normalization on
# export, so there's no training-quality reason to force every hot target
# down to some fixed "safe" ceiling like -3 dBFS (docs/phase3.md review
# section 4). We therefore only ever apply the MINIMUM whole-file attenuation
# needed to bring a clipping/near-clipping target just under 0 dBFS -- a
# target that never reaches this ceiling is left completely untouched, so
# the trained model reproduces our hybrid's actual level as faithfully as
# possible rather than an arbitrarily quieter copy of it.
A2_TARGET_PEAK_CEILING_DBFS = -0.2

HYBRID_BUILDER_VERSION = "phase3-a2-v1"

# The official NAM v3.0.0 training/reamp input file, identified by its MD5 --
# the same value nam.train.core._detect_input_version uses internally to
# strong-match it (verified live against the real installed
# neural-amp-modeler==0.13.0, see requirements-training.txt). We require V3
# SPECIFICALLY, not merely "any recognized official input": the current
# official simplified trainer's own data checks are calibrated around V3's
# validation-signal layout (two ~9s repeated passages at the head/tail) and
# explicitly fail for other versions unless force-ignored -- which we never
# do (docs/phase3.md section 17). See
# https://github.com/sdatkinson/neural-amp-modeler for how to obtain it.
OFFICIAL_V3_INPUT_MD5 = "36cd1af62985c2fac3e654333e36431e"


class TrainingInputError(ValueError):
    """Raised when the supplied official training input, or the generated
    target derived from it, fails validation. Generation aborts rather than
    working around the problem -- see docs/phase3.md section 7."""


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return None


@dataclass
class TrainingInputInfo:
    path: str
    sample_rate: int
    frame_count: int
    sha256: str
    md5: Optional[str] = None
    detected_version: Optional[str] = None


def _detect_nam_input_version(path: Path) -> str:
    """Ask the installed `neural-amp-modeler` package itself to identify the
    training input -- NOT a filename guess. Verified against the actually
    installed neural-amp-modeler==0.13.0 (see requirements-training.txt):
    `nam.train.core._detect_input_version` MD5-hashes the file and matches it
    against its own table of known official reamp files (v1.0.0, v1.1.1,
    v2.0.0, v3.0.0, Proteus), the same mechanism `nam.train.core.train()`
    itself uses -- so a version reported here is authoritative, not a guess.

    If `neural-amp-modeler` isn't importable in THIS (Flask/runtime)
    environment (the normal case per CLAUDE.md), reports that plainly rather
    than guessing -- the authoritative check that actually gates training
    happens for real inside scripts/train_a2.py, which runs in the dedicated
    training environment where the package is installed (see
    requirements-training.txt), and that script aborts if this doesn't
    resolve to a known version.
    """
    try:
        from nam.train.core import _detect_input_version
    except ImportError:
        return "unverified (neural-amp-modeler not installed in this environment -- see scripts/train_a2.py)"
    try:
        version, strong_match = _detect_input_version(str(path))
    except Exception as exc:  # noqa: BLE001 -- report, don't guess at nam's internal error types
        return f"unrecognized ({exc})"
    return f"v{version.major}.{version.minor}.{version.patch}" + ("" if strong_match else " (weak match)")


def validate_training_input(path: str | Path) -> tuple[np.ndarray, TrainingInputInfo]:
    """Load and validate the official NAM V3 training input WAV: must exist,
    be mono, be at `REQUIRED_TRAINING_INPUT_SAMPLE_RATE`, and MD5-match the
    official v3.0.0 file exactly (`OFFICIAL_V3_INPUT_MD5`) -- see that
    constant's docstring for why V3 specifically, not "any recognized
    version". Does not resample or downmix -- a file that fails this is the
    wrong file, not something to silently coerce.
    """
    path = Path(path)
    if not path.is_file():
        raise TrainingInputError(f"official training input not found: {path}")

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise TrainingInputError(f"training input must be mono, got shape {audio.shape} for {path}")
    if sample_rate != REQUIRED_TRAINING_INPUT_SAMPLE_RATE:
        raise TrainingInputError(
            f"training input must be {REQUIRED_TRAINING_INPUT_SAMPLE_RATE} Hz, got {sample_rate} Hz for {path}"
        )
    if not np.all(np.isfinite(audio)):
        raise TrainingInputError(f"training input contains non-finite samples: {path}")

    md5 = _md5_file(path)
    if md5 != OFFICIAL_V3_INPUT_MD5:
        raise TrainingInputError(
            f"training input does not match the official NAM v3.0.0 input file "
            f"(expected MD5 {OFFICIAL_V3_INPUT_MD5}, got {md5} for {path}). Hybrid A2 generation "
            "requires the official V3 training input specifically -- see "
            "OFFICIAL_V3_INPUT_MD5's docstring and "
            "https://github.com/sdatkinson/neural-amp-modeler for how to obtain it."
        )

    info = TrainingInputInfo(
        path=str(path),
        sample_rate=sample_rate,
        frame_count=len(audio),
        sha256=_sha256_file(path),
        md5=md5,
        detected_version=_detect_nam_input_version(path),
    )
    return audio, info


@dataclass
class TargetSafetyReport:
    raw_peak_dbfs: float
    final_peak_dbfs: float
    gain_reduction_db: float
    raw_sha256: str
    final_sha256: str


@dataclass
class TrainingBundle:
    bundle_dir: Path
    input_path: Path
    hybrid_target_raw_path: Path
    hybrid_target_path: Path
    hybrid_metadata_path: Path
    training_manifest_path: Path
    design: HybridDesign
    safety: TargetSafetyReport
    training_input_info: TrainingInputInfo
    warnings: list[str]
    manifest: dict


def _hybrid_metadata_dict(design: HybridDesign) -> dict:
    level_match = "automatic" if abs(design.auto_trim_db) > 1e-9 else "manual"
    meta = HybridMetadata(
        amp_a=design.amp_a_path,
        amp_b=design.amp_b_path,
        crossover_dbfs=design.crossover_dbfs,
        transition_width_db=design.transition_width_db,
        amp_a_trim_db=0.0,
        amp_b_trim_db=design.effective_b_trim_db,
        level_match=level_match,
        blend_algorithm=design.blend_algorithm,
        instrument_type=design.instrument_type,
        input_profile_id=design.design_reference_profile_id,
        input_profile_gain_db=design.design_reference_profile_gain_db,
        calibration_mode=design.calibration_effective_mode,
        reference_input_level_dbu=design.reference_input_level_dbu,
        amp_a_input_level_dbu=design.amp_a_input_level_dbu,
        amp_b_input_level_dbu=design.amp_b_input_level_dbu,
        amp_a_calibration_gain_db=design.amp_a_calibration_gain_db,
        amp_b_calibration_gain_db=design.amp_b_calibration_gain_db,
    )
    d = meta.to_dict()
    # Explicit per docs/phase3.md section 23 -- must never become ambiguous later.
    d["hybrid"]["pickup_profile_applied_to_training_input"] = False
    return d


def build_training_manifest(
    design: HybridDesign,
    amp_a: NamModel,
    amp_b: NamModel,
    amp_a_sha256: str,
    amp_b_sha256: str,
    calibration: CalibrationResult,
    training_input: TrainingInputInfo,
    safety: TargetSafetyReport,
    alignment_offset_samples: int,
    envelope_config: BoundedEnvelopeConfig,
    warnings: list[str],
    training_env: Optional[dict] = None,
) -> dict:
    return {
        "hybrid_builder_version": HYBRID_BUILDER_VERSION,
        "git_commit": _git_commit(),
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
            "crossover_dbfs": design.crossover_dbfs,
            "transition_width_db": design.transition_width_db,
            "blend_algorithm": design.blend_algorithm,
            "envelope_algorithm": "bounded_causal_envelope_db",
            "envelope_rms_window_ms": envelope_config.rms_window_ms,
            "envelope_attack_avg_ms": envelope_config.attack_avg_ms,
            "envelope_release_window_ms": envelope_config.release_window_ms,
            "envelope_release_range_db": envelope_config.release_range_db,
            "envelope_max_history_ms": bounded_envelope_max_history_ms(envelope_config),
            "original_auto_trim_db": design.auto_trim_db,
            "manual_trim_db": design.manual_b_trim_db,
            "frozen_effective_b_trim_db": design.effective_b_trim_db,
            "alignment_enabled": design.alignment_enabled,
            "alignment_offset_samples": alignment_offset_samples,
            "design_di_file": design.design_di_file,
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
        "warnings": warnings,
    }


def generate_training_bundle(
    design: HybridDesign,
    official_input_path: str | Path,
    output_directory: str | Path,
    target_peak_dbfs: float = A2_TARGET_PEAK_CEILING_DBFS,
    envelope_config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG,
) -> TrainingBundle:
    """Generate a self-contained, reproducible A2 training bundle from a
    FROZEN `HybridDesign` and the official NAM training excitation. Pure
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

    # NOTE: design.design_reference_profile_gain_db is deliberately NOT
    # applied here -- see module docstring / docs/phase3.md section 1.
    amp_a_input = (official_input * db_to_amplitude(calib.amp_a_gain_db)).astype(np.float32)
    amp_b_input = (official_input * db_to_amplitude(calib.amp_b_gain_db)).astype(np.float32)

    amp_a_render = render(amp_a, amp_a_input, input_info.sample_rate)
    amp_b_render = render(amp_b, amp_b_input, input_info.sample_rate)

    # Crossover envelope sees the common, uncalibrated official input --
    # BEFORE the per-model calibration split above (docs/phase3.md section 8).
    envelope_db = bounded_causal_envelope_db(official_input, input_info.sample_rate, envelope_config)

    amp_b_aligned, alignment_offset = align_to_reference(
        amp_a_render, amp_b_render, enabled=design.alignment_enabled
    )

    config = CrossoverConfig(
        crossover_dbfs=design.crossover_dbfs,
        transition_width_db=design.transition_width_db,
        amp_b_trim_db=design.effective_b_trim_db,  # frozen -- not recomputed
    )
    hybrid_raw, _t_curve = blend(envelope_db, amp_a_render, amp_b_aligned, config)

    if len(hybrid_raw) != len(official_input):
        raise TrainingInputError(
            f"generated target length {len(hybrid_raw)} != training input length "
            f"{len(official_input)} -- input/target must be exactly sample-aligned"
        )

    safety_check = check_audio(hybrid_raw)
    if safety_check.has_nan_or_inf:
        raise TrainingInputError("generated hybrid target contains NaN/Inf -- aborting")
    if safety_check.is_silent:
        warnings.append("generated hybrid target is silent -- check amp/DI/design selection")

    raw_peak_dbfs = safety_check.peak_dbfs
    hybrid_final, gain_reduction_db = apply_peak_ceiling(hybrid_raw, target_peak_dbfs)
    final_peak_dbfs = raw_peak_dbfs - gain_reduction_db if np.isfinite(raw_peak_dbfs) else raw_peak_dbfs

    input_out = output_directory / "input.wav"
    raw_out = output_directory / "hybrid_target_raw.wav"
    final_out = output_directory / "hybrid_target.wav"
    metadata_out = output_directory / "hybrid.hybrid.json"
    manifest_out = output_directory / "training_manifest.json"

    # Byte-for-byte copy, NOT a re-encode: the official trainer's own input-
    # version detection (nam.train.core._detect_input_version) strong-matches
    # by hashing the file's exact bytes, and training_manifest.json's
    # training_input.sha256 is computed from the ORIGINAL file (see
    # validate_training_input above) -- re-writing it through soundfile as
    # float32 would silently change both the byte-for-byte content and the
    # hash, breaking the very provenance check scripts/train_a2.py relies on.
    shutil.copyfile(official_input_path, input_out)
    sf.write(raw_out, hybrid_raw.astype(np.float32), input_info.sample_rate, subtype="FLOAT")
    sf.write(final_out, hybrid_final.astype(np.float32), input_info.sample_rate, subtype="FLOAT")

    # Hash the WRITTEN FILES, not the in-memory arrays: scripts/train_a2.py
    # (and anyone else checking provenance) can only ever re-hash the file on
    # disk, and _sha256_array's raw array bytes don't include the WAV
    # container -- hashing the array here would silently record a hash that
    # can never be reproduced by re-hashing hybrid_target.wav itself.
    safety_report = TargetSafetyReport(
        raw_peak_dbfs=raw_peak_dbfs,
        final_peak_dbfs=final_peak_dbfs,
        gain_reduction_db=gain_reduction_db,
        raw_sha256=_sha256_file(raw_out),
        final_sha256=_sha256_file(final_out),
    )

    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(_hybrid_metadata_dict(design), f, indent=2)

    manifest = build_training_manifest(
        design=design, amp_a=amp_a, amp_b=amp_b,
        amp_a_sha256=amp_a_sha, amp_b_sha256=amp_b_sha,
        calibration=calib, training_input=input_info, safety=safety_report,
        alignment_offset_samples=alignment_offset, envelope_config=envelope_config,
        warnings=warnings,
    )
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return TrainingBundle(
        bundle_dir=output_directory,
        input_path=input_out,
        hybrid_target_raw_path=raw_out,
        hybrid_target_path=final_out,
        hybrid_metadata_path=metadata_out,
        training_manifest_path=manifest_out,
        design=design,
        safety=safety_report,
        training_input_info=input_info,
        warnings=warnings,
        manifest=manifest,
    )
