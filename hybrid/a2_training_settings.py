"""Single source of truth for the A2 (PackedWaveNet) training settings used by
BOTH the local trainer (`scripts/train_a2.py`) and the Kaggle cloud worker
(`cloud/kaggle/train_a2_cloud.py`) -- see docs/kaggle_training.md.

Pure data, no torch/nam import, safe to import from the normal torch-free
Flask/runtime environment as well as both training environments. The whole
point of this module is to make local/cloud drift structurally impossible:
both trainers import the same frozen dataclass instances rather than each
hard-coding their own copy of "epochs=100, batch_size=16, ...".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class A2TrainingSettings:
    epochs: int
    batch_size: int
    ny: int
    seed: int
    latency: int
    ignore_checks: bool
    fast_dev_run: bool
    silent: bool


# Quality/speed presets for a real (non-smoke-test) training run -- draft for
# a fast preview, standard for normal use, high_def for the best result at
# the cost of a longer run. Every non-epochs setting (batch_size/ny/seed/
# latency/etc, see _full_settings below) stays identical across presets --
# only the number of epochs differs.
A2_EPOCH_PRESETS: dict[str, int] = {
    "draft": 20,
    "standard": 60,
    "high_def": 120,
}
DEFAULT_EPOCH_PRESET = "standard"


def _full_settings(epochs: int) -> A2TrainingSettings:
    return A2TrainingSettings(
        epochs=epochs,
        batch_size=16,
        ny=8192,
        seed=0,
        latency=0,  # synthetic target -- authoritatively zero, never auto-detected
        ignore_checks=False,
        fast_dev_run=False,
        silent=True,
    )


def settings_for_preset(preset: str) -> A2TrainingSettings:
    """Full (non-smoke-test) training settings for one of A2_EPOCH_PRESETS'
    named quality levels ("draft"/"standard"/"high_def"). Raises ValueError
    on an unrecognized preset name rather than silently falling back --
    an invalid preset should never quietly train at the wrong length."""
    if preset not in A2_EPOCH_PRESETS:
        raise ValueError(f"unknown A2 epoch preset {preset!r} -- choose one of {sorted(A2_EPOCH_PRESETS)}")
    return _full_settings(A2_EPOCH_PRESETS[preset])


# Normal, full-quality training run at the default preset -- kept for
# backward compatibility with callers that don't need preset selection.
A2_TRAINING_SETTINGS = settings_for_preset(DEFAULT_EPOCH_PRESET)

# Fast development/smoke-test run only -- never the final model (docs/phase3.md
# section 15, docs/kaggle_training.md "real end-to-end test" section).
A2_QUICK_SETTINGS = A2TrainingSettings(
    epochs=1,
    batch_size=A2_TRAINING_SETTINGS.batch_size,
    ny=A2_TRAINING_SETTINGS.ny,
    seed=A2_TRAINING_SETTINGS.seed,
    latency=A2_TRAINING_SETTINGS.latency,
    ignore_checks=A2_TRAINING_SETTINGS.ignore_checks,
    fast_dev_run=True,
    silent=A2_TRAINING_SETTINGS.silent,
)

# Pinned trainer package version -- both environments must install exactly
# this (requirements-training.txt locally; cloud/kaggle/train_a2_cloud.py
# installs it explicitly inside the Kaggle kernel).
NEURAL_AMP_MODELER_VERSION = "0.13.0"

# The official NAM v3.0.0 training/reamp input file's MD5 -- see
# hybrid/training_target.py's OFFICIAL_V3_INPUT_MD5 docstring for how this was
# verified. Duplicated here (rather than imported) because the Kaggle cloud
# worker is deliberately self-contained (see cloud/kaggle/train_a2_cloud.py's
# module docstring) and cannot import hybrid/training_target.py, which pulls
# in soundfile/numpy assumptions tied to this repo's package layout. Tested
# for equality against the authoritative constant in
# tests/test_a2_training_settings.py so the two can never silently diverge.
OFFICIAL_V3_INPUT_MD5 = "36cd1af62985c2fac3e654333e36431e"


def settings_for(quick: bool) -> A2TrainingSettings:
    return A2_QUICK_SETTINGS if quick else A2_TRAINING_SETTINGS


def user_metadata_kwargs(manifest: dict) -> dict:
    """Build the plain-dict kwargs for `nam.models.metadata.UserMetadata` from
    a training_manifest.json dict -- pure Python, no nam/torch import, so both
    `scripts/train_a2.py` (local) and `cloud/kaggle/train_a2_cloud.py` (cloud)
    can share this exact logic instead of maintaining two copies that could
    silently drift (see docs/kaggle_training.md). Callers construct the real
    `UserMetadata(**user_metadata_kwargs(manifest))` themselves, after
    importing `nam.models.metadata` in their own environment.

    Deliberately omits gear_type/tone_type/output_level_dbu semantics that
    require the nam package's own enums -- callers that want GearType.AMP set
    it themselves; this only supplies the plain string/number/None fields.
    """
    from pathlib import Path

    amp_a_name = Path(manifest.get("amp_a", {}).get("filename", "Amp A")).stem
    amp_b_name = Path(manifest.get("amp_b", {}).get("filename", "Amp B")).stem
    calibration = manifest.get("calibration", {})

    # Only report input_level_dbu when calibration was genuinely applied to
    # BOTH source models -- never invent one for a Raw-fallback pair
    # (docs/phase3.md section 9).
    input_level_dbu = calibration.get("reference_input_level_dbu") if calibration.get("applied") else None

    mode = manifest.get("mode", "hybrid")
    if mode == "blend":
        mix_b = manifest.get("design", {}).get("mix_b")
        ratio = f" {round((1 - mix_b) * 100)}-{round(mix_b * 100)}" if mix_b is not None else ""
        name = f"Blend {amp_a_name} + {amp_b_name}{ratio}"
        gear_model = f"{amp_a_name} + {amp_b_name}{ratio}"
    elif mode == "character":
        name = f"Character Blend {amp_a_name} + {amp_b_name}"
        gear_model = f"Character Blend {amp_a_name} + {amp_b_name}"
    else:
        name = f"Hybrid {amp_a_name} -> {amp_b_name}"
        gear_model = f"{amp_a_name} -> {amp_b_name}"

    # `model_name` is supplied by the builder UI/API and is persisted in the
    # manifest.  Prefer it for the name displayed by NAM tools; retain the
    # source-model-derived fallback for legacy manifests.
    model_name = str(manifest.get("model_name") or "").strip() or name

    return {
        "name": model_name,
        "modeled_by": "Hybrid NAM Builder",
        "gear_make": "Hybrid" if mode == "hybrid" else "Character Blend" if mode == "character" else "Blend",
        "gear_model": gear_model,
        # tone_type/output_level_dbu deliberately absent -- see
        # scripts/train_a2.py's _build_user_metadata docstring for why.
        "input_level_dbu": input_level_dbu,
    }
