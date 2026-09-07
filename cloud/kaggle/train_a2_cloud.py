#!/usr/bin/env python3
"""Kaggle GPU cloud worker for A2 training -- runs INSIDE a private Kaggle
kernel, never in this repo's normal environments. See docs/kaggle_training.md.

This script is deliberately SELF-CONTAINED: it does not import `hybrid/` or
depend on the native `nam_render` C++ tool (`hybrid/render.py`'s shell-out
target), because the Kaggle sandbox has neither this repo's package layout
nor a way to build that binary, and training itself only needs
`nam.train.core`, never our own NAMCore wrapper -- NAMCore verification of
the returned model happens back on the local machine
(`hybrid.kaggle_training.validate_downloaded_model`), exactly like
`scripts/train_a2.py` does for a local run. The few constants that must stay
identical to the local trainer (official V3 input MD5, training
hyperparameters) are duplicated here in literal form and are checked for
equality against the authoritative `hybrid/a2_training_settings.py` values by
tests/test_a2_training_settings.py on the machine that ships this file, so
the two can never silently diverge without a failing test.

Expects, next to itself in `/kaggle/working` (pushed alongside via the
private dataset attached to this kernel):
    input.wav                official NAM V3 training excitation
    hybrid_target.wav        the synthetic hybrid target, exactly aligned
    training_manifest.json   provenance -- used to build UserMetadata
    cloud_job.json           {job_id, design_id, accelerator}

Never bypasses nam.train.core.train()'s own data-validation checks. Never
writes a fake success result on failure -- exits non-zero instead.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants shared with hybrid/a2_training_settings.py -- kept as literals
# here (see module docstring) rather than imported, and cross-checked by
# tests/test_a2_training_settings.py.
# ---------------------------------------------------------------------------
NEURAL_AMP_MODELER_VERSION = "0.13.0"
OFFICIAL_V3_INPUT_MD5 = "36cd1af62985c2fac3e654333e36431e"

REQUIRED_SAMPLE_RATE = 48000


class CloudTrainingError(RuntimeError):
    pass


# Quality/speed presets -- draft for a fast preview, standard for normal use,
# high_def for the best result at the cost of a longer run. Only "epochs"
# differs between presets; every other setting below stays identical.
EPOCH_PRESETS = {
    "draft": 20,
    "standard": 60,
    "high_def": 120,
}
DEFAULT_EPOCH_PRESET = "standard"

_BASE_SETTINGS = {
    "batch_size": 16,
    "ny": 8192,
    "seed": 0,
    "latency": 0,
    "ignore_checks": False,
    "fast_dev_run": False,
    "silent": True,
}


def settings_for_preset(preset: str) -> dict:
    if preset not in EPOCH_PRESETS:
        raise CloudTrainingError(f"unknown A2 epoch preset {preset!r} -- choose one of {sorted(EPOCH_PRESETS)}")
    return {**_BASE_SETTINGS, "epochs": EPOCH_PRESETS[preset]}


# Kept for backward compatibility / parity checks against
# hybrid/a2_training_settings.py's A2_TRAINING_SETTINGS (both use the same
# DEFAULT_EPOCH_PRESET).
TRAINING_SETTINGS = settings_for_preset(DEFAULT_EPOCH_PRESET)
QUICK_SETTINGS = {**_BASE_SETTINGS, "epochs": 1, "fast_dev_run": True}


def _find_input_dir() -> Path:
    """Locate the attached dataset directory robustly -- Kaggle mounts
    dataset sources under /kaggle/input/<dataset-slug>/, but we don't
    hard-code the slug: search for the expected files instead."""
    search_roots = [Path("/kaggle/input"), Path(".")]
    for root in search_roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("input.wav"):
            if (candidate.parent / "hybrid_target.wav").is_file():
                return candidate.parent
    raise CloudTrainingError("could not locate input.wav/hybrid_target.wav under /kaggle/input or .")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def print_diagnostics() -> dict:
    info: dict = {"python_version": platform.python_version()}
    print(f"Python: {info['python_version']}")

    try:
        import nam
        info["nam_version"] = getattr(nam, "__version__", "unknown")
    except ImportError:
        info["nam_version"] = None
    print(f"neural-amp-modeler (pre-install check): {info['nam_version']}")

    import torch
    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    print(f"Torch: {info['torch_version']}  CUDA available: {info['cuda_available']}")
    if not info["cuda_available"]:
        raise CloudTrainingError("CUDA is not available in this Kaggle kernel -- refusing to train on CPU in the cloud path")

    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["cuda_runtime_version"] = getattr(torch.version, "cuda", None)
    print(f"GPU: {info['gpu_name']}  CUDA runtime: {info['cuda_runtime_version']}")
    return info


def ensure_nam_installed() -> str:
    try:
        import nam
        if getattr(nam, "__version__", None) == NEURAL_AMP_MODELER_VERSION:
            return nam.__version__
    except ImportError:
        pass
    print(f"Installing neural-amp-modeler=={NEURAL_AMP_MODELER_VERSION} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", f"neural-amp-modeler=={NEURAL_AMP_MODELER_VERSION}"],
        check=True,
    )
    import importlib
    import nam
    importlib.reload(nam)
    return getattr(nam, "__version__", NEURAL_AMP_MODELER_VERSION)


def validate_inputs(bundle_dir: Path) -> dict:
    import numpy as np
    import soundfile as sf

    input_path = bundle_dir / "input.wav"
    target_path = bundle_dir / "hybrid_target.wav"

    md5 = _md5_file(input_path)
    if md5 != OFFICIAL_V3_INPUT_MD5:
        raise CloudTrainingError(
            f"input.wav does not match official NAM V3 (expected md5 {OFFICIAL_V3_INPUT_MD5}, got {md5})"
        )

    input_audio, input_sr = sf.read(input_path, dtype="float32", always_2d=False)
    target_audio, target_sr = sf.read(target_path, dtype="float32", always_2d=False)

    if input_audio.ndim != 1 or target_audio.ndim != 1:
        raise CloudTrainingError("input/target must both be mono")
    if input_sr != REQUIRED_SAMPLE_RATE or target_sr != REQUIRED_SAMPLE_RATE:
        raise CloudTrainingError(f"input/target must be {REQUIRED_SAMPLE_RATE} Hz, got {input_sr}/{target_sr}")
    if len(input_audio) != len(target_audio):
        raise CloudTrainingError(f"frame count mismatch: input={len(input_audio)} target={len(target_audio)}")
    if not (np.all(np.isfinite(input_audio)) and np.all(np.isfinite(target_audio))):
        raise CloudTrainingError("input/target contain non-finite samples")
    if float(target_audio.max(initial=0.0)) >= 1.0 or float((-target_audio).max(initial=0.0)) >= 1.0:
        raise CloudTrainingError("hybrid_target.wav peaks at/above 0 dBFS")

    return {
        "input_sha256": _sha256_file(input_path),
        "target_sha256": _sha256_file(target_path),
        "input_md5": md5,
        "sample_rate": input_sr,
        "frame_count": len(input_audio),
    }


def _resolve_baked_cab_fir_samples(manifest: dict) -> int:
    """Resolve a baked cab's formal serial FIR-history sample count from
    whatever the manifest recorded -- this self-contained worker never
    receives the actual cab IR file (see docs/blend-mode.md "TRAINING /
    KAGGLE": source NAMs/cab IRs are not uploaded to Kaggle), so, unlike
    scripts/train_a2.py's local equivalent, it can only ever trust numbers
    already computed by hybrid.training_target.compute_receptive_field_record
    at generation time. Prefers the NEW nested receptive_field.cab record
    (computed at the OFFICIAL TRAINING INPUT's sample rate, i.e. accurate for
    what was actually baked into hybrid_target.wav) over the legacy flat
    receptive_field.cab_fir_serial_samples key, over the CabDesign's own
    audition-time (possibly different sample rate) fir_history_samples.
    """
    cab = manifest.get("cab") or {}
    if not cab.get("baked"):
        return 0
    rf_record = manifest.get("receptive_field") or {}
    rf_cab = rf_record.get("cab") or {}
    if rf_cab.get("fir_history_samples") is not None:
        return int(rf_cab["fir_history_samples"])
    legacy = rf_record.get("cab_fir_serial_samples")
    if legacy is not None:
        return int(legacy)
    return int(cab.get("fir_history_samples") or 0)


def check_receptive_field(manifest: dict, sample_rate: int) -> dict:
    """Mode-aware receptive-field policy, duplicated from
    scripts/train_a2.py's check_receptive_field (see this module's docstring
    for why this script duplicates rather than imports hybrid/ code) --
    MUST stay semantically identical to it (see
    tests/test_receptive_field_parity.py). Reads ONLY the manifest -- this
    script never receives the source .nam files or cab IR (see
    docs/blend-mode.md "TRAINING / KAGGLE"), so branch samples come from
    manifest["receptive_field"]["branch_samples"], computed locally at
    generation time by hybrid.training_target.compute_receptive_field_record.

    Two separate questions, exactly as in the local script:

    1. CORE (hard) dependency -- max(Amp A RF, Amp B RF[, envelope RF for
       Hybrid]) -- MUST fit inside the destination A2's receptive field
       (the packed A2 config of the neural-amp-modeler version actually
       installed in THIS kernel; `ensure_nam_installed()` must have run
       first). Failing this raises `CloudTrainingError` -- training aborts.
    2. A baked cabinet's FORMAL total (core + FIR history) is calculated and
       reported honestly, but NEVER gates training by itself -- exceeding
       the A2's receptive field means A2 will approximate the post-cab
       response within its available temporal capacity, not that training
       is invalid.
    """
    rf_record = manifest.get("receptive_field")
    if not rf_record:
        print("WARNING: manifest has no receptive_field record -- skipping receptive-field check.")
        return {}

    branch_samples = {k: v for k, v in (rf_record.get("branch_samples") or {}).items() if v is not None}
    if not branch_samples:
        print("WARNING: receptive_field.branch_samples is empty/unavailable -- skipping receptive-field check.")
        return {}

    hard_required = max(branch_samples.values())
    mode = manifest.get("mode", "hybrid")

    print(f"Core target dependency by branch (mode={mode}):")
    for label, samples in branch_samples.items():
        print(f"  {label:<12} {samples:>6} samples ({samples / sample_rate * 1000:6.1f} ms)")
    print(f"  {'hard core':<12} {hard_required:>6} samples ({hard_required / sample_rate * 1000:6.1f} ms)")

    import importlib.resources

    try:
        resource = importlib.resources.files("nam.train._resources").joinpath("config_model_packed.json")
        raw_config = json.loads(resource.read_text(encoding="utf-8"))
        submodels = raw_config["net"]["config"]["submodels"]
        best_samples = 0
        names = []
        for entry in submodels:
            names.append(entry["name"])
            layer_arrays = entry["config"].get("layers_configs", entry["config"].get("layers"))
            total = 1
            for layer_cfg in layer_arrays:
                kernel_sizes, dilations = layer_cfg["kernel_sizes"], layer_cfg["dilations"]
                total += sum((int(k) - 1) * int(d) for k, d in zip(kernel_sizes, dilations))
            best_samples = max(best_samples, total)
    except Exception as exc:  # noqa: BLE001 -- report, never guess at the config shape
        raise CloudTrainingError(f"could not determine installed A2's receptive field: {exc}") from exc

    # CORE dependency is the HARD gate -- unlike a baked cab below, this is
    # never relaxed to an advisory warning.
    if hard_required > best_samples:
        raise CloudTrainingError(
            f"REFUSING to train: core dependency {hard_required} samples exceeds the installed "
            f"A2's receptive field {best_samples} samples (submodels={names})."
        )

    margin = best_samples - hard_required
    core_status = "EXACT FIT" if margin == 0 else "OK"
    print("\nDestination A2 RF:")
    print(f"  {'available':<12} {best_samples:>6} samples ({best_samples / sample_rate * 1000:6.1f} ms)")
    if margin == 0:
        print(f"  {'core status':<12} {core_status} -- zero temporal margin. Training permitted.")
    else:
        print(f"  {'core status':<12} {core_status}  margin {margin / sample_rate * 1000:.1f} ms")

    result = {
        "mode": mode,
        "branch_samples": branch_samples,
        "hard_required_samples": hard_required,
        "a2_receptive_field_samples": best_samples,
        "a2_submodels": names,
        "core_status": core_status,
        "cab_baked": False,
        "cab_fir_history_samples": 0,
        "formal_total_required_samples": hard_required,
        "cab_requires_approximation": False,
    }

    cab = manifest.get("cab") or {}
    if not cab.get("baked"):
        return result

    cab_fir_samples = _resolve_baked_cab_fir_samples(manifest)
    if not cab_fir_samples:
        return result

    formal_total = hard_required + cab_fir_samples
    result.update({
        "cab_baked": True,
        "cab_fir_history_samples": cab_fir_samples,
        "formal_total_required_samples": formal_total,
    })

    print("\nBaked cabinet:")
    print(f"  {'FIR history':<12} {cab_fir_samples:>6} samples ({cab_fir_samples / sample_rate * 1000:6.1f} ms)")
    print(f"  {'formal total':<12} {formal_total:>6} samples ({formal_total / sample_rate * 1000:6.1f} ms)")

    # The formal total is NEVER a hard gate -- only report/record whether A2
    # is being asked to approximate the cab.
    if formal_total > best_samples:
        result["cab_requires_approximation"] = True
        print(
            "\nCABINET APPROXIMATION:\n"
            f"  The core {mode.capitalize()} target fits within the A2 receptive field "
            f"({hard_required} / {best_samples} samples).\n"
            f"  The baked cabinet extends the teacher's formal temporal dependency to "
            f"{formal_total} samples, beyond the A2 receptive field of {best_samples} samples.\n"
            "  Training will continue: the cabinet response will be approximated by the A2 within its "
            "available temporal capacity.\n"
            "  Validate the resulting model against the baked target and by listening."
        )
    else:
        print(
            f"  Formal total also fits inside the A2 receptive field "
            f"({formal_total} <= {best_samples}) -- no approximation needed for the cab."
        )

    return result


def user_metadata_kwargs(manifest: dict) -> dict:
    """Identical logic to hybrid/a2_training_settings.py's
    user_metadata_kwargs -- duplicated here per this module's
    self-containment rule (see module docstring); parity is asserted in
    tests/test_a2_training_settings.py."""
    amp_a_name = Path(manifest.get("amp_a", {}).get("filename", "Amp A")).stem
    amp_b_name = Path(manifest.get("amp_b", {}).get("filename", "Amp B")).stem
    calibration = manifest.get("calibration", {})
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

    model_name = str(manifest.get("model_name") or "").strip() or name
    return {
        "name": model_name,
        "modeled_by": "Hybrid NAM Builder",
        "gear_make": "Hybrid" if mode == "hybrid" else "Character Blend" if mode == "character" else "Blend",
        "gear_model": gear_model,
        "input_level_dbu": input_level_dbu,
    }


def run_training(bundle_dir: Path, output_dir: Path, quick: bool, epoch_preset: str = DEFAULT_EPOCH_PRESET) -> dict:
    # Validated before importing torch/nam -- an unknown preset should fail
    # immediately, not after paying for a heavy import first.
    settings = QUICK_SETTINGS if quick else settings_for_preset(epoch_preset)

    import nam.train.core as core
    from nam.models.metadata import GearType, UserMetadata
    from nam.train.metadata import TRAINING_KEY

    with open(bundle_dir / "training_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rf_check = check_receptive_field(manifest, REQUIRED_SAMPLE_RATE)

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = core.train(
        input_path=str(bundle_dir / "input.wav"),
        output_path=str(bundle_dir / "hybrid_target.wav"),
        train_path=str(output_dir),
        epochs=settings["epochs"],
        latency=settings["latency"],
        batch_size=settings["batch_size"],
        ny=settings["ny"],
        seed=settings["seed"],
        ignore_checks=settings["ignore_checks"],
        silent=settings["silent"],
        modelname="model",
        fast_dev_run=settings["fast_dev_run"],
    )
    duration_s = time.time() - start

    if result is None or result.model is None:
        raise CloudTrainingError("nam.train.core.train() returned no model -- official data checks likely failed")

    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    # docs/blend-mode.md "METADATA / OUTPUT NAM": use an official amp+cab/rig
    # gear type when baking a cab, IF the installed package actually has one
    # -- mirrors scripts/train_a2.py's _build_user_metadata, duplicated here
    # per this module's self-containment rule (see module docstring).
    gear_type = GearType.AMP
    if (manifest.get("cab") or {}).get("baked"):
        for candidate_name in ("AMP_CAB", "RIG", "PREAMP_CAB", "AMP_AND_CAB"):
            candidate = getattr(GearType, candidate_name, None)
            if candidate is not None:
                gear_type = candidate
                break
    user_metadata = UserMetadata(gear_type=gear_type, **user_metadata_kwargs(manifest))
    artifact_stem = str(manifest.get("artifact_stem") or "hybrid_a2")
    result.model.net.export(
        export_dir,
        basename=artifact_stem,
        user_metadata=user_metadata,
        other_metadata={TRAINING_KEY: result.metadata.model_dump()},
    )

    nam_path = export_dir / f"{artifact_stem}.nam"
    if not nam_path.is_file():
        raise CloudTrainingError(f"export did not produce expected file: {nam_path}")

    if rf_check.get("cab_requires_approximation"):
        print(
            "\nBaked cab was trained as an approximation because its formal temporal "
            "dependency exceeded A2 RF. Review validation metrics and listening result."
        )

    return {
        "nam_path": nam_path,
        "duration_seconds": duration_s,
        "settings": settings,
        "receptive_field_check": rf_check,
    }


def main() -> int:
    result: dict = {"success": False, "timestamps": {"started": time.time()}}
    try:
        bundle_dir = _find_input_dir()
        quick = False
        epoch_preset = DEFAULT_EPOCH_PRESET
        cloud_job_path = bundle_dir / "cloud_job.json"
        if cloud_job_path.is_file():
            with open(cloud_job_path, "r", encoding="utf-8") as f:
                cloud_job = json.load(f)
            requested_preset = cloud_job.get("epoch_preset")
            if requested_preset in EPOCH_PRESETS:
                epoch_preset = requested_preset
            elif requested_preset is not None:
                print(f"WARNING: cloud_job.json requested unknown epoch_preset {requested_preset!r} -- "
                      f"falling back to {DEFAULT_EPOCH_PRESET!r}")

        env_info = print_diagnostics()
        result["environment"] = env_info

        nam_version = ensure_nam_installed()
        result["neural_amp_modeler_version"] = nam_version
        if nam_version != NEURAL_AMP_MODELER_VERSION:
            print(f"WARNING: installed neural-amp-modeler {nam_version} != pinned {NEURAL_AMP_MODELER_VERSION}")

        input_info = validate_inputs(bundle_dir)
        result["input"] = input_info

        output_dir = Path("/kaggle/working/a2_output")
        train_result = run_training(bundle_dir, output_dir, quick=quick, epoch_preset=epoch_preset)

        nam_path = train_result["nam_path"]
        final_nam_path = Path("/kaggle/working") / nam_path.name
        if nam_path != final_nam_path:
            final_nam_path.write_bytes(nam_path.read_bytes())

        result.update({
            "success": True,
            "torch_version": env_info.get("torch_version"),
            "cuda_version": env_info.get("cuda_runtime_version"),
            "gpu_name": env_info.get("gpu_name"),
            "epoch_preset": epoch_preset if not quick else None,
            "epochs": train_result["settings"]["epochs"],
            "batch_size": train_result["settings"]["batch_size"],
            "ny": train_result["settings"]["ny"],
            "seed": train_result["settings"]["seed"],
            "duration_seconds": train_result["duration_seconds"],
            "output_nam_filename": final_nam_path.name,
            "output_nam_sha256": _sha256_file(final_nam_path),
            "receptive_field_check": train_result.get("receptive_field_check"),
        })
    except Exception as exc:  # noqa: BLE001 -- report every failure, never crash silently past this point
        result["success"] = False
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
    finally:
        result["timestamps"]["finished"] = time.time()
        with open("/kaggle/working/training_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
