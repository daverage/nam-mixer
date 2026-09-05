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

TRAINING_SETTINGS = {
    "epochs": 100,
    "batch_size": 16,
    "ny": 8192,
    "seed": 0,
    "latency": 0,
    "ignore_checks": False,
    "fast_dev_run": False,
    "silent": True,
}
QUICK_SETTINGS = {**TRAINING_SETTINGS, "epochs": 1, "fast_dev_run": True}

REQUIRED_SAMPLE_RATE = 48000


class CloudTrainingError(RuntimeError):
    pass


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


def user_metadata_kwargs(manifest: dict) -> dict:
    """Identical logic to hybrid/a2_training_settings.py's
    user_metadata_kwargs -- duplicated here per this module's
    self-containment rule (see module docstring); parity is asserted in
    tests/test_a2_training_settings.py."""
    amp_a_name = Path(manifest.get("amp_a", {}).get("filename", "Amp A")).stem
    amp_b_name = Path(manifest.get("amp_b", {}).get("filename", "Amp B")).stem
    calibration = manifest.get("calibration", {})
    input_level_dbu = calibration.get("reference_input_level_dbu") if calibration.get("applied") else None
    return {
        "name": f"Hybrid {amp_a_name} -> {amp_b_name}",
        "modeled_by": "Hybrid NAM Builder",
        "gear_make": "Hybrid",
        "gear_model": f"{amp_a_name} -> {amp_b_name}",
        "input_level_dbu": input_level_dbu,
    }


def run_training(bundle_dir: Path, output_dir: Path, quick: bool) -> dict:
    import nam.train.core as core
    from nam.models.metadata import GearType, UserMetadata
    from nam.train.metadata import TRAINING_KEY

    with open(bundle_dir / "training_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    settings = QUICK_SETTINGS if quick else TRAINING_SETTINGS
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
    user_metadata = UserMetadata(gear_type=GearType.AMP, **user_metadata_kwargs(manifest))
    result.model.net.export(
        export_dir,
        basename="hybrid_a2",
        user_metadata=user_metadata,
        other_metadata={TRAINING_KEY: result.metadata.model_dump()},
    )

    nam_path = export_dir / "hybrid_a2.nam"
    if not nam_path.is_file():
        raise CloudTrainingError(f"export did not produce expected file: {nam_path}")

    return {
        "nam_path": nam_path,
        "duration_seconds": duration_s,
        "settings": settings,
    }


def main() -> int:
    result: dict = {"success": False, "timestamps": {"started": time.time()}}
    try:
        bundle_dir = _find_input_dir()
        quick = False
        cloud_job_path = bundle_dir / "cloud_job.json"
        if cloud_job_path.is_file():
            with open(cloud_job_path, "r", encoding="utf-8") as f:
                json.load(f)  # currently informational only (job_id/design_id/accelerator)

        env_info = print_diagnostics()
        result["environment"] = env_info

        nam_version = ensure_nam_installed()
        result["neural_amp_modeler_version"] = nam_version
        if nam_version != NEURAL_AMP_MODELER_VERSION:
            print(f"WARNING: installed neural-amp-modeler {nam_version} != pinned {NEURAL_AMP_MODELER_VERSION}")

        input_info = validate_inputs(bundle_dir)
        result["input"] = input_info

        output_dir = Path("/kaggle/working/a2_output")
        train_result = run_training(bundle_dir, output_dir, quick=quick)

        nam_path = train_result["nam_path"]
        final_nam_path = Path("/kaggle/working") / nam_path.name
        if nam_path != final_nam_path:
            final_nam_path.write_bytes(nam_path.read_bytes())

        result.update({
            "success": True,
            "torch_version": env_info.get("torch_version"),
            "cuda_version": env_info.get("cuda_runtime_version"),
            "gpu_name": env_info.get("gpu_name"),
            "epochs": train_result["settings"]["epochs"],
            "batch_size": train_result["settings"]["batch_size"],
            "ny": train_result["settings"]["ny"],
            "seed": train_result["settings"]["seed"],
            "duration_seconds": train_result["duration_seconds"],
            "output_nam_filename": final_nam_path.name,
            "output_nam_sha256": _sha256_file(final_nam_path),
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
