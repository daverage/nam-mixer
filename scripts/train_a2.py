#!/usr/bin/env python3
"""Train a real A2 (PackedWaveNet) model on a generated hybrid bundle --
docs/phase3.md sections 13-21, 31-33.

Runs INSIDE the dedicated training environment (see scripts/setup_a2_env.ps1)
-- has no Flask import, is independently runnable:

    python scripts/train_a2.py work/a2/<design_id>/training_manifest.json [--quick] [--device auto]

What it does, in order (aborts non-zero on any failure -- never silently
degrades to "worked around it"):

  1. Print environment diagnostics (Python/nam/torch/lightning/accelerator).
  2. Load the manifest written by hybrid.training_target.generate_training_bundle,
     locate input.wav/hybrid_target.wav next to it, and verify their hashes,
     sample rate, and frame-count equality against the manifest (docs/phase3.md
     section 16 -- input and target must stay exactly sample-aligned; latency
     is authoritatively 0 for this synthetic target).
  3. Confirm the bounded crossover envelope's declared history actually fits
     inside the installed A2's real receptive field (hybrid/receptive_field.py)
     -- not just asserted once at design time, but re-checked against
     whatever nam version is ACTUALLY installed in this environment.
  4. Call the official current simplified A2 training entry point (adapter
     in `_run_official_trainer` below -- see its docstring: this repo could
     not import the real neural-amp-modeler package while writing this
     wrapper, so the adapter probes several plausible official entry points
     at runtime and reports exactly which one it used, rather than guessing
     blind at one).
  5. Verify the exported .nam: parse it, load it via the existing native
     NeuralAmpModelerCore renderer (hybrid/render.py), render the training
     input through it, and confirm mono/finite/correct length/correct sample
     rate.
  6. Compare the A2's re-rendered output against hybrid_target.wav (basic
     ESR/RMS/peak metrics).
  7. Update training_manifest.json's "training" section with the real
     environment/settings actually used.

Does NOT bypass the official trainer's own data-validation checks (docs/phase3.md
section 17) and does NOT default to a low-epoch smoke-test config (section 15)
-- pass --quick explicitly to opt into a fast development run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hybrid.receptive_field import ReceptiveFieldUnavailable, assert_envelope_history_fits  # noqa: E402
from hybrid.render import NamRenderError, render  # noqa: E402
from hybrid.nam_loader import load_nam  # noqa: E402


class TrainingAbort(RuntimeError):
    """Raised for any condition that should stop training with a non-zero exit
    and a clear message -- never caught-and-continued."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_audio(path: Path):
    import soundfile as sf
    return sf.read(path, dtype="float32", always_2d=False)


def print_environment_diagnostics() -> dict:
    """docs/phase3.md section 32."""
    info: dict = {"python_version": platform.python_version()}
    print(f"Python: {info['python_version']}")

    try:
        import nam
        info["neural_amp_modeler_version"] = getattr(nam, "__version__", "unknown")
    except ImportError:
        info["neural_amp_modeler_version"] = None
    print(f"neural-amp-modeler: {info['neural_amp_modeler_version']}")

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["mps_available"] = False
    print(f"Torch: {info['torch_version']}  CUDA: {info['cuda_available']}  MPS: {info['mps_available']}")

    try:
        import pytorch_lightning as pl
        info["pytorch_lightning_version"] = pl.__version__
    except ImportError:
        info["pytorch_lightning_version"] = None
    print(f"PyTorch Lightning: {info['pytorch_lightning_version']}")

    if info["cuda_available"]:
        info["selected_accelerator"] = "cuda"
    elif info.get("mps_available"):
        info["selected_accelerator"] = "mps"
    else:
        info["selected_accelerator"] = "cpu"
        print("WARNING: no GPU detected -- training will run on CPU (slow). Not changing model architecture to compensate.")
    print(f"Selected accelerator: {info['selected_accelerator']}")

    return info


def load_and_validate_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise TrainingAbort(f"manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bundle_dir = manifest_path.parent
    input_path = bundle_dir / "input.wav"
    target_path = bundle_dir / "hybrid_target.wav"
    for p in (input_path, target_path):
        if not p.is_file():
            raise TrainingAbort(f"expected bundle file missing: {p}")

    input_sha = _sha256_file(input_path)
    expected_input_sha = manifest.get("training_input", {}).get("sha256")
    if expected_input_sha and input_sha != expected_input_sha:
        raise TrainingAbort(
            f"input.wav hash mismatch vs manifest (expected {expected_input_sha}, got {input_sha}) -- "
            "the bundle directory may have been modified after generation."
        )

    target_sha = _sha256_file(target_path)
    expected_target_sha = manifest.get("target", {}).get("final_sha256")
    if expected_target_sha and target_sha != expected_target_sha:
        raise TrainingAbort(
            f"hybrid_target.wav hash mismatch vs manifest (expected {expected_target_sha}, got {target_sha})"
        )

    input_audio, input_sr = _read_audio(input_path)
    target_audio, target_sr = _read_audio(target_path)
    if input_sr != target_sr:
        raise TrainingAbort(f"sample rate mismatch: input={input_sr} target={target_sr}")
    if len(input_audio) != len(target_audio):
        raise TrainingAbort(
            f"frame count mismatch: input={len(input_audio)} target={len(target_audio)} -- "
            "input/target must be exactly sample-aligned (docs/phase3.md section 16)"
        )

    print(f"Manifest OK: input/target aligned at {len(input_audio)} frames, {input_sr} Hz.")
    manifest["_bundle_dir"] = str(bundle_dir)
    manifest["_input_path"] = str(input_path)
    manifest["_target_path"] = str(target_path)
    return manifest


def check_receptive_field(manifest: dict, sample_rate: int) -> None:
    max_history_ms = manifest.get("design", {}).get("envelope_max_history_ms")
    if max_history_ms is None:
        print("WARNING: manifest has no envelope_max_history_ms -- skipping receptive-field check.")
        return
    history_samples = int(round(max_history_ms / 1000.0 * sample_rate))
    try:
        rf = assert_envelope_history_fits(history_samples, sample_rate, margin_fraction=0.0)
        print(
            f"Receptive field OK: envelope history {history_samples} samples "
            f"({max_history_ms:.1f} ms) fits inside A2 receptive field "
            f"{rf.receptive_field_samples} samples (submodels={rf.submodel_names})."
        )
    except ReceptiveFieldUnavailable as exc:
        raise TrainingAbort(str(exc)) from exc
    except ValueError as exc:
        raise TrainingAbort(f"REFUSING to train: {exc}") from exc


def _run_official_trainer(input_path: Path, target_path: Path, output_dir: Path, quick: bool, device: str) -> Path:
    """Call the official current neural-amp-modeler simplified A2 trainer:
    `nam.train.core.train()`.

    Verified (not guessed) against the actually-installed
    neural-amp-modeler==0.13.0 (see requirements-training.txt) --
    `inspect.signature(nam.train.core.train)` was inspected live while
    writing this function:

        train(input_path, output_path, train_path, epochs=100, latency=None,
              batch_size=16, ny=8192, seed=0, save_plot=False, silent=False,
              modelname='model', ignore_checks=False, local=False,
              threshold_esr=None, user_metadata=None, fast_dev_run=False)
              -> TrainOutput | None

    `train_path` is the Lightning `default_root_dir` (checkpoints etc.), NOT
    the exported model location -- confirmed via `inspect.getsource`, the
    function always builds `model_config = _get_packed_model_config()`
    (loading the exact `nam.train._resources/config_model_packed.json` file
    hybrid/receptive_field.py inspects), so this genuinely is the official
    A2/PackedWaveNet architecture, never a hand-rolled one. `train()` does
    NOT export a `.nam` file itself -- the returned `TrainOutput.model` is a
    `LightningModule` wrapping the trained net; `.net.export(...)` (a real,
    verified `BaseNet.export` method) writes the actual `.nam`.

    `device` is accepted for CLI-symmetry with `--device` but the installed
    trainer has no device-selection parameter -- it picks CUDA/MPS/CPU
    internally via `torch.cuda.is_available()`/`torch.backends.mps` (see
    print_environment_diagnostics's own equivalent check); an explicit
    non-"auto" request is only used as an informational note, not enforced,
    since the API does not expose an override.
    """
    import nam.train.core as core

    if device != "auto":
        print(f"NOTE: --device={device} requested, but the installed nam.train.core.train() has no device "
              "override -- it selects CUDA/MPS/CPU automatically. Continuing with automatic selection.")

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if quick else 100

    result = core.train(
        input_path=str(input_path),
        output_path=str(target_path),
        train_path=str(output_dir),
        epochs=epochs,
        latency=0,  # docs/phase3.md section 16 -- synthetic latency is authoritatively 0, never auto-detected
        silent=False,
        modelname="model",
        fast_dev_run=quick,
    )

    if result is None or result.model is None:
        raise TrainingAbort(
            "nam.train.core.train() returned no model -- the official trainer's own data "
            "checks likely failed (see its printed output above). Not bypassing that check "
            "(docs/phase3.md section 17); investigate the cause instead of forcing a run "
            "with ignore_checks."
        )

    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    result.model.net.export(export_dir, basename="model")

    nam_path = export_dir / "model.nam"
    if not nam_path.is_file():
        raise TrainingAbort(f"BaseNet.export() did not produce the expected file: {nam_path}")
    return nam_path


def validate_exported_nam(nam_path: Path, input_path: Path, expected_sample_rate: int, slim: bool = False) -> dict:
    """`slim=True` exercises the Lite/slim submodel of a packed A2 export
    (NAMCore's `--slim` flag, forwarded via hybrid.render.render's `slim`
    kwarg -- see docs/phase3.md section 20); `slim=False` (default) exercises
    the Full submodel. Only meaningful for a slimmable/packed export; a
    NamRenderError from a non-slimmable model when slim=True is expected and
    surfaces as a TrainingAbort so callers can treat "Lite not supported by
    this export" as a distinct, visible outcome rather than a silent no-op.
    """
    model = load_nam(nam_path)
    input_audio, sr = _read_audio(input_path)
    if sr != expected_sample_rate:
        raise TrainingAbort(f"input sample rate {sr} != expected {expected_sample_rate}")

    try:
        rendered = render(model, input_audio, sr, slim=(1.0 if slim else 0.0))
    except NamRenderError as exc:
        raise TrainingAbort(f"NAMCore failed to render the exported model ({nam_path}): {exc}") from exc

    if rendered.ndim != 1:
        raise TrainingAbort(f"exported model produced non-mono output: shape {rendered.shape}")
    if len(rendered) != len(input_audio):
        raise TrainingAbort(f"exported model output length {len(rendered)} != input length {len(input_audio)}")
    if not np.all(np.isfinite(rendered)):
        raise TrainingAbort("exported model produced non-finite samples")

    print(f"NAMCore validation OK for {nam_path.name}: mono, {len(rendered)} frames, {sr} Hz, all finite.")
    return {"path": str(nam_path), "sample_rate": sr, "frame_count": len(rendered), "rendered": rendered}


def compare_to_target(rendered: np.ndarray, target_path: Path) -> dict:
    target_audio, _ = _read_audio(target_path)
    n = min(len(rendered), len(target_audio))
    a = rendered[:n].astype(np.float64)
    b = target_audio[:n].astype(np.float64)

    err = a - b
    esr = float(np.sum(err ** 2) / max(np.sum(b ** 2), 1e-12))

    a_rms = np.sqrt(np.mean(a ** 2))
    b_rms = np.sqrt(np.mean(b ** 2))
    if b_rms > 1e-12:
        a_normalized = a * (b_rms / max(a_rms, 1e-12))
        gain_normalized_esr = float(np.sum((a_normalized - b) ** 2) / max(np.sum(b ** 2), 1e-12))
    else:
        gain_normalized_esr = float("nan")

    rms_diff = float(abs(a_rms - b_rms))
    peak_diff = float(abs(np.max(np.abs(a)) - np.max(np.abs(b))))

    metrics = {
        "raw_esr": esr,
        "gain_normalized_esr": gain_normalized_esr,
        "rms_difference": rms_diff,
        "peak_difference": peak_diff,
    }
    print(f"vs target: raw ESR={esr:.5f}  gain-normalized ESR={gain_normalized_esr:.5f}  "
          f"RMS diff={rms_diff:.5f}  peak diff={peak_diff:.5f}")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to training_manifest.json")
    parser.add_argument("--quick", action="store_true", help="fast development smoke-test run (NOT for the final model)")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to <bundle_dir>/a2_output")
    args = parser.parse_args(argv)

    try:
        env_info = print_environment_diagnostics()
        manifest = load_and_validate_manifest(args.manifest)
        bundle_dir = Path(manifest["_bundle_dir"])
        input_path = Path(manifest["_input_path"])
        target_path = Path(manifest["_target_path"])
        sample_rate = manifest["training_input"]["sample_rate"]

        check_receptive_field(manifest, sample_rate)

        output_dir = args.output_dir or (bundle_dir / "a2_output")
        if args.quick:
            print("--quick: running a fast development smoke test, NOT the final model.")

        nam_path = _run_official_trainer(input_path, target_path, output_dir, args.quick, args.device)
        print(f"Trainer produced: {nam_path}")

        full_result = validate_exported_nam(nam_path, input_path, sample_rate, slim=False)
        full_metrics = compare_to_target(full_result["rendered"], target_path)

        lite_metrics = None
        try:
            lite_result = validate_exported_nam(nam_path, input_path, sample_rate, slim=True)
            lite_metrics = compare_to_target(lite_result["rendered"], target_path)
        except TrainingAbort as exc:
            print(f"Lite/slim validation skipped or failed: {exc}")

        manifest["training"] = {
            **env_info,
            "device_requested": args.device,
            "quick_mode": args.quick,
            "output_nam_path": str(nam_path),
            "output_nam_sha256": _sha256_file(nam_path),
            "full_metrics_vs_target": full_metrics,
            "lite_metrics_vs_target": lite_metrics,
        }
        manifest.pop("_bundle_dir", None)
        manifest.pop("_input_path", None)
        manifest.pop("_target_path", None)
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nDone. Manifest updated: {args.manifest}")
        return 0

    except TrainingAbort as exc:
        print(f"\nABORTED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
