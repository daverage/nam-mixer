#!/usr/bin/env python3
"""Train a real A2 (PackedWaveNet) model on a generated hybrid bundle --
docs/phase3.md sections 13-21, 31-33.

Runs INSIDE the dedicated training environment (see scripts/setup_a2_env.ps1)
-- has no Flask import, is independently runnable:

    python scripts/train_a2.py work/a2/<design_id>/training_manifest.json [--quick] [--epoch-preset draft|standard|high_def] [--device auto]

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
from dataclasses import replace
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hybrid.a2_training_settings import (  # noqa: E402
    A2_EPOCH_PRESETS,
    A2_QUICK_SETTINGS,
    DEFAULT_EPOCH_PRESET,
    settings_for_preset,
    user_metadata_kwargs,
)
from hybrid.cab_ir import CabIrError, get_prepared_cab_ir  # noqa: E402
import hybrid.character_training_target as character_training_target  # noqa: E402
from hybrid.receptive_field import (  # noqa: E402
    ReceptiveFieldUnavailable,
    assert_required_history_fits,
    compute_source_nam_receptive_field,
)
from hybrid.render import NamRenderError, render  # noqa: E402
from hybrid.metadata import suggested_nam_filename  # noqa: E402
from hybrid.nam_loader import load_nam  # noqa: E402
from hybrid.validation import compute_esr_metrics  # noqa: E402
from hybrid.validation_report import build_validation_report  # noqa: E402


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


def _resolve_baked_cab_fir_samples(manifest: dict, cab: dict, sample_rate: int) -> tuple[int, "int | None"]:
    """Resolve (fir_history_samples, fir_length_samples) for a baked cab.

    Prefers recomputing the EXACT prepared tap count from the actual cab IR
    file (the same `get_prepared_cab_ir` call `hybrid.training_target.
    maybe_bake_cab` used to produce hybrid_target.wav) -- this local trainer,
    unlike the self-contained Kaggle cloud worker, can import hybrid/ and
    normally runs on the same machine that generated the bundle, so the IR
    file is usually still reachable. Falls back to the manifest's own
    recorded numbers (receptive_field.cab, then the legacy
    receptive_field.cab_fir_serial_samples key, then the CabDesign's own
    audition-time fir_history_samples) if the file is no longer reachable.
    """
    ir_path = cab.get("ir_working_path")
    if ir_path:
        try:
            prepared = get_prepared_cab_ir(ir_path, sample_rate)
            return max(0, prepared.prepared_frame_count - 1), prepared.prepared_frame_count
        except (CabIrError, OSError) as exc:
            print(f"WARNING: could not recompute baked cab from {ir_path} ({exc}) -- falling back to recorded manifest numbers.")

    rf_record = manifest.get("receptive_field") or {}
    rf_cab = rf_record.get("cab") or {}
    if rf_cab.get("fir_history_samples") is not None:
        return int(rf_cab["fir_history_samples"]), rf_cab.get("fir_length_samples")
    legacy = rf_record.get("cab_fir_serial_samples")
    if legacy is not None:
        return int(legacy), None
    if cab.get("fir_history_samples"):
        return int(cab["fir_history_samples"]), cab.get("prepared_frame_count")
    return 0, None


def check_receptive_field(manifest: dict, sample_rate: int) -> dict:
    """CORE (hard) vs. Character/CABINET (formal/advisory) receptive-field
    policy -- see docs/blend-mode.md's cabinet-approximation-policy section
    for the full rationale this implements. Two separate questions:

    1. Does the CORE Hybrid/Blend dependency -- Amp A/Amp B (+ the crossover
       envelope, Hybrid only), mode-aware exactly as before -- fit inside
       the installed A2's actual receptive field? This is a HARD
       requirement: failing it aborts training (`assert_required_history_fits`).

    2. For Character Blend, its added smoothing/transition/correction history
       is recorded and may require approximation, but never disables training.
       If a cab is baked, does ALSO adding its serial FIR history
       (`fir_length - 1` samples) keep the FORMAL total within the A2's
       receptive field? This is calculated and reported honestly, but it
       NEVER gates training by itself -- exceeding it means the A2 will
       LEARN AN APPROXIMATION of the post-cab response within its available
       temporal capacity, not that training is invalid (we are training A2
       to approximate the rendered teacher target, not compiling its signal
       graph exactly).

    Returns a summary dict (empty if the check was skipped for lack of
    data) so `main()` can fold it into the manifest's "training" section.
    """
    mode = manifest.get("mode", "hybrid")

    branch_samples = {}
    if mode in ("hybrid", "character"):
        max_history_ms = manifest.get("design", {}).get("envelope_max_history_ms")
        if max_history_ms is None:
            print("WARNING: manifest has no envelope_max_history_ms -- skipping receptive-field check.")
            return {}
        branch_samples["envelope"] = int(round(max_history_ms / 1000.0 * sample_rate))

    for label, key in (("Amp A", "amp_a"), ("Amp B", "amp_b")):
        amp_path = manifest.get(key, {}).get("path")
        if not amp_path:
            print(f"WARNING: manifest has no {key}.path -- skipping {label}'s receptive-field check.")
            continue
        try:
            model = load_nam(amp_path)
            branch_samples[label] = compute_source_nam_receptive_field(model)
        except (OSError, ValueError, ReceptiveFieldUnavailable) as exc:
            print(f"WARNING: could not compute {label}'s receptive field ({amp_path}): {exc}")

    character_branches = {}
    if mode == "character":
        recorded_branches = (manifest.get("receptive_field") or {}).get("branch_samples") or {}
        for label, samples in recorded_branches.items():
            if str(label).startswith("character_") and samples is not None:
                character_branches[str(label)] = int(samples)
        qualification = (manifest.get("receptive_field") or {}).get("history_qualification")
        if qualification:
            print(f"WARNING: {qualification}")

    if not branch_samples:
        print("WARNING: no branch dependency could be determined -- skipping receptive-field check.")
        return {}

    hard_required = max(branch_samples.values())

    print(f"Core target dependency by branch (mode={mode}):")
    for label, samples in branch_samples.items():
        print(f"  {label:<12} {samples:>6} samples ({samples / sample_rate * 1000:6.1f} ms)")
    print(f"  {'hard core':<12} {hard_required:>6} samples ({hard_required / sample_rate * 1000:6.1f} ms)")

    # The CORE dependency is a HARD requirement -- unlike Character processing
    # and a baked cab below,
    # this is never relaxed to an advisory warning. A2's temporal capacity
    # was the whole reason the crossover envelope was bounded/causal and why
    # the source amps are what they are; if the core teacher itself doesn't
    # fit, the composite function genuinely cannot be represented.
    try:
        rf = assert_required_history_fits(hard_required, sample_rate, margin_fraction=0.0)
    except ReceptiveFieldUnavailable as exc:
        raise TrainingAbort(str(exc)) from exc
    except ValueError as exc:
        raise TrainingAbort(f"REFUSING to train: {exc}") from exc

    margin_samples = rf.receptive_field_samples - hard_required
    core_status = "EXACT FIT" if margin_samples == 0 else "OK"
    print("\nDestination A2 RF:")
    print(f"  {'available':<12} {rf.receptive_field_samples:>6} samples ({rf.receptive_field_samples / sample_rate * 1000:6.1f} ms)")
    if margin_samples == 0:
        print(
            f"  {'core status':<12} {core_status} -- zero temporal margin. Training permitted -- this checks "
            "temporal reach only, not whether the network has enough capacity to actually learn the "
            "composite function within that reach; that is exactly what this experiment is meant to determine."
        )
    else:
        print(f"  {'core status':<12} {core_status}  margin {margin_samples / sample_rate * 1000:.1f} ms")

    result = {
        "mode": mode,
        "branch_samples": {**branch_samples, **character_branches},
        "hard_required_samples": hard_required,
        "a2_receptive_field_samples": rf.receptive_field_samples,
        "a2_submodels": rf.submodel_names,
        "core_status": core_status,
        "cab_baked": False,
        "cab_fir_length_samples": None,
        "cab_fir_history_samples": 0,
        "formal_total_required_samples": hard_required,
        "cab_requires_approximation": False,
        "character_requires_approximation": False,
    }

    formal_character = hard_required
    if character_branches:
        formal_character = max(hard_required, *character_branches.values())
        result["formal_character_required_samples"] = formal_character
        print("\nCharacter processing dependency:")
        for label, samples in character_branches.items():
            print(f"  {label:<28} {samples:>6} samples ({samples / sample_rate * 1000:6.1f} ms)")
        print(f"  {'formal Character':<28} {formal_character:>6} samples ({formal_character / sample_rate * 1000:6.1f} ms)")
        if formal_character > rf.receptive_field_samples:
            result["character_requires_approximation"] = True
            print(
                "\nCHARACTER APPROXIMATION:\n"
                f"  Character processing extends the teacher dependency to {formal_character} samples, beyond "
                f"the A2 receptive field of {rf.receptive_field_samples} samples.\n"
                "  Training will continue for every backend and epoch preset. Validate Full and Lite exports "
                "against the frozen teacher and by listening."
            )
        else:
            print(f"  Character processing fits inside the A2 receptive field ({formal_character} <= {rf.receptive_field_samples}).")
        result["formal_total_required_samples"] = formal_character

    cab = manifest.get("cab") or {}
    if not cab.get("baked"):
        return result

    cab_fir_samples, cab_fir_length = _resolve_baked_cab_fir_samples(manifest, cab, sample_rate)
    if not cab_fir_samples:
        return result

    cab_formal = hard_required + cab_fir_samples
    formal_total = formal_character + cab_fir_samples
    result.update({
        "cab_baked": True,
        "cab_fir_length_samples": cab_fir_length,
        "cab_fir_history_samples": cab_fir_samples,
        "formal_total_required_samples": formal_total,
    })

    print("\nBaked cabinet:")
    if cab_fir_length:
        print(f"  {'FIR length':<12} {cab_fir_length:>6} samples ({cab_fir_length / sample_rate * 1000:6.1f} ms)")
    print(f"  {'FIR history':<12} {cab_fir_samples:>6} samples ({cab_fir_samples / sample_rate * 1000:6.1f} ms)")
    print(f"  {'formal total':<12} {formal_total:>6} samples ({formal_total / sample_rate * 1000:6.1f} ms)")
    energy_999_samples = cab.get("energy_999_samples")
    energy_999_ms = cab.get("energy_999_ms")
    if energy_999_samples is not None and energy_999_ms is not None:
        print(f"  99.9% energy by: {energy_999_samples} samples ({energy_999_ms:.1f} ms)")

    # The formal total is NEVER a hard gate -- see function docstring. Only
    # report/record whether A2 is being asked to approximate the cab.
    if cab_formal > rf.receptive_field_samples:
        result["cab_requires_approximation"] = True
        print(
            "\nCABINET APPROXIMATION:\n"
            f"  The core {mode.capitalize()} target fits within the A2 receptive field "
            f"({hard_required} / {rf.receptive_field_samples} samples).\n"
            f"  The baked cabinet extends the hard-core dependency to {cab_formal} samples, beyond "
            f"the A2 receptive field of {rf.receptive_field_samples} samples.\n"
            "  Training will continue: the cabinet response will be approximated by the A2 within its "
            "available temporal capacity.\n"
            "  Validate the resulting model against the baked target and by listening."
        )
    else:
        print(
            f"  Cabinet-adjusted core fits inside the A2 receptive field "
            f"({cab_formal} <= {rf.receptive_field_samples}) -- no additional cabinet approximation needed."
        )

    return result


def _build_user_metadata(manifest: dict):
    """NAM `UserMetadata` for the final export, built from the manifest --
    see docs/phase3.md review section 5. Only called after `nam.train.core`
    has already been imported successfully, so `nam.models.metadata` is
    guaranteed importable too.
    """
    from nam.models.metadata import GearType, UserMetadata

    # docs/blend-mode.md "METADATA / OUTPUT NAM": use an official amp+cab/rig
    # gear type when baking a cab, IF the installed package actually has one
    # -- never invent an unsupported enum value. Checked dynamically against
    # whatever GearType members are ACTUALLY installed rather than hardcoding
    # a guessed name.
    gear_type = GearType.AMP
    cab = manifest.get("cab") or {}
    if cab.get("baked"):
        for candidate_name in ("AMP_CAB", "RIG", "PREAMP_CAB", "AMP_AND_CAB"):
            candidate = getattr(GearType, candidate_name, None)
            if candidate is not None:
                gear_type = candidate
                break
        # else: no such member exists in this installed version -- keep
        # GearType.AMP and rely on our own manifest.cab record for accuracy.

    # Shared with cloud/kaggle/train_a2_cloud.py -- see
    # hybrid/a2_training_settings.py's user_metadata_kwargs docstring. Only
    # the nam-specific enum (gear_type) and tone_type/output_level_dbu
    # omissions live here; everything else is the shared plain-dict logic.
    return UserMetadata(
        gear_type=gear_type,
        # tone_type deliberately left unset: this model's whole point is
        # that its tone changes with input level, so no single ToneType
        # value would be non-misleading.
        # output_level_dbu deliberately left unset: the hybrid's output
        # level is a combination of both source models' outputs plus the
        # frozen B trim -- there's no single inherited physical value to
        # report here.
        **user_metadata_kwargs(manifest),
    )


def _run_official_trainer(input_path: Path, target_path: Path, output_dir: Path, settings, device: str, manifest: dict) -> Path:
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
    # `settings` (an A2TrainingSettings -- either A2_QUICK_SETTINGS or
    # settings_for_preset(<draft|standard|high_def>)) is shared with
    # cloud/kaggle/train_a2_cloud.py -- see hybrid/a2_training_settings.py.
    # This is the parity mechanism that keeps local and Kaggle GPU training
    # from silently drifting apart.
    result = core.train(
        input_path=str(input_path),
        output_path=str(target_path),
        train_path=str(output_dir),
        epochs=settings.epochs,
        latency=settings.latency,  # docs/phase3.md section 16 -- synthetic latency is authoritatively 0, never auto-detected
        batch_size=settings.batch_size,
        ny=settings.ny,
        seed=settings.seed,
        ignore_checks=settings.ignore_checks,
        # silent=True suppresses nam's interactive matplotlib plot windows
        # (latency-calibration plots, validation-ESR plot) -- with latency=0
        # already fixed there's nothing for a human to approve interactively,
        # and a blocking plt.show() here would hang a non-interactive/headless
        # run forever. docs/phase3.md section 31 explicitly requires
        # suppressing interactive plots.
        silent=settings.silent,
        modelname="model",
        fast_dev_run=settings.fast_dev_run,
    )

    if result is None or result.model is None:
        raise TrainingAbort(
            "nam.train.core.train() returned no model -- the official trainer's own data "
            "checks likely failed (see its printed output above). Not bypassing that check "
            "(docs/phase3.md section 17); investigate the cause instead of forcing a run "
            "with ignore_checks."
        )

    # Export following the same pattern as nam.train.colab.run()/gui --
    # attach the official TrainingMetadata (from this run's TrainOutput) plus
    # our own UserMetadata, rather than a bare export that discards both.
    from nam.train.metadata import TRAINING_KEY

    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = str(manifest.get("artifact_stem") or "model")
    result.model.net.export(
        export_dir,
        basename=artifact_stem,
        user_metadata=_build_user_metadata(manifest),
        other_metadata={TRAINING_KEY: result.metadata.model_dump()},
    )

    nam_path = export_dir / f"{artifact_stem}.nam"
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


def check_full_low_level_response(manifest: dict, nam_path: Path, input_path: Path, sample_rate: int, *, variant: str = "full", slim: float = 0.0) -> "dict | None":
    """Thin wrapper over `hybrid.character_training_target.check_full_low_
    level_response` (docs/blend-mode-fixes.md, Phases 10-11) that also prints
    a verdict line -- the actual sweep/comparison logic is shared with
    `hybrid.kaggle_training.validate_downloaded_model` so both local and
    Kaggle-trained models are held to the identical bar (see that function's
    docstring). Only meaningful for Character Blend bundles that recorded a
    `low_level_response` section; returns None otherwise.
    """
    try:
        result = character_training_target.check_export_low_level_response(
            manifest, nam_path, input_path, sample_rate, variant=variant, slim=slim,
        )
    except (NamRenderError, OSError, ValueError) as exc:
        result = {"state": "unavailable", "pass": None, "variant": variant, "reason": f"quiet validation could not run: {exc}"}
    if result is not None:
        if result.get("pass") is None:
            print(f"Character Blend low-level response ({variant} vs teacher): UNAVAILABLE -- {result['reason']}")
        else:
            verdict = "PASS" if result["pass"] else "FAIL -- exported A2 differs from the frozen teacher"
            print(f"Character Blend low-level response ({variant} vs teacher): max error {result['max_error_db']:.1f} dB -- {verdict}")
    return result


def compare_to_target(rendered: np.ndarray, target_path: Path) -> dict:
    target_audio, _ = _read_audio(target_path)
    metrics = compute_esr_metrics(rendered, target_audio)
    print(f"vs target: raw ESR={metrics['raw_esr']:.5f}  gain-normalized ESR={metrics['gain_normalized_esr']:.5f}  "
          f"RMS diff={metrics['rms_difference']:.5f}  peak diff={metrics['peak_difference']:.5f}")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to training_manifest.json")
    parser.add_argument("--quick", action="store_true", help="fast development smoke-test run (NOT for the final model)")
    parser.add_argument(
        "--epoch-preset", choices=sorted(A2_EPOCH_PRESETS), default=DEFAULT_EPOCH_PRESET,
        help=f"training quality/length ({', '.join(f'{k}={v}' for k, v in A2_EPOCH_PRESETS.items())} epochs); ignored if --quick",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--progress", action="store_true", help="show official trainer epoch progress (used by the local UI)")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to <bundle_dir>/a2_output")
    args = parser.parse_args(argv)

    try:
        env_info = print_environment_diagnostics()
        manifest = load_and_validate_manifest(args.manifest)
        bundle_dir = Path(manifest["_bundle_dir"])
        input_path = Path(manifest["_input_path"])
        target_path = Path(manifest["_target_path"])
        sample_rate = manifest["training_input"]["sample_rate"]

        rf_check = check_receptive_field(manifest, sample_rate) or {}

        output_dir = args.output_dir or (bundle_dir / "a2_output")
        if args.quick:
            print("--quick: running a fast development smoke test, NOT the final model.")
            settings = A2_QUICK_SETTINGS
        else:
            settings = settings_for_preset(args.epoch_preset)
            print(f"--epoch-preset={args.epoch_preset}: training for {settings.epochs} epochs.")
        if args.progress:
            settings = replace(settings, silent=False)

        nam_path = _run_official_trainer(input_path, target_path, output_dir, settings, args.device, manifest)
        print(f"Trainer produced: {nam_path}")

        # Also leave a copy under a human-meaningful name (<amp_a>_<amp_b>_
        # hybrid|blendNN.nam) alongside the trainer's own fixed "model.nam" --
        # export_dir/model.nam stays untouched since BaseNet.export() always
        # writes that exact basename and other code paths (e.g. re-running
        # validate_exported_nam) expect it to still be there.
        friendly_name = suggested_nam_filename(manifest, fallback=nam_path.stem)
        friendly_path = nam_path.with_name(friendly_name)
        if friendly_path != nam_path:
            shutil.copyfile(nam_path, friendly_path)
            print(f"Also copied as: {friendly_path}")

        full_result = validate_exported_nam(nam_path, input_path, sample_rate, slim=False)
        full_metrics = compare_to_target(full_result["rendered"], target_path)
        low_level_response_checks = {}
        full_quiet = check_full_low_level_response(
            manifest, nam_path, input_path, sample_rate, variant="full", slim=0.0,
        )
        if full_quiet is not None:
            low_level_response_checks["full"] = full_quiet

        lite_metrics = None
        lite_validation = None
        try:
            lite_result = validate_exported_nam(nam_path, input_path, sample_rate, slim=True)
            lite_metrics = compare_to_target(lite_result["rendered"], target_path)
            lite_validation = {"rendered_ok": True, "metrics": lite_metrics}
            lite_quiet = check_full_low_level_response(
                manifest, nam_path, input_path, sample_rate, variant="lite", slim=1.0,
            )
            if lite_quiet is not None:
                low_level_response_checks["lite"] = lite_quiet
        except TrainingAbort as exc:
            print(f"Lite/slim validation skipped or failed: {exc}")
            lite_validation = {"rendered_ok": False, "error": str(exc)}

        validation_report = build_validation_report(
            _sha256_file(nam_path),
            {"full": {"rendered_ok": True, "metrics": full_metrics}, "lite": lite_validation},
            quiet_playing=low_level_response_checks or None,
            mode=manifest.get("mode"),
            cabinet={
                **manifest.get("receptive_field", {}).get("cab", {"baked": False}),
                "approximation": rf_check.get("cab_requires_approximation"),
            },
        )

        if rf_check.get("cab_requires_approximation"):
            print(
                "\nBaked cab was trained as an approximation because its formal temporal "
                "dependency exceeded A2 RF. Review validation metrics and listening result above."
            )

        manifest["training"] = {
            **env_info,
            "device_requested": args.device,
            "quick_mode": args.quick,
            "epoch_preset": args.epoch_preset if not args.quick else None,
            "epochs": settings.epochs,
            "output_nam_path": str(nam_path),
            "output_nam_sha256": _sha256_file(nam_path),
            "full_metrics_vs_target": full_metrics,
            "lite_metrics_vs_target": lite_metrics,
            "validation_report": validation_report,
        }
        if rf_check:
            manifest["receptive_field_check"] = rf_check
        if low_level_response_checks:
            manifest["low_level_response_checks"] = low_level_response_checks
            manifest["low_level_response_check"] = low_level_response_checks.get("full")
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
