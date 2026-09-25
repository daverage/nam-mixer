"""Read-only structural inspection and short NAMCore validation for NAM files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.nam_loader import NamModel, load_nam
from ..core.receptive_field import ReceptiveFieldUnavailable, compute_source_nam_receptive_field
from ..core.render import SLIM_LITE, NamRenderError, render
from ..core.safety import check_audio

_PROBE_SAMPLES = 4096


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _sample_rate(model: NamModel) -> int | None:
    value = _number(model.sample_rate)
    if value is None or value <= 0 or not value.is_integer():
        return None
    return int(value)


def _sequential_stages(raw: dict[str, Any], sample_rate: int | None) -> tuple[list[dict[str, Any]], int | None, int | None]:
    config = raw.get("config")
    models = config.get("models") if isinstance(config, dict) else None
    if not isinstance(models, list):
        return [], None, None
    stages: list[dict[str, Any]] = []
    fir_history = 0
    neural_history = 0
    all_stage_histories_known = True
    for index, child in enumerate(models, start=1):
        if not isinstance(child, dict):
            stages.append({"index": index, "architecture": "Unknown"})
            all_stage_histories_known = False
            continue
        architecture = child.get("architecture")
        child_config = child.get("config")
        stage: dict[str, Any] = {"index": index, "architecture": architecture or "Unknown"}
        child_rate = _number(child.get("sample_rate"))
        if child_rate is not None:
            stage["sample_rate"] = int(child_rate) if child_rate.is_integer() else child_rate
        if architecture == "Linear" and isinstance(child_config, dict):
            length = child_config.get("receptive_field")
            weights = child.get("weights")
            if isinstance(length, int) and length >= 0:
                stage["ir_length_samples"] = length
                if sample_rate:
                    stage["duration_ms"] = round(length * 1000 / sample_rate, 3)
                if isinstance(weights, list) and len(weights) == length:
                    stage["ir_weights_length_matches"] = True
                fir_history += max(0, length - 1)
            else:
                all_stage_histories_known = False
        elif architecture in ("WaveNet", "PackedWaveNet", "SlimmableContainer"):
            try:
                stage_rf = compute_source_nam_receptive_field(child)
                stage["neural_receptive_field_samples"] = stage_rf
                neural_history += max(0, stage_rf - 1)
            except (ReceptiveFieldUnavailable, KeyError, TypeError, ValueError):
                all_stage_histories_known = False
        else:
            all_stage_histories_known = False
        stages.append(stage)
    # A partial sum would understate the dependency, so report nothing unless
    # every stage's history is known.
    if not all_stage_histories_known:
        return stages, None, None
    has_linear = any(stage.get("architecture") == "Linear" for stage in stages)
    return stages, neural_history, fir_history if has_linear else None


def _branch_result(model: NamModel, probe: np.ndarray, sample_rate: int, slim: float | None) -> dict[str, Any]:
    try:
        output = render(model, probe, sample_rate, slim=slim)
        report = check_audio(output)
    except (NamRenderError, OSError, ValueError, RuntimeError) as exc:
        return {"status": "failed", "detail": str(exc)}
    status = "failed" if report.has_nan_or_inf else "silent" if report.is_silent else "passed"
    detail = "Non-finite output" if report.has_nan_or_inf else "Silent output" if report.is_silent else "Render passed"
    return {
        "status": status,
        "detail": detail,
        "sample_count": int(len(output)),
        "sample_rate": sample_rate,
        "input_channels": 1,
        "output_channels": 1,
        "peak_dbfs": None if not np.isfinite(report.peak_dbfs) else round(report.peak_dbfs, 2),
        "finite": not report.has_nan_or_inf,
    }


def inspect_nam(path: str | Path) -> dict[str, Any]:
    """Inspect one NAM without modifying it; return metadata even on render failure."""
    model = load_nam(path)
    raw = model.raw
    if not isinstance(raw, dict):
        raise ValueError("the JSON root must be a NAM object")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    architecture = model.architecture if isinstance(model.architecture, str) else None
    sample_rate = _sample_rate(model)
    input_level = _number(model.input_level_dbu)
    output_level = _number(model.output_level_dbu)
    calibration_status = "complete" if input_level is not None and output_level is not None else (
        "input-only" if input_level is not None else "absent"
    )

    stages: list[dict[str, Any]] = []
    fir_history = None
    sequential_neural_history = None
    if architecture == "Sequential":
        stages, sequential_neural_history, fir_history = _sequential_stages(raw, sample_rate)
    has_linear_stage = any(stage["architecture"] == "Linear" for stage in stages)
    gear_type = metadata.get("gear_type")
    if has_linear_stage:
        cabinet = {"status": "embedded_linear", "label": "Sequential model with embedded Linear cabinet stage"}
    elif gear_type in ("amp_cab", "amp_pedal_cab"):
        cabinet = {"status": "metadata_indicates_cabinet", "label": "Metadata indicates cabinet included"}
    elif gear_type == "amp":
        cabinet = {"status": "amp_only", "label": "Amp only / no embedded cabinet detected"}
    else:
        cabinet = {"status": "unknown", "label": "Unknown"}

    rf_samples = None
    rf_detail = None
    try:
        rf_samples = compute_source_nam_receptive_field(model)
    except (ReceptiveFieldUnavailable, KeyError, TypeError, ValueError) as exc:
        rf_detail = str(exc)
    neural_samples = rf_samples
    total_dependency_samples = rf_samples
    if architecture == "Sequential":
        neural_samples = sequential_neural_history
        total_dependency_samples = (
            1 + sequential_neural_history + (fir_history or 0)
            if sequential_neural_history is not None
            else None
        )
    temporal = {
        "neural_receptive_field_samples": neural_samples,
        "neural_receptive_field_ms": round(neural_samples * 1000 / sample_rate, 3) if neural_samples is not None and sample_rate else None,
        "fir_history_samples": fir_history,
        "fir_history_ms": round(fir_history * 1000 / sample_rate, 3) if fir_history is not None and sample_rate else None,
        "total_formal_dependency_samples": total_dependency_samples,
        "total_formal_dependency_ms": round(total_dependency_samples * 1000 / sample_rate, 3) if total_dependency_samples is not None and sample_rate else None,
        "receptive_field_note": rf_detail,
    }

    is_a2 = architecture == "SlimmableContainer"
    submodels = model.config.get("submodels") if isinstance(model.config, dict) else None
    has_full_lite = is_a2 and isinstance(submodels, list) and len(submodels) >= 2
    branches: dict[str, Any] = {}
    load_status = "passed" if architecture and isinstance(raw.get("config"), dict) and "weights" in raw else "failed"
    load_detail = "NAM metadata parsed" if load_status == "passed" else "JSON parsed, but architecture/config/weights are incomplete"

    if load_status != "passed":
        branches["render"] = {"status": "not_run", "detail": "Could not validate model structure"}
    elif sample_rate is None:
        branches["render"] = {"status": "not_run", "detail": "NAM does not declare a valid positive integer sample rate"}
    else:
        # A deterministic low-level probe with an impulse and seeded noise exercises
        # both transient and sustained model paths while keeping the validation fast.
        rng = np.random.default_rng(0x4E414D)
        probe = rng.standard_normal(_PROBE_SAMPLES).astype(np.float32) * np.float32(0.03)
        probe[0] = np.float32(0.2)
        branches["render"] = _branch_result(model, probe, sample_rate, None)
        if has_full_lite:
            branches["full"] = {**branches["render"], "detail": "Full: " + branches["render"]["detail"]}
            branches["lite"] = _branch_result(model, probe, sample_rate, SLIM_LITE)

    passed = branches.get("render", {}).get("status") == "passed"
    validation_status = "passed" if passed else "failed"
    if branches.get("render", {}).get("status") == "silent":
        validation_status = "warning"
    if has_full_lite:
        validation_status = "passed" if branches.get("full", {}).get("status") == "passed" and branches.get("lite", {}).get("status") == "passed" else "failed"
    if validation_status == "passed":
        validation_summary = "NAMCore render passed"
    elif validation_status == "warning":
        validation_summary = "Silent output"
    else:
        failed_branch = next(((name, branch) for name, branch in branches.items() if branch.get("status") == "failed"), None)
        not_run_branch = branches.get("render", {})
        if failed_branch:
            branch_name, branch = failed_branch
            validation_summary = f"{branch_name.title()} failed: {branch.get('detail', 'render failed')}"
        elif not_run_branch.get("status") == "not_run":
            validation_summary = not_run_branch.get("detail", "Could not validate model")
        else:
            validation_summary = "Could not validate model"

    version = raw.get("version")
    if version is not None and not isinstance(version, (str, int, float)):
        version = None
    result = {
        "identity": {
            "filename": Path(path).name,
            "name": metadata.get("name"), "modeled_by": metadata.get("modeled_by"),
            "gear_make": metadata.get("gear_make"), "gear_model": metadata.get("gear_model"),
            "gear_type": gear_type, "tone_type": metadata.get("tone_type"),
            "format_version": version,
        },
        "architecture": {
            "name": architecture or "Unknown", "sample_rate": sample_rate,
            "a2_packed": is_a2, "slimmable": is_a2, "full_lite_supported": bool(has_full_lite),
            "sequential": architecture == "Sequential", "stages": stages,
            "input_channels": 1 if branches.get("render", {}).get("input_channels") else None,
            "output_channels": 1 if branches.get("render", {}).get("output_channels") else None,
        },
        "calibration": {
            "input_level_dbu": input_level, "output_level_dbu": output_level,
            "loudness_db": _number(metadata.get("loudness")) if not isinstance(metadata.get("loudness"), dict) else _number(metadata["loudness"].get("db")),
            "status": calibration_status,
        },
        "cabinet": cabinet,
        "temporal": temporal,
        "validation": {"metadata_status": load_status, "metadata_detail": load_detail,
                       "status": validation_status, "branches": branches,
                       "summary": validation_summary},
    }
    return result
