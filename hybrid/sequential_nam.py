"""One shared, strict packager for experimental embedded-cab NAM exports.

The conventional A2 artifact remains intact. Because the proven NAMCore
Sequential wrapper does not expose nested Slimmable Full/Lite selection, the
embedded child is an explicitly extracted **Full** WaveNet, selected using
the same normal-container Full fixture exercised by the runtime gate. The
second child is canonical NAMCore Linear and contains
the *prepared* causal IR, optionally scaled by the single final output/safety
gain required after the cabinet stage.  Local and Kaggle completion paths
must both call this module; neither gets a private Sequential serializer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class SequentialNamError(ValueError):
    pass


def weights_sha256(taps: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(taps, dtype="<f4").tobytes()).hexdigest()


def _sample_rate(model: dict[str, Any]) -> int:
    rate = model.get("sample_rate")
    if not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0 or int(rate) != rate:
        raise SequentialNamError("NAM child has no valid sample_rate")
    return int(rate)


def extract_full_a2_child(head_model: dict[str, Any]) -> dict[str, Any]:
    """Extract the deterministic Full child from a conventional A2 export.

    This is deliberately strict: accepting arbitrary NAM JSON would make an
    embedded export look verified even though its Full/Lite semantics were
    not. NAMCore's normal A2/SlimmableContainer fixture selects the lowest
    `max_value` child for `--slim 0.0`; the native gate proves that equivalence.
    """
    if head_model.get("architecture") != "SlimmableContainer":
        raise SequentialNamError("embedded export requires a SlimmableContainer A2 head")
    submodels = (head_model.get("config") or {}).get("submodels")
    if not isinstance(submodels, list) or not submodels:
        raise SequentialNamError("A2 head has no usable SlimmableContainer submodels")
    try:
        selected = min(submodels, key=lambda item: float(item["max_value"]))
        child = selected["model"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SequentialNamError("A2 head has malformed SlimmableContainer submodels") from exc
    if not isinstance(child, dict) or child.get("architecture") != "WaveNet":
        raise SequentialNamError("A2 Full submodel must be a complete WaveNet NAM")
    if _sample_rate(child) != _sample_rate(head_model):
        raise SequentialNamError("A2 Full submodel sample_rate does not match its container")
    return child


def build_embedded_sequential(head_model: dict[str, Any], prepared_taps: np.ndarray, *, sample_rate: int,
                              final_scalar: float = 1.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return canonical v0.7 Sequential JSON and auditable package metadata.

    `head_model` remains a separate conventional reusable head artifact. The
    Sequential child is its verified explicit Full submodel; this avoids the
    runtime's unavailable nested Full/Lite selection.
    """
    if _sample_rate(head_model) != int(sample_rate):
        raise SequentialNamError("trained head sample_rate does not match embedded export sample_rate")
    full_head = extract_full_a2_child(head_model)
    taps = np.asarray(prepared_taps, dtype=np.float32)
    if taps.ndim != 1 or len(taps) == 0 or not np.all(np.isfinite(taps)):
        raise SequentialNamError("prepared cabinet IR must be a non-empty finite mono tap sequence")
    if not np.isfinite(final_scalar) or final_scalar <= 0:
        raise SequentialNamError("final cabinet scalar must be finite and greater than zero")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = (taps * np.float32(final_scalar)).astype(np.float32)
    if not np.all(np.isfinite(scaled)):
        raise SequentialNamError("final cabinet scalar produces non-finite Linear weights")
    linear = {
        "version": "0.7.0",
        "architecture": "Linear",
        "config": {"receptive_field": int(len(scaled)), "bias": False, "implementation": "auto"},
        "weights": scaled.tolist(),
        "sample_rate": int(sample_rate),
    }
    sequential = {
        "version": "0.7.0",
        "architecture": "Sequential",
        "config": {"models": [full_head, linear]},
        "weights": [],
        "sample_rate": int(sample_rate),
    }
    return sequential, {
        "experimental": True,
        "embedded_head_variant": "full_extracted",
        "head_sample_rate": int(sample_rate),
        "linear_sample_rate": int(sample_rate),
        "prepared_ir_tap_count": int(len(taps)),
        "prepared_ir_weights_sha256": weights_sha256(taps),
        "linear_weights_sha256": weights_sha256(scaled),
        "final_linear_scalar": float(final_scalar),
    }


def package_embedded_sequential(head_nam_path: str | Path, destination: str | Path, prepared_taps: np.ndarray,
                                *, sample_rate: int, final_scalar: float = 1.0) -> dict[str, Any]:
    """Serialize the shared package once, for both local and Kaggle results."""
    with open(head_nam_path, encoding="utf-8") as f:
        head = json.load(f)
    sequential, record = build_embedded_sequential(head, prepared_taps, sample_rate=sample_rate,
                                                    final_scalar=final_scalar)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(sequential, f, separators=(",", ":"))
    return {**record, "head_nam_path": str(head_nam_path), "sequential_nam_path": str(destination)}
