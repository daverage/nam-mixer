"""One shared, strict packager for experimental embedded-cab NAM exports.

The learned A2 JSON is deliberately retained byte-for-byte as the first
Sequential child.  The second child is canonical NAMCore Linear and contains
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
    if not isinstance(rate, int) or rate <= 0:
        raise SequentialNamError("NAM child has no valid sample_rate")
    return rate


def build_embedded_sequential(head_model: dict[str, Any], prepared_taps: np.ndarray, *, sample_rate: int,
                              final_scalar: float = 1.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return canonical v0.7 Sequential JSON and auditable package metadata.

    `head_model` is not copied or altered: JSON object identity is preserved
    as the first child so an A2 export remains a conventional reusable head.
    """
    if _sample_rate(head_model) != int(sample_rate):
        raise SequentialNamError("trained head sample_rate does not match embedded export sample_rate")
    taps = np.asarray(prepared_taps, dtype=np.float32)
    if taps.ndim != 1 or len(taps) == 0 or not np.all(np.isfinite(taps)):
        raise SequentialNamError("prepared cabinet IR must be a non-empty finite mono tap sequence")
    if not np.isfinite(final_scalar) or final_scalar < 0:
        raise SequentialNamError("final cabinet scalar must be finite and non-negative")
    scaled = (taps * np.float32(final_scalar)).astype(np.float32)
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
        "config": {"models": [head_model, linear]},
        "weights": [],
        "sample_rate": int(sample_rate),
    }
    return sequential, {
        "experimental": True,
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
