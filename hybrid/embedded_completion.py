"""Shared post-training embedded-artifact state machine for local/Kaggle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .cab_ir import CabDesign
from .render import NamRenderError, sequential_renderer_record
from .sequential_nam import package_embedded_artifacts


def complete_embedded_artifact(manifest: dict[str, Any], head_nam_path: str | Path, output_dir: str | Path,
                               *, sample_rate: int, final_scalar: float) -> dict[str, Any]:
    """Return embedded audit state without changing the caller's successful
    head result. Callers persist this record atomically as their commit point."""
    cab_data = manifest.get("cab") or {}
    if cab_data.get("export_mode") != "embedded":
        return {"state": "not_requested"}
    state: dict[str, Any] = {"state": "packaging", "experimental": True, "variant": "full_only"}
    try:
        artifacts = package_embedded_artifacts(head_nam_path, output_dir, CabDesign.from_dict(cab_data),
                                               sample_rate=sample_rate, final_scalar=final_scalar)
        state.update({"state": "packaged", "artifacts": artifacts})
        # Explicitly establish the renderer before callers offer a download.
        state.update({"state": "validating", "renderer": sequential_renderer_record()})
        # Rendering/metric validation is backend-specific until the current
        # native CLI accepts a model-path override; callers must set validated
        # only after their exact output comparison succeeds.
        return state
    except (OSError, ValueError, NamRenderError) as exc:
        return {"state": "failed", "experimental": True, "variant": "full_only", "error": str(exc)}
