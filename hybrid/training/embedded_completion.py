"""Shared post-training embedded-artifact state machine for local/Kaggle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.cab_ir import CabDesign
from ..core.render import sequential_renderer_record
from .nam_provenance import export_base_name
from .sequential_nam import package_embedded_artifacts


def _sequential_warmup_samples(path: str | Path) -> int:
    """Mirror the canonical A2 WaveNet + Linear prewarm calculation used by
    the native compatibility gate; never use a fixed arbitrary exclusion."""
    import json
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    children = model["config"]["models"]
    wave, linear = children
    history = 1
    for layer in wave["config"]["layers"]:
        history += sum((int(k) - 1) * int(d) for k, d in zip(layer["kernel_sizes"], layer["dilations"]))
        history += int(layer["head"]["kernel_size"]) - 1
    return history + int(linear["config"]["receptive_field"])


def complete_embedded_artifact(manifest: dict[str, Any], head_nam_path: str | Path, output_dir: str | Path,
                               *, sample_rate: int, final_scalar: float, validation_input: str | Path | None = None) -> dict[str, Any]:
    """Return embedded audit state without changing the caller's successful
    head result. Callers persist this record atomically as their commit point."""
    cab_data = manifest.get("cab") or {}
    if cab_data.get("export_mode") != "embedded":
        return {"state": "not_requested"}
    state: dict[str, Any] = {"state": "packaging", "experimental": True, "variant": "full_only"}
    try:
        artifacts = package_embedded_artifacts(head_nam_path, output_dir, CabDesign.from_dict(cab_data),
                                               sample_rate=sample_rate, final_scalar=final_scalar,
                                               base_name=export_base_name(manifest))
        state.update({"state": "packaged", "artifacts": artifacts})
        # Explicitly establish the renderer before callers offer a download.
        state.update({"state": "validating", "renderer": sequential_renderer_record()})
        if validation_input is None:
            return state
        import numpy as np
        import soundfile as sf
        from ..core.cab_ir import apply_cab_ir, get_frozen_prepared_cab_ir
        from ..core.nam_loader import load_nam
        from ..core.render import SLIM_FULL, render, find_sequential_nam_render_exe
        dry, actual_rate = sf.read(validation_input, dtype="float32", always_2d=False)
        if actual_rate != sample_rate or dry.ndim != 1:
            raise ValueError("embedded validation input is not compatible with the frozen sample rate")
        # The embedded package contains the explicit highest-capacity A2
        # child (Full); compare it with that same renderer selection.
        head = render(load_nam(head_nam_path), dry, sample_rate, slim=SLIM_FULL)
        expected = apply_cab_ir(head, get_frozen_prepared_cab_ir(CabDesign.from_dict(cab_data), sample_rate)) * final_scalar
        actual = render(load_nam(artifacts["sequential_nam_path"]), dry, sample_rate,
                        executable=find_sequential_nam_render_exe())
        warmup = _sequential_warmup_samples(artifacts["sequential_nam_path"])
        if len(actual) <= warmup:
            # Nothing would be compared: never report that as "validated".
            raise ValueError(f"embedded validation input ({len(actual)} samples) is not longer than the "
                             f"{warmup}-sample warm-up, so the package could not be checked")
        # Package identity is strict after startup; native gate establishes
        # the exact derived-history policy for supported A2 structures.
        error = float(np.max(np.abs(actual[warmup:] - expected[warmup:])))
        if error > 3e-6:
            raise ValueError(f"embedded Sequential package mismatch after warm-up: {error:g}")
        state.update({"state": "validated", "package_max_abs_error": error, "warmup_samples": warmup,
                      "download_available": True})
        return state
    except Exception as exc:  # noqa: BLE001 -- never let the optional package fail the caller's valid head
        return {"state": "failed", "experimental": True, "variant": "full_only", "error": str(exc) or type(exc).__name__}
