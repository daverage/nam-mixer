"""Safe structural edits for Neural Amp Modeler JSON (.nam) files.

These helpers intentionally know only the documented final-output scales.  They
never walk arbitrary ``head_scale`` keys, since those may belong to controls or
conditioning networks rather than the audio output head.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from hybrid.core.nam_loader import NamModel


class NamToolError(ValueError):
    """Raised when a NAM layout cannot be edited safely."""


# NAM 0.13's UserMetadata schema. ``date``, ``training`` and ``loudness`` are
# exporter/trainer-owned values, so this editor deliberately does not forge
# them; loudness is changed only by the output-volume operation.
#
# `gear_type` is deliberately NOT here (see the metadata-categories design
# review): it's a fact about what was actually built -- amp-only vs.
# amp+cab -- determined by the export/cabinet mode at generation time
# (hybrid/training/nam_provenance.py, scripts/train_a2.py), not a free-text label a
# user can retroactively relabel on an already-exported file. It's still
# shown to the user via READ_ONLY_METADATA_FIELDS/inspect(), just not
# editable.
EDITABLE_METADATA_FIELDS = frozenset({
    "name", "modeled_by", "gear_make", "gear_model", "tone_type",
})
READ_ONLY_METADATA_FIELDS = frozenset({"gear_type"})
_STRING_METADATA_FIELDS = frozenset({"name", "modeled_by", "gear_make", "gear_model"})
_TONE_TYPES = frozenset({"clean", "overdrive", "crunch", "hi_gain", "fuzz"})


def load_nam(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise NamToolError("a NAM file must contain a JSON object")
    return data


def calculate_gain_multiplier(db_change: float) -> float:
    if not math.isfinite(db_change):
        raise NamToolError("dB change must be a finite number")
    try:
        multiplier = 10 ** (db_change / 20.0)
    except OverflowError as exc:
        raise NamToolError("dB change is too large to represent safely") from exc
    if not math.isfinite(multiplier):
        raise NamToolError("dB change is too large to represent safely")
    return multiplier


def find_output_scalers(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return known final-audio output scale locations, or refuse to guess."""
    config = data.get("config")
    if not isinstance(config, dict):
        raise NamToolError("unsupported NAM: missing object config")
    architecture = data.get("architecture")
    if architecture == "SlimmableContainer":
        submodels = config.get("submodels")
        if not isinstance(submodels, list) or not submodels:
            raise NamToolError("SlimmableContainer has no submodels to edit")
        found = []
        for index, submodel in enumerate(submodels):
            try:
                model_config = submodel["model"]["config"]
            except (KeyError, TypeError) as exc:
                raise NamToolError(
                    f"SlimmableContainer submodel {index} has no recognised audio model config"
                ) from exc
            if not isinstance(model_config, dict) or "head_scale" not in model_config:
                raise NamToolError(f"SlimmableContainer submodel {index} has no output head_scale")
            if not isinstance(model_config["head_scale"], (int, float)):
                raise NamToolError(f"SlimmableContainer submodel {index} output head_scale is not numeric")
            found.append((f"config.submodels[{index}].model.config.head_scale", model_config))
        return found
    # Older single-output files: only the root model config is unambiguous.
    if "head_scale" in config and isinstance(config["head_scale"], (int, float)):
        return [("config.head_scale", config)]
    raise NamToolError(
        f"unsupported or ambiguous NAM architecture {architecture!r}: no recognised final output head_scale"
    )


def describe_nam_tools(data: dict[str, Any]) -> dict[str, Any]:
    """Return safe editor data and read-only calibration without model internals.

    Volume adjustment and metadata editing are independent features: an
    architecture find_output_scalers() can't safely handle (e.g.
    "Sequential", an embedded-cab export -- see hybrid/training/sequential_nam.py)
    must not block the metadata editor, which only ever touches the
    top-level `metadata` object regardless of architecture. Any such
    failure is reported via `volume_unsupported_reason` instead of raised,
    so the UI can disable just the volume control and keep the rest working.
    """
    try:
        scalers = find_output_scalers(data)
        volume_unsupported_reason = None
    except NamToolError as exc:
        scalers = []
        volume_unsupported_reason = str(exc)
    model = NamModel(path=Path("<metadata>"), raw=data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    scales = [{"path": path, "value": config["head_scale"]} for path, config in scalers]
    loudness = metadata.get("loudness")
    if not isinstance(loudness, (int, float)):
        loudness_values = []
        for path, _ in scalers:
            if path.startswith("config.submodels"):
                index = int(path.split("[")[1].split("]")[0])
                value = data["config"]["submodels"][index]["model"].get("metadata", {}).get("loudness")
                if isinstance(value, (int, float)):
                    loudness_values.append(value)
        loudness = sum(loudness_values) / len(loudness_values) if loudness_values else None
    exact_cab_embed_supported = False
    exact_cab_embed_reason = None
    try:
        from .sequential_nam import SequentialNamError, extract_full_a2_child

        extract_full_a2_child(data)
        exact_cab_embed_supported = True
    except SequentialNamError as exc:
        exact_cab_embed_reason = str(exc)
    return {
        "architecture": data.get("architecture"),
        "sample_rate": data.get("sample_rate"),
        "exact_cab_embed_supported": exact_cab_embed_supported,
        "exact_cab_embed_reason": exact_cab_embed_reason,
        "metadata": {key: metadata.get(key) for key in EDITABLE_METADATA_FIELDS if key in metadata},
        "read_only_metadata": {key: metadata.get(key) for key in READ_ONLY_METADATA_FIELDS if key in metadata},
        "head_scales": scales,
        "volume_unsupported_reason": volume_unsupported_reason,
        "loudness_db": loudness,
        "calibration": {
            "input_level_dbu": model.input_level_dbu,
            "output_level_dbu": model.output_level_dbu,
            "status": model.calibration_status,
        },
    }


def compare_changes(original: Any, modified: Any, path: str = "") -> list[str]:
    """Recursively list every JSON path whose value changed."""
    if type(original) is not type(modified):
        return [path or "<root>"]
    if isinstance(original, dict):
        paths = []
        # Preserve JSON insertion order so CLI/API change reports are stable.
        keys = list(original)
        keys.extend(key for key in modified if key not in original)
        for key in keys:
            child = f"{path}.{key}" if path else key
            if key not in original or key not in modified:
                paths.append(child)
            else:
                paths.extend(compare_changes(original[key], modified[key], child))
        return paths
    if isinstance(original, list):
        if len(original) != len(modified):
            return [path or "<root>"]
        paths = []
        for index, (before, after) in enumerate(zip(original, modified)):
            paths.extend(compare_changes(before, after, f"{path}[{index}]"))
        return paths
    return [] if original == modified else [path or "<root>"]


def apply_volume_change(data: dict[str, Any], db_change: float) -> tuple[dict[str, Any], list[str], float]:
    """Edit only recognised audio-output scales and loudness metadata."""
    db_change = float(db_change)
    multiplier = calculate_gain_multiplier(db_change)
    result = deepcopy(data)
    scalers = find_output_scalers(result)
    expected = []
    # Only paths whose value actually changes are expected: a 0 dB request, or
    # a head_scale of 0, changes nothing and must be a no-op, not a rejection.
    for path, config in scalers:
        before = config["head_scale"]
        config["head_scale"] *= multiplier
        if config["head_scale"] != before:
            expected.append(path)
        # The only submodel metadata associated with a recognised scaler.
        metadata_path = path.rsplit(".config.head_scale", 1)[0] + ".metadata.loudness"
        if path.startswith("config.submodels") and db_change != 0.0:
            index = int(path.split("[")[1].split("]")[0])
            metadata = result["config"]["submodels"][index]["model"].get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("loudness"), (int, float)):
                metadata["loudness"] += db_change
                expected.append(metadata_path)
    metadata = result.get("metadata")
    if db_change != 0.0 and isinstance(metadata, dict) and isinstance(metadata.get("loudness"), (int, float)):
        metadata["loudness"] += db_change
        expected.append("metadata.loudness")
    validate_changes(data, result, expected)
    return result, expected, multiplier


def apply_metadata_changes(data: dict[str, Any], updates: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply explicit top-level metadata fields only (never config or weights)."""
    if not isinstance(updates, dict) or not updates:
        raise NamToolError("provide at least one metadata field to update")
    if any(not isinstance(key, str) or not key.strip() for key in updates):
        raise NamToolError("metadata field names must be non-empty strings")
    if set(updates) - EDITABLE_METADATA_FIELDS:
        raise NamToolError("metadata editor only permits: " + ", ".join(sorted(EDITABLE_METADATA_FIELDS)))
    for key, value in updates.items():
        if value is None:
            continue
        if key in _STRING_METADATA_FIELDS and not isinstance(value, str):
            raise NamToolError(f"metadata.{key} must be text")
        if key == "tone_type" and value not in _TONE_TYPES:
            raise NamToolError("metadata.tone_type is not a NAM tone type")
    result = deepcopy(data)
    had_metadata = "metadata" in data
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise NamToolError("top-level metadata is not an object")
    for key, value in updates.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    # Expect exactly what changed: a new metadata object as a whole, or the
    # individual fields whose value differs (clearing an absent field is a no-op).
    if not had_metadata:
        if not metadata:
            del result["metadata"]
        expected = ["metadata"] if metadata else []
    else:
        missing = object()
        before = data["metadata"]
        expected = [f"metadata.{key}" for key in updates if before.get(key, missing) != metadata.get(key, missing)]
    validate_changes(data, result, expected)
    return result, expected


def validate_changes(original: dict[str, Any], modified: dict[str, Any], expected_paths: list[str]) -> list[str]:
    changed = compare_changes(original, modified)
    if set(changed) != set(expected_paths):
        unexpected = sorted(set(changed) ^ set(expected_paths))
        raise NamToolError("unsafe edit rejected; changed paths differ from approved paths: " + ", ".join(unexpected))
    return changed


def save_nam(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
