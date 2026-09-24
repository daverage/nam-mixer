"""Shared source-model provenance and metadata-inheritance rules.

Used by all three design modes' manifest builders (hybrid/modes/training_target.py,
hybrid/modes/blend_training_target.py, hybrid/modes/character_training_target.py) so the
"what do we know about the two source NAMs, and what may the mixed result
inherit from them" logic lives in exactly one place instead of drifting
across three near-identical dict blocks.

Inheritance rules (do not invent facts about a mixed model):
  - `modeled_by`/`gear_make`/`gear_model` are never auto-copied from either
    source -- a mixed Fender/Mesa model is not genuinely either one. These
    stay user-editable, defaulting to blank/"NAM Mixer".
  - `tone_type` is copied only when both sources report the exact same
    value; otherwise the mixed result leaves it unset for the user.
  - `gear_type` is never inherited at all -- it always describes the
    OUTPUT's own architecture/cabinet mode, set by the exporter itself
    (see scripts/train_a2.py's _build_user_metadata and
    hybrid/training/sequential_nam.py), never a source's gear_type.
"""
from __future__ import annotations

from typing import Optional

from ..core.nam_loader import NamModel

TONE_TYPES = frozenset({"clean", "overdrive", "crunch", "hi_gain", "fuzz"})

# Suffixes appended to the user's editable base name, per hybrid/design-modes
# in CLAUDE.md's export table -- makes the distinction between a no-cab,
# learned-cab, and embedded-cab export legible in ordinary NAM players
# without the user needing to understand architecture/gear_type at all.
SUFFIX_AMP_ONLY = "[Amp Only]"
SUFFIX_LEARNED_CAB = "[Learned Cab]"
SUFFIX_EMBEDDED_CAB = "[Embedded Cab · Full]"


def source_metadata_fields(model: NamModel) -> dict:
    """The locked, non-editable source-NAM metadata fields a manifest's
    amp_a/amp_b block was missing: name/modeled_by/gear_type/gear_make/
    gear_model/tone_type. Additive to each caller's existing filename/path/
    sha256/architecture/sample_rate/input_level_dbu/output_level_dbu keys
    (hybrid/modes/training_target.py, hybrid/modes/blend_training_target.py,
    hybrid/modes/character_training_target.py all build that dict separately --
    this is the one place the "what else do we know about this source NAM"
    fields live, so it can't drift across the three).
    """
    return {
        "name": model.name,
        "modeled_by": model.modeled_by,
        "gear_type": model.gear_type,
        "gear_make": model.gear_make,
        "gear_model": model.gear_model,
        "tone_type": model.tone_type,
    }


def agreed_tone_type(amp_a: NamModel, amp_b: NamModel) -> Optional[str]:
    """The tone_type to copy onto the mixed result, or None.

    Only copied when both sources report the identical, officially
    recognised value -- a clean+hi_gain hybrid is not genuinely either, so
    an unset tone_type (left for the user to fill in, or to leave blank) is
    the honest default rather than guessing one source's label.
    """
    a, b = amp_a.tone_type, amp_b.tone_type
    if a is not None and a == b and a in TONE_TYPES:
        return a
    return None


def cabinet_display_name(cab: Optional[dict]) -> str:
    """The cabinet's user-facing name: its editable display_name, falling
    back to the original IR filename, never auto-derived from IR content."""
    cab = cab or {}
    return str(cab.get("display_name") or cab.get("original_filename") or "Cabinet").strip()


def cab_export_mode(cab: Optional[dict]) -> str:
    """The manifest cab record's export mode. Records written before
    `export_mode` existed only have `baked`; a baked cab was convolved into
    the training target, i.e. today's "learned" mode (same rule as
    hybrid.core.cab_ir.CabDesign.from_dict)."""
    cab = cab or {}
    return cab.get("export_mode") or ("learned" if cab.get("baked") else "none")


def default_model_base_name(manifest: dict) -> str:
    """The base name derived from the sources, for manifests without a
    user-entered model_name."""
    from pathlib import Path

    amp_a_name = Path(manifest.get("amp_a", {}).get("filename", "Amp A")).stem
    amp_b_name = Path(manifest.get("amp_b", {}).get("filename", "Amp B")).stem
    mode = manifest.get("mode", "hybrid")
    if mode == "blend":
        mix_b = manifest.get("design", {}).get("mix_b")
        ratio = f" {round((1 - mix_b) * 100)}-{round(mix_b * 100)}" if mix_b is not None else ""
        return f"Blend {amp_a_name} + {amp_b_name}{ratio}"
    if mode == "character":
        return f"Character Blend {amp_a_name} + {amp_b_name}"
    if mode == "continuous_gain":
        return "Continuous Gain"
    return f"Hybrid {amp_a_name} -> {amp_b_name}"


def export_model_name(manifest: dict) -> str:
    """The display name written into a trained A2 export's metadata -- the
    one naming rule for every export mode, used by local training
    (a2_training_settings.user_metadata_kwargs) and mirrored literally by
    cloud/kaggle/train_a2_cloud.py (parity-tested).

    The base is the user-entered `model_name`, or the source-derived default.
    The suffix states what the model's audio contains:

    - "none" (no cab, or a preview-only cab that needs an external IR): the
      model has no NAM Mixer cabinet stage -> "<base> [Amp Only]".
      Continuous Gain keeps the plain base name (one amp's own gain range).
    - "learned" (and historical records with only `baked: true`): the IR was
      convolved into the training target, so the model learned amp + cabinet
      -> "<base> + <cabinet> [Learned Cab]".
    - "embedded": this trained head is amp-only; the packaged Sequential file
      carries the cabinet and is named by embedded_package_name -> "<base>"
      (a head suffix would double up inside the package name).

    Only newly trained exports use this: a historical export keeps the name
    already written into its .nam, and its download filename comes from the
    manifest's artifact_filename (hybrid.modes.metadata.suggested_nam_filename).
    """
    base_name = str(manifest.get("model_name") or "").strip() or default_model_base_name(manifest)
    cab = manifest.get("cab") or {}
    export_mode = cab_export_mode(cab)
    if export_mode == "learned":
        return f"{base_name} + {cabinet_display_name(cab)} {SUFFIX_LEARNED_CAB}"
    if export_mode == "embedded":
        return base_name
    if manifest.get("mode", "hybrid") == "continuous_gain":
        return base_name
    return f"{base_name} {SUFFIX_AMP_ONLY}"


def embedded_package_name(head_name: Optional[str], cabinet_name: Optional[str]) -> str:
    """Name of the packaged Sequential (head + exact cabinet IR) export."""
    head = head_name if isinstance(head_name, str) else "NAM Head"
    cab = cabinet_name.strip() if isinstance(cabinet_name, str) and cabinet_name.strip() else "Cabinet"
    return f"{head} + {cab} {SUFFIX_EMBEDDED_CAB}"
