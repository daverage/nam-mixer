"""Shared source-model provenance and metadata-inheritance rules.

Used by all three design modes' manifest builders (hybrid/training_target.py,
hybrid/blend_training_target.py, hybrid/character_training_target.py) so the
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
    hybrid/sequential_nam.py), never a source's gear_type.
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
    (hybrid/training_target.py, hybrid/blend_training_target.py,
    hybrid/character_training_target.py all build that dict separately --
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


def export_name_suffix(cab: Optional[dict]) -> str:
    """The automatic suffix for a SlimmableContainer export (no-cab/learned-cab).

    `cab` is the manifest's "cab" dict (or None), whose `export_mode` is
    "none"/"learned"/"embedded" -- see hybrid/cab_ir.py's CabDesign. An
    "embedded" export's SlimmableContainer head is deliberately left
    unsuffixed here: it isn't the final deliverable in that mode (the
    packaged Sequential file is, see hybrid/sequential_nam.py, which adds
    its own SUFFIX_EMBEDDED_CAB), and suffixing the head too would produce
    a doubled-up name like "X [Amp Only] + Y [Embedded Cab - Full]".
    """
    export_mode = (cab or {}).get("export_mode", "none")
    if export_mode == "learned":
        return SUFFIX_LEARNED_CAB
    if export_mode == "embedded":
        return ""
    return SUFFIX_AMP_ONLY


def cabinet_display_name(cab: Optional[dict]) -> str:
    """The cabinet's user-facing name: its editable display_name, falling
    back to the original IR filename, never auto-derived from IR content."""
    cab = cab or {}
    name = str(cab.get("display_name") or "").strip()
    if name:
        return name
    filename = str(cab.get("original_filename") or "").strip()
    return filename or "Cabinet"


def build_export_name(base_name: str, cab: Optional[dict]) -> str:
    """The full auto-suffixed name for a SlimmableContainer export.

    `base_name` is the user-editable name (e.g. project/output name);
    the cabinet clause and technical suffix are appended automatically and
    are never directly editable -- see the Settings/metadata design review.
    """
    base_name = base_name.strip()
    suffix = export_name_suffix(cab)
    export_mode = (cab or {}).get("export_mode", "none")
    if export_mode == "learned":
        return f"{base_name} + {cabinet_display_name(cab)} {suffix}".strip()
    if suffix:
        return f"{base_name} {suffix}".strip()
    return base_name
