"""Metadata describing how a generated hybrid target was produced.

The .nam/.wav formats have no agreed place to embed this today, so we write it
as a JSON sidecar next to the generated audio file (e.g. `foo.wav` +
`foo.hybrid.json`) rather than guessing at an embedding scheme. If NAM tooling
later adds a supported metadata field for derived/synthetic models, this is the
dict to put there.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class HybridMetadata:
    amp_a: str
    amp_b: str
    crossover_dbfs: float
    transition_width_db: float
    amp_a_trim_db: float
    amp_b_trim_db: float
    level_match: str  # "automatic" or "manual"
    blend_algorithm: str = "smoothstep-linear"

    # Input-profile/calibration provenance (see hybrid/pipeline.py,
    # docs/INPUT_PROFILE_RESEARCH.md) -- not yet populated by any generation
    # step (no A2 target generation exists yet), but recorded here now so a
    # future generated hybrid is reproducible: which virtual instrument/pickup
    # profile and which NAM calibration assumptions produced it.
    instrument_type: Optional[str] = None
    input_profile_id: Optional[str] = None
    input_profile_gain_db: Optional[float] = None
    calibration_mode: Optional[str] = None
    reference_input_level_dbu: Optional[float] = None
    amp_a_input_level_dbu: Optional[float] = None
    amp_b_input_level_dbu: Optional[float] = None
    amp_a_calibration_gain_db: Optional[float] = None
    amp_b_calibration_gain_db: Optional[float] = None

    def to_dict(self) -> dict:
        return {"hybrid": asdict(self)}

    def write_sidecar(self, audio_path: str | Path) -> Path:
        """Write this metadata to `<audio_path stem>.hybrid.json` next to the audio file."""
        audio_path = Path(audio_path)
        sidecar_path = audio_path.with_suffix(".hybrid.json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return sidecar_path
