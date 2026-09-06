"""`HybridDesign`: an immutable snapshot of everything that went into an
auditioned hybrid, frozen before generating a real training target -- see
docs/phase3.md sections 2-3.

The critical rule this module exists to enforce (see docs/phase3.md section
1): `design_reference_profile_gain_db` is DESIGN CONTEXT ONLY. It records
which pickup profile the design was auditioned with and is written into
provenance, but nothing in this module or `hybrid/training_target.py` ever
re-applies it to the official NAM training excitation -- see
`training_target.generate_training_bundle`'s docstring for where that
distinction is actually enforced.

Likewise `effective_b_trim_db` is a FROZEN number, captured once from a
`build_hybrid()` call the user actually auditioned. Target generation must
reuse it verbatim (`auto_level=False, manual_b_trim_db=design.effective_b_trim_db`)
rather than recomputing auto-match against the (completely different) official
training input -- see docs/phase3.md section 3.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .blend import DEFAULT_TRANSITION_WIDTH_DB
from .cab_ir import CabDesign
from .calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU


@dataclass(frozen=True)
class HybridDesign:
    amp_a_path: str
    amp_b_path: str

    crossover_dbfs: float
    transition_width_db: float = DEFAULT_TRANSITION_WIDTH_DB

    auto_trim_db: float = 0.0
    manual_b_trim_db: float = 0.0
    effective_b_trim_db: float = 0.0

    blend_algorithm: str = "smoothstep-linear"

    alignment_enabled: bool = False
    alignment_offset_samples: int = 0

    instrument_type: str = "guitar"
    design_reference_profile_id: str = "vintage_humbucker"
    design_reference_profile_gain_db: float = 0.0

    calibration_mode: str = "auto"  # requested mode
    reference_input_level_dbu: float = DEFAULT_REFERENCE_INPUT_LEVEL_DBU

    amp_a_input_level_dbu: Optional[float] = None
    amp_b_input_level_dbu: Optional[float] = None
    amp_a_calibration_gain_db: float = 0.0
    amp_b_calibration_gain_db: float = 0.0
    calibration_applied: bool = False
    calibration_effective_mode: str = "raw"
    calibration_warning: Optional[str] = None

    envelope_rms_window_ms: float = 20.0
    envelope_attack_avg_ms: float = 5.0
    envelope_release_window_ms: float = 55.0
    envelope_release_range_db: float = 60.0
    envelope_max_history_ms: float = 0.0  # filled in by from_pair_and_result

    design_di_file: Optional[str] = None  # provenance only -- NOT the training input

    mode: str = "hybrid"

    # Shared Cabinet IR stage, mode-independent -- see hybrid/cab_ir.py.
    # `None` (the default) means "no cab selected", identical to every
    # pre-Blend-mode design and fully backward compatible with previously
    # written hybrid_design.json files that predate this field.
    cab: Optional[CabDesign] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def read_json(path: str | Path) -> "HybridDesign":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cab = data.get("cab")
        if isinstance(cab, dict):
            data = {**data, "cab": CabDesign(**cab)}
        return HybridDesign(**data)


def freeze_design(
    pair,  # hybrid.pipeline.RenderedPair
    result,  # hybrid.pipeline.HybridResult
    amp_a_path: str,
    amp_b_path: str,
    crossover_dbfs: float,
    transition_width_db: float,
    alignment_enabled: bool,
    design_di_file: Optional[str] = None,
    blend_algorithm: str = "smoothstep-linear",
    envelope_config=None,
    cab: Optional[CabDesign] = None,
) -> HybridDesign:
    """Build a `HybridDesign` from a `RenderedPair`/`HybridResult` the user
    actually auditioned. This is the ONLY intended way to construct a real
    (non-test) `HybridDesign` -- it pulls every field from what was actually
    rendered/blended rather than letting a caller hand-assemble a design that
    doesn't match what was heard.
    """
    from .envelope import DEFAULT_BOUNDED_ENVELOPE_CONFIG, bounded_envelope_max_history_ms

    envelope_config = envelope_config or DEFAULT_BOUNDED_ENVELOPE_CONFIG
    return HybridDesign(
        amp_a_path=str(amp_a_path),
        amp_b_path=str(amp_b_path),
        crossover_dbfs=crossover_dbfs,
        transition_width_db=transition_width_db,
        auto_trim_db=result.auto_trim_db,
        manual_b_trim_db=result.manual_trim_db,
        effective_b_trim_db=result.effective_b_trim_db,
        blend_algorithm=blend_algorithm,
        alignment_enabled=alignment_enabled,
        alignment_offset_samples=result.alignment_offset_samples,
        instrument_type=pair.instrument_type,
        design_reference_profile_id=pair.input_profile_id,
        design_reference_profile_gain_db=pair.input_profile_gain_db,
        calibration_mode=pair.calibration_mode,
        reference_input_level_dbu=pair.reference_input_level_dbu,
        amp_a_input_level_dbu=pair.amp_a_model_input_level_dbu,
        amp_b_input_level_dbu=pair.amp_b_model_input_level_dbu,
        amp_a_calibration_gain_db=pair.amp_a_calibration_gain_db,
        amp_b_calibration_gain_db=pair.amp_b_calibration_gain_db,
        calibration_applied=pair.calibration_applied,
        calibration_effective_mode=pair.calibration_mode if pair.calibration_applied else "raw",
        calibration_warning=pair.calibration_warning,
        envelope_rms_window_ms=envelope_config.rms_window_ms,
        envelope_attack_avg_ms=envelope_config.attack_avg_ms,
        envelope_release_window_ms=envelope_config.release_window_ms,
        envelope_release_range_db=envelope_config.release_range_db,
        envelope_max_history_ms=bounded_envelope_max_history_ms(envelope_config),
        design_di_file=design_di_file,
        cab=cab,
    )
