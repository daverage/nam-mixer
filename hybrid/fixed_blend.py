"""Fixed Blend: the second design mode -- see docs/blend-mode.md.

Do not confuse this module with `hybrid/blend.py`, which implements the
level-DRIVEN crossfade used by Dynamic Hybrid mode. Fixed Blend has no
crossover envelope at all: Amp A and Amp B are combined at one constant,
user-chosen ratio, independent of playing level --

    result = A * (1 - mix_b) + B * mix_b

using LINEAR amplitude blending (not equal-power) for the same reason
`hybrid.blend.blend` does: Amp A and Amp B renders are correlated versions of
the same guitar signal, not independent/decorrelated sources.

Shares `hybrid.pipeline.RenderedPair` with Dynamic Hybrid mode -- rendering
Amp A/B is the expensive step and is identical regardless of which design
mode is auditioned afterwards (see hybrid/pipeline.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .align import align_to_reference
from .blend import DEFAULT_TRANSITION_WIDTH_DB  # noqa: F401 -- re-exported for symmetry, unused here
from .cab_ir import CabDesign
from .calibration import DEFAULT_REFERENCE_INPUT_LEVEL_DBU
from .coverage import ACTIVE_SIGNAL_THRESHOLD_DBFS, active_signal_mask

_EPS = 1e-10


def _rms_dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))
    return 20.0 * np.log10(max(rms, _EPS))


@dataclass
class BlendLevelMatchResult:
    amp_a_active_dbfs: float
    amp_b_active_dbfs: float
    suggested_b_trim_db: float
    n_active_samples: int


def compute_active_trim(
    envelope_db: np.ndarray,
    amp_a_render: np.ndarray,
    amp_b_render: np.ndarray,
    active_threshold_dbfs: float = ACTIVE_SIGNAL_THRESHOLD_DBFS,
) -> BlendLevelMatchResult:
    """Suggest a trim (dB) to apply to Amp B so it matches Amp A's loudness
    over the DI's ACTIVE playing material -- deliberately NOT the crossover
    band `hybrid.level_match.compute_crossover_trim` uses, since Fixed Blend
    has no crossover at all (see module docstring / docs/blend-mode.md
    "BLEND LEVEL MATCHING"). Reuses the same active/silence threshold
    convention as `hybrid.coverage.active_signal_mask`.

    `envelope_db` must be sample-aligned with amp_a_render/amp_b_render
    (the profiled dry envelope from the same `RenderedPair`, see
    hybrid.pipeline.render_pair).
    """
    n = min(len(envelope_db), len(amp_a_render), len(amp_b_render))
    mask = active_signal_mask(envelope_db[:n], active_threshold_dbfs)
    a = amp_a_render[:n][mask]
    b = amp_b_render[:n][mask]

    a_db = _rms_dbfs(a)
    b_db = _rms_dbfs(b)
    suggested_trim = a_db - b_db if (np.isfinite(a_db) and np.isfinite(b_db)) else 0.0

    return BlendLevelMatchResult(
        amp_a_active_dbfs=a_db,
        amp_b_active_dbfs=b_db,
        suggested_b_trim_db=suggested_trim,
        n_active_samples=int(mask.sum()),
    )


@dataclass
class BlendResult:
    blend: np.ndarray
    mix_b: float
    auto_trim_db: float
    manual_trim_db: float
    effective_b_trim_db: float
    alignment_offset_samples: int
    level_match: Optional[BlendLevelMatchResult]


def build_fixed_blend(
    pair,  # hybrid.pipeline.RenderedPair
    mix_b: float,
    auto_level: bool = True,
    manual_b_trim_db: float = 0.0,
    align_enabled: bool = False,
) -> BlendResult:
    """Combine an already-rendered amp pair at a FIXED mix ratio. Cheap --
    pure numpy, safe to call on every mix-slider move without re-running NAM
    inference (see hybrid/pipeline.py's cost split).

    `mix_b` is clamped to [0, 1]: 0.0 -> 100% Amp A, 1.0 -> 100% Amp B.
    """
    mix_b = max(0.0, min(1.0, float(mix_b)))

    amp_b_render, offset = align_to_reference(pair.amp_a, pair.amp_b, enabled=align_enabled)

    n = min(len(pair.amp_a), len(amp_b_render), len(pair.envelope_db))
    a = pair.amp_a[:n]
    b = amp_b_render[:n]

    level_match_result: Optional[BlendLevelMatchResult] = None
    auto_trim_db = 0.0
    if auto_level:
        level_match_result = compute_active_trim(pair.envelope_db[:n], a, b)
        auto_trim_db = level_match_result.suggested_b_trim_db

    effective_b_trim_db = auto_trim_db + manual_b_trim_db
    b_trimmed = b * (10.0 ** (effective_b_trim_db / 20.0))

    blended = a * (1.0 - mix_b) + b_trimmed * mix_b

    return BlendResult(
        blend=blended,
        mix_b=mix_b,
        auto_trim_db=auto_trim_db,
        manual_trim_db=manual_b_trim_db,
        effective_b_trim_db=effective_b_trim_db,
        alignment_offset_samples=offset,
        level_match=level_match_result,
    )


@dataclass(frozen=True)
class BlendDesign:
    """Immutable snapshot of an auditioned Fixed Blend, frozen before
    generating a real training target -- analogous to
    `hybrid.design.HybridDesign` for Dynamic Hybrid, see docs/blend-mode.md
    "CODE STRUCTURE FOR FIXED BLEND".

    `effective_b_trim_db` is FROZEN from a `build_fixed_blend()` call the
    user actually auditioned -- target generation reuses it verbatim rather
    than recomputing the active-playing trim against the (completely
    different) official training input, exactly like HybridDesign's own
    `effective_b_trim_db` -- see hybrid/blend_training_target.py.
    """

    amp_a_path: str
    amp_b_path: str

    mix_b: float

    auto_trim_db: float = 0.0
    manual_b_trim_db: float = 0.0
    effective_b_trim_db: float = 0.0

    alignment_enabled: bool = False
    alignment_offset_samples: int = 0

    instrument_type: str = "guitar"
    design_reference_profile_id: str = "vintage_humbucker"
    design_reference_profile_gain_db: float = 0.0

    calibration_mode: str = "auto"
    reference_input_level_dbu: float = DEFAULT_REFERENCE_INPUT_LEVEL_DBU

    amp_a_input_level_dbu: Optional[float] = None
    amp_b_input_level_dbu: Optional[float] = None
    amp_a_calibration_gain_db: float = 0.0
    amp_b_calibration_gain_db: float = 0.0
    calibration_applied: bool = False
    calibration_effective_mode: str = "raw"
    calibration_warning: Optional[str] = None

    design_di_file: Optional[str] = None  # provenance only -- NOT the training input

    mode: str = "blend"

    cab: Optional[CabDesign] = None

    @property
    def mix_a(self) -> float:
        return 1.0 - self.mix_b

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mix_a"] = self.mix_a
        return d

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def read_json(path: str | Path) -> "BlendDesign":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("mix_a", None)  # derived property, not a constructor field
        cab = data.get("cab")
        if isinstance(cab, dict):
            data = {**data, "cab": CabDesign(**cab)}
        return BlendDesign(**data)


def freeze_blend_design(
    pair,  # hybrid.pipeline.RenderedPair
    result: BlendResult,
    amp_a_path: str,
    amp_b_path: str,
    alignment_enabled: bool,
    design_di_file: Optional[str] = None,
    cab: Optional[CabDesign] = None,
) -> BlendDesign:
    """Build a `BlendDesign` from a `RenderedPair`/`BlendResult` the user
    actually auditioned -- the only intended way to construct a real
    (non-test) `BlendDesign`, mirroring `hybrid.design.freeze_design`."""
    return BlendDesign(
        amp_a_path=str(amp_a_path),
        amp_b_path=str(amp_b_path),
        mix_b=result.mix_b,
        auto_trim_db=result.auto_trim_db,
        manual_b_trim_db=result.manual_trim_db,
        effective_b_trim_db=result.effective_b_trim_db,
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
        design_di_file=design_di_file,
        cab=cab,
    )
