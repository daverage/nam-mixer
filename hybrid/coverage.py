"""Render-free crossover reachability analysis.

Answers "does a realistic instrument input profile actually traverse Amp A ->
transition -> Amp B for these crossover settings?" without running NAM
inference for every profile -- it only needs the (already-rendered) source
DI's envelope, since the smoothstep blend weight is a pure function of the
envelope and the crossover config (see `hybrid/blend.py`). Reuses
`blend_weight` directly rather than re-implementing the smoothstep math.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blend import CrossoverConfig, blend_weight

# A sample below this absolute dBFS is treated as "not actively playing" and
# excluded from coverage percentages, matching the -50 dBFS-relative silence
# convention already used by scripts/analyze_di.py for lead/trail silence.
ACTIVE_SIGNAL_THRESHOLD_DBFS = -50.0

# Boundaries used to bucket a per-sample blend weight into a coarse
# "mostly Amp A / transition / mostly Amp B" classification.
AMP_A_MAX_T = 0.10
AMP_B_MIN_T = 0.90


@dataclass
class CoverageResult:
    profile_id: str
    gain_db: float
    amp_a_fraction: float
    transition_fraction: float
    amp_b_fraction: float


def active_signal_mask(envelope_db: np.ndarray, threshold_dbfs: float = ACTIVE_SIGNAL_THRESHOLD_DBFS) -> np.ndarray:
    return envelope_db > threshold_dbfs


def analyse_profile_coverage(
    source_envelope_db: np.ndarray,
    profiles: list[tuple[str, float]],
    crossover_dbfs: float,
    transition_width_db: float,
) -> list[CoverageResult]:
    """For each `(profile_id, gain_db)`, report what fraction of the DI's
    ACTIVE playing time would land in the Amp A / transition / Amp B regions
    of the blend if that profile's gain were applied to this DI, at the given
    crossover settings. Does not render or otherwise touch NAM inference.
    """
    config = CrossoverConfig(crossover_dbfs=crossover_dbfs, transition_width_db=transition_width_db)
    mask = active_signal_mask(source_envelope_db)
    active_envelope_db = source_envelope_db[mask] if mask.any() else source_envelope_db

    results = []
    for profile_id, gain_db in profiles:
        shifted_db = active_envelope_db + gain_db
        t = blend_weight(shifted_db, config)
        n = len(t)
        if n == 0:
            results.append(CoverageResult(profile_id, gain_db, 0.0, 0.0, 0.0))
            continue
        amp_a_fraction = float(np.mean(t <= AMP_A_MAX_T))
        amp_b_fraction = float(np.mean(t >= AMP_B_MIN_T))
        transition_fraction = max(0.0, 1.0 - amp_a_fraction - amp_b_fraction)
        results.append(CoverageResult(profile_id, gain_db, amp_a_fraction, transition_fraction, amp_b_fraction))
    return results


# Percentile used for the "suggested crossover" hint -- chosen (not derived
# from a formula) as a point somewhat above the median of active playing
# level, on the reasoning that the crossover should sit past "typical"
# playing level rather than at it, so ordinary dynamics don't spend most of
# their time hovering in the transition band. This is a starting point for
# the user to override, not a claimed-optimal value -- see the "Suggested"
# (not "Best"/"Optimal") label used wherever this is surfaced.
SUGGESTED_CROSSOVER_PERCENTILE = 62.5


def envelope_percentiles(active_envelope_db: np.ndarray) -> dict:
    if len(active_envelope_db) == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    p10, p25, p50, p75, p90 = np.percentile(active_envelope_db, [10, 25, 50, 75, 90])
    return {"p10": float(p10), "p25": float(p25), "p50": float(p50), "p75": float(p75), "p90": float(p90)}


def suggest_crossover_dbfs(source_envelope_db: np.ndarray) -> float | None:
    """A starting-point crossover suggestion from this DI's own active-signal
    envelope distribution -- see SUGGESTED_CROSSOVER_PERCENTILE for why this
    particular percentile. Returns None if there's no active signal at all."""
    mask = active_signal_mask(source_envelope_db)
    active = source_envelope_db[mask]
    if len(active) == 0:
        return None
    return float(np.percentile(active, SUGGESTED_CROSSOVER_PERCENTILE))
