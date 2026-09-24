"""Continuous Gain player Input-gain anchors and mapping (the FC production default).

FC anchors ("response-distance" anchors, `scripts/fc_common.py::anchor_levels`, archived): the measured profile's
arc-length coordinate (from `cg_selection.response_coordinate`) of each selected capture is mapped linearly onto
the plugin-compatible Input-gain range [-20, +14] dB, keeping neighbouring anchors at least 4 dB apart. Anchors
are rounded to 0.1 dB, exactly as the frozen FC configurations record them. Other physical positions map
linearly in the response coordinate between the anchors (`position_input_gain_db`).

The fixed-spacing ladder (v3, `scripts/cg_build.py`, archived) is supported ONLY as an explicit Advanced alternative;
it is never a silent substitute for the FC anchors.

Designated Input gain for a capture = anchor; its chain level is `anchor + REFERENCE_DB` (hybrid.continuous_gain.multi_blend).
"""
from __future__ import annotations

import numpy as np

REFERENCE_DB = -30.0
T_LO = -20.0
T_HI = 14.0
MIN_SEP_DB = 4.0
FIXED_LADDER_LO_DB = -22.0
FIXED_LADDER_STEP_DB = 4.0


def _coordinate(response_coordinate: dict):
    g = np.array(response_coordinate["gains"])
    arc = np.array(response_coordinate["arc"])
    return lambda x: float(np.interp(x, g, arc))


def effective_min_sep(n: int, t_lo: float = T_LO, t_hi: float = T_HI, min_sep: float = MIN_SEP_DB) -> float:
    """The FC 4 dB minimum separation, reduced only when n captures cannot fit inside [t_lo, t_hi] at that
    spacing (n > 9 with the default range); identical to the FC rule for every set that does fit."""
    return min(min_sep, (t_hi - t_lo) / (n - 1))


def response_anchors(response_coordinate: dict, gains: list[float], t_lo: float = T_LO, t_hi: float = T_HI,
                     min_sep: float = MIN_SEP_DB) -> tuple[list[float], list[float]]:
    """FC anchors for a capture set -> (sorted gains, anchor Input gains in dB, rounded to 0.1)."""
    at = _coordinate(response_coordinate)
    s = sorted(float(g) for g in gains)
    if len(s) < 2:
        raise ValueError("need at least two captures to construct anchors")
    min_sep = effective_min_sep(len(s), t_lo, t_hi, min_sep)
    lo, hi = at(s[0]), at(s[-1])
    T = [t_lo + (t_hi - t_lo) * (at(x) - lo) / ((hi - lo) or 1.0) for x in s]
    for _ in range(8):
        for i in range(1, len(T)):
            T[i] = max(T[i], T[i - 1] + min_sep)
        T = [t_lo + (t_hi - t_lo) * (t - T[0]) / ((T[-1] - T[0]) or 1.0) for t in T]
    return s, [round(t, 1) for t in T]


def fixed_ladder_anchors(gains: list[float]) -> tuple[list[float], list[float]]:
    """v3 fixed-spacing anchors (Advanced alternative): -22 + 4 dB per physical step from position 1."""
    s = sorted(float(g) for g in gains)
    return s, [FIXED_LADDER_LO_DB + FIXED_LADDER_STEP_DB * (g - 1.0) for g in s]


def position_input_gain_db(response_coordinate: dict, gains: list[float], anchors_db: list[float], position: float) -> float:
    """Intended Input gain for ANY physical position: linear in the response coordinate between the anchors."""
    at = _coordinate(response_coordinate)
    return float(np.interp(at(position), [at(x) for x in gains], anchors_db))


def mapping_table(response_coordinate: dict, gains: list[float], anchors_db: list[float], positions: list[float]) -> list[dict]:
    """The actual player Input-gain mapping (physical position -> Input gain), flagged as training anchor or interpolation."""
    anchor = {float(g): a for g, a in zip(gains, anchors_db)}
    return [{"position": float(p), "input_gain_db": anchor[float(p)] if float(p) in anchor else position_input_gain_db(response_coordinate, gains, anchors_db, float(p)),
             "kind": "training_anchor" if float(p) in anchor else "interpolated"} for p in sorted(positions)]
