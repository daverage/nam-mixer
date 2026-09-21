"""Final-candidate task: teacher-only candidate evaluation helpers (no training, no NAM). Import after setting SINGLE_NAM_AMP."""
import json
import numpy as np
from pl_common import *          # teacher, render_aligned, config, REF, PL, P4E, HELD, OFFS, clip, db, feats2, EQ, GainChain, chain_weights ...
from pathlib import Path

FCDIR = P4E / "final"; FCDIR.mkdir(parents=True, exist_ok=True)
MIN_SEP = 4.0

def response_arc(amp):
    sel = json.loads((REPO / "work" / "p4" / amp / "selection.json").read_text())["response_coordinate"]; return np.array(sel["gains"]), np.array(sel["arc"])

def anchor_levels(amp, gains, t_lo=-20.0, t_hi=14.0):
    """Response-distance anchors for a capture set: arc-length coordinate of the measured profile, mapped onto [t_lo, t_hi] dB, >= MIN_SEP apart."""
    g, arc = response_arc(amp); at = lambda x: float(np.interp(x, g, arc)); s = sorted(gains)
    lo, hi = at(s[0]), at(s[-1]); T = [t_lo + (t_hi - t_lo) * (at(x) - lo) / ((hi - lo) or 1.0) for x in s]
    for _ in range(8):
        for i in range(1, len(T)): T[i] = max(T[i], T[i - 1] + MIN_SEP)
        T = [t_lo + (t_hi - t_lo) * (t - T[0]) / ((T[-1] - T[0]) or 1.0) for t in T]
    return s, T

def T_of_position(amp, gains, Tanch, g):
    """Intended Input gain for ANY physical position: linear in the response coordinate between the anchors."""
    ag, arc = response_arc(amp); at = lambda x: float(np.interp(x, ag, arc))
    return float(np.interp(at(g), [at(x) for x in gains], Tanch))
