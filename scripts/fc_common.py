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

FC_CFG = {"jcm800": ([1.0, 2.0, 4.0, 10.0], [-20.0, -9.2, -1.7, 14.0]), "vibrolux": ([1.0, 2.0, 3.0, 4.0, 7.0, 10.0], [-20.0, -15.6, -7.1, -2.2, 5.7, 14.0])}

_EXP = REPO / "docs" / "expand" / "manifest_frozen.json"
if _EXP.exists():
    for _a, _v in json.loads(_EXP.read_text())["amps"].items(): FC_CFG[_a] = ([float(g) for g in _v["selected_positions"]], [float(t) for t in _v["anchors_input_gain_db"]])

def base_capture(amp):
    """Single-middle-capture baseline: the ELIGIBLE capture whose response-arc coordinate is closest to the midpoint of the eligible span (an actual capture, never an assumed G5)."""
    au = audit(amp); el = sorted(float(k) for k, v in au.items() if v["status"] in ("VALID", "CORRECTED")); ag, arc = response_arc(amp); at = lambda x: float(np.interp(x, ag, arc)); mid = 0.5 * (at(el[0]) + at(el[-1]))
    return min(el, key=lambda g: abs(at(g) - mid))

def fc_model_path(amp, key):
    """FC_s0/FC_s1 -> the final-candidate models; anything else -> the Phase 4E / v3 models (unchanged)."""
    if key == "BASE":
        from single_nam_common import capture_path
        return capture_path(base_capture(amp)), 1.0
    if key.startswith("FC_"):
        d = FCDIR / amp / "FC_bundle"; s = key.split("_s")[1]
        return next(d.glob(f"{amp}_FC_s{s}/*.nam")), json.loads((d / "manifest.json").read_text())["output_scale_c"]
    return model_path(amp, key)

def fc_intended_T(amp, key, gains):
    """Each model's OWN intended Input gain per physical position: FC mapping for FC models, response-distance B mapping for B, fixed rule for v3 C3."""
    if key == "BASE": tb = json.loads((FCDIR / amp / "baseline_T.json").read_text())["T_by_position"]; return [float(tb[f"{g:g}"]) for g in gains]
    if key.startswith("FC_"): g_, T_ = FC_CFG[amp]; return [T_of_position(amp, g_, T_, g) for g in gains]
    if key.startswith("B_"): return intended_T(amp, "B", gains)
    return [fixed_T(g) for g in gains]
