"""Phase 4E shared code: features, mappings, regions, model locations. Import after setting SINGLE_NAM_AMP."""
import json
import os
from pathlib import Path

import numpy as np

from single_nam_common import SR, REPO, capture, db, load_di
from hybrid.render import render
from cg_report import feats, harmonics, tone

P4E = REPO / "work" / "p4e"
MAN = json.loads((REPO / "docs" / "phase4e" / "manifest_frozen.json").read_text())
HELD = MAN["evaluation"]["held_out_dis"]; OFFS = MAN["evaluation"]["di_offsets_db"]; SECONDS = MAN["evaluation"]["clip_seconds"]
TONE_F0 = (110.0, 440.0); TONE_LV = (-54.0, -42.0, -30.0, -18.0, -6.0, 0.0)
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]

def clip(name): return load_di(name)[: SECONDS * SR]

def feats2(y):
    f = feats(y)
    env = 20 * np.log10(np.sqrt(np.mean(y[SR // 2: SR // 2 + (len(y) - SR // 2) // 240 * 240].astype(np.float64).reshape(-1, 240) ** 2, axis=1)) + 1e-9)
    act = env[1:] > env.max() - 50
    d = np.diff(env)[act[: len(env) - 1]] if act.any() else np.array([0.0])
    f["rise_p95_db"] = float(np.percentile(d[d > 0], 95)) if (d > 0).any() else 0.0
    return f

def fixed_T(g): return -22.0 + 4.0 * (g - 1.0)

def intended_T(amp, cfg, gains):
    """A and v3 baselines: fixed rule. B: linear in the response coordinate between its declared anchors."""
    if cfg != "B": return [fixed_T(g) for g in gains]
    a = MAN["amps"][amp]; rc = a["response_coordinate"]
    anc = a["anchors_designated_input_gain_db"]["B_response_distance_spacing"]
    ag = [float(k[1:]) for k in anc]; at = [float(np.interp(g, rc["gains"], rc["arc"])) for g in ag]
    return [float(np.interp(np.interp(g, rc["gains"], rc["arc"]), at, list(anc.values()))) for g in gains]

def regions(amp, gains):
    p = MAN["amps"][amp]["regions_for_reporting"]["plateau_from_position"]
    pre = sorted(g for g in gains if g < p); n = len(pre); out = {}
    for i, g in enumerate(pre): out[g] = "low" if i < n / 3 else ("transition" if i < 2 * n / 3 else "high")
    for g in gains:
        if g >= p: out[g] = "plateau"
    return out

def model_path(amp, key):
    """key: v3_C3 / v3_C5 / v3_C10 / A_s0 / A_s1 / B_s0 / B_s1 -> (nam path, output scale c)"""
    if key.startswith("v3_"):
        n = key.split("C")[1]; d = REPO / "work" / "cg" / amp / f"cg_{n}"
        return next(d.glob("*/*.nam")), json.loads((d / "manifest.json").read_text())["output_scale_c"]
    cfg, seed = key.split("_s"); d = P4E / amp / f"{cfg}_bundle"
    return next(d.glob(f"{amp}_P4E_{cfg}_s{seed}/*.nam")), json.loads((d / "manifest.json").read_text())["output_scale_c"]
