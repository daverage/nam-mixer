"""Phase 4E: one ordered playback mapping per model, fitted ONLY on the fit DIs (same objective as Phase 4C 'measured_all'):
tone (6 EQ bands) + saturation (HF, crest, THD@-30) + compression (two IO slopes, dynamic range), level EXCLUDED, monotone DP.
Module (no side effects). fit_mapping(amp, sweep_json_path) -> list of T (dB) aligned with gains of that amp."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
FIT_DIS = ["clean_smooth", "moderate_hotrod", "high_thrash", "high_metalcore"]
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]
TF = np.arange(-48, 34, 1.0)
THD_FLOOR = -80.0
TMAX = 30.0

class _Model:
    def __init__(self, sweep):
        self.d = sweep; self.T = np.array(sweep["T"], float)
        self.io = {f0: (np.array(v["levels"], float), v["stats"]) for f0, v in sweep["io"].items()}
    def music(self, di, f, T): return float(np.interp(T, self.T, np.array([r[f] for r in self.d["music"][di]])))
    def tone(self, lv, f, f0="440"):
        l, st = self.io[f0]; return float(np.interp(lv, l, np.array([r[f] for r in st])))

def fit_mapping(amp, sweep_path):
    CAP = json.loads((ROOT / "work" / "p4" / amp / "captures.json").read_text())["captures"]
    AUD = json.loads((ROOT / "work" / "p4" / amp / "audit.json").read_text())["captures"]
    gains = sorted(float(k) for k in CAP); key = lambda g: f"{g:g}"
    GF = [g for g in gains if g == int(g)]
    rm = lambda g, di, f: CAP[key(g)]["probe"]["music"][f"{di}@0"][f]
    rt = lambda g, lv, f: CAP[key(g)]["probe"]["tones"][f"440@{lv:g}"][f]
    thd = lambda v: max(v, THD_FLOOR)
    FEAT = {}
    for b in EQ: FEAT[b] = ("tone", (lambda b: lambda g, di: rm(g, di, b))(b), (lambda b: lambda M, di, T: M.music(di, b, T))(b))
    FEAT["hf3k_db"] = ("sat", lambda g, di: rm(g, di, "hf3k_db"), lambda M, di, T: M.music(di, "hf3k_db", T))
    FEAT["crest_db"] = ("sat", lambda g, di: rm(g, di, "crest_db"), lambda M, di, T: M.music(di, "crest_db", T))
    FEAT["thd@-30"] = ("sat", lambda g, di: thd(rt(g, -30, "thd_db")), lambda M, di, T: thd(M.tone(-30 + T, "thd_db")))
    FEAT["io1"] = ("comp", lambda g, di: rt(g, -30, "out_rms_db") - rt(g, -54, "out_rms_db") - 24, lambda M, di, T: M.tone(-30 + T, "out_rms_db") - M.tone(-54 + T, "out_rms_db") - 24)
    FEAT["io2"] = ("comp", lambda g, di: rt(g, 0, "out_rms_db") - rt(g, -30, "out_rms_db") - 30, lambda M, di, T: M.tone(0 + T, "out_rms_db") - M.tone(-30 + T, "out_rms_db") - 30)
    FEAT["dyn"] = ("comp", lambda g, di: rm(g, di, "dyn_range_db"), lambda M, di, T: M.music(di, "dyn_range_db", T))
    SCALE = {f: max(float(np.ptp([FEAT[f][1](g, FIT_DIS[0]) for g in gains])), 1.0) for f in FEAT}
    M = _Model(json.loads(Path(sweep_path).read_text()))
    C = np.zeros((len(GF), len(TF)))
    dims = ["tone", "sat", "comp"]
    for i, g in enumerate(GF):
        for dm in dims:
            fs = [f for f in FEAT if FEAT[f][0] == dm]
            for k, T in enumerate(TF):
                if T > TMAX: C[i, k] = 1e9; continue
                C[i, k] += np.mean([abs(FEAT[f][2](M, di, T) - FEAT[f][1](g, di)) / SCALE[f] for f in fs for di in FIT_DIS]) / len(dims)
    n, m = C.shape; best = C[0].copy(); back = []
    for i in range(1, n):
        run = np.minimum.accumulate(best); arg = np.array([int(np.argmin(best[:k + 1])) for k in range(m)])
        back.append(arg); best = C[i] + run
    k = int(np.argmin(best)); path = [k]
    for arg in reversed(back): k = int(arg[k]); path.append(k)
    T = [float(TF[k]) for k in path[::-1]]
    return T if len(GF) == len(gains) else [float(v) for v in PchipInterpolator(GF, T)(gains)], gains
