"""Phase 4C: can a better input-gain mapping make an EXISTING v3 model behave more like the real amp? No retraining.
Fit on FIT_DIS only (one ordered mapping for all DIs/levels, level excluded from the fit), evaluate on HELD_OUT_DIS.
Compares: fixed 4 dB spacing (v3 design) | measured mapping (tone+saturation+compression) | single-dimension mappings (trade-off) |
oracle-per-dimension (upper bound chosen ON held-out data: a diagnostic ceiling, NOT a valid mapping). Usage: p4_mapping.py <amp>"""
import json
import os
import sys

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402

from cg_build import REF, level_of  # noqa: E402
from p4_common import FIT_DIS, HELD_OUT_DIS, load_json, out_dir  # noqa: E402

amp = sys.argv[1]
D = out_dir(amp)
CAP = load_json(D / "captures.json")["captures"]
AUD = load_json(D / "audit.json")["captures"]
gains = sorted(float(k) for k in CAP)
key = lambda g: f"{g:g}"
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]
TF = np.arange(-48, 34, 1.0)                     # fine T grid (dB); model features interpolated between the 3 dB sweep points
THD_FLOOR = -80.0

def real_music(g, di, o, f): return CAP[key(g)]["probe"]["music"][f"{di}@{o:g}"][f]
def real_tone(g, lv, f, f0=440): return CAP[key(g)]["probe"]["tones"][f"{f0}@{lv:g}"][f]

class Model:
    def __init__(self, n):
        self.d = load_json(D / f"model_sweep_{n}.json"); self.T = np.array(self.d["T"], float)
        self.io = {f0: (np.array(v["levels"], float), v["stats"]) for f0, v in self.d["io"].items()}
    def music(self, di, f, T):
        y = np.array([r[f] for r in self.d["music"][di]]); return float(np.interp(T, self.T, y))
    def tone(self, lv_abs, f, f0="440"):
        lv, st = self.io[f0]; y = np.array([r[f] for r in st]); return float(np.interp(lv_abs, lv, y))

# ---- feature definitions: name -> (dimension, unit, real(g, di, o), model(M, di, T, o))
def thd(v): return max(v, THD_FLOOR)
FEAT = {}
for b in EQ:
    FEAT[b] = ("tone", "dB", lambda g, di, o, b=b: real_music(g, di, o, b), lambda M, di, T, o, b=b: M.music(di, b, T + o))
FEAT["hf3k_db"] = ("saturation", "dB", lambda g, di, o: real_music(g, di, o, "hf3k_db"), lambda M, di, T, o: M.music(di, "hf3k_db", T + o))
FEAT["crest_db"] = ("saturation", "dB", lambda g, di, o: real_music(g, di, o, "crest_db"), lambda M, di, T, o: M.music(di, "crest_db", T + o))
FEAT["thd@-30"] = ("saturation", "dB", lambda g, di, o: thd(real_tone(g, -30 + o if o in (-12, 0) else -30, "thd_db")), lambda M, di, T, o: thd(M.tone(-30 + (o if o in (-12, 0) else 0) + T, "thd_db")))
FEAT["io_-54->-30"] = ("compression", "dB", lambda g, di, o: real_tone(g, -30, "out_rms_db") - real_tone(g, -54, "out_rms_db") - 24, lambda M, di, T, o: M.tone(-30 + T, "out_rms_db") - M.tone(-54 + T, "out_rms_db") - 24)
FEAT["io_-30->0"] = ("compression", "dB", lambda g, di, o: real_tone(g, 0, "out_rms_db") - real_tone(g, -30, "out_rms_db") - 30, lambda M, di, T, o: M.tone(0 + T, "out_rms_db") - M.tone(-30 + T, "out_rms_db") - 30)
FEAT["dyn_range_db"] = ("compression", "dB", lambda g, di, o: real_music(g, di, o, "dyn_range_db"), lambda M, di, T, o: M.music(di, "dyn_range_db", T + o))
FEAT["rms_db"] = ("level", "dB", lambda g, di, o: real_music(g, di, o, "rms_db"), lambda M, di, T, o: M.music(di, "rms_db", T + o))
DIMS = ["tone", "saturation", "compression", "level"]
# normalisation: each feature's spread across the real amp's captures, so a dimension's fit cost is scale-free
SCALE = {f: max(float(np.ptp([FEAT[f][2](g, FIT_DIS[0], 0.0) for g in gains])), 1.0) for f in FEAT}
Tmax = 30.0
GF = [g for g in gains if g == int(g)] if any(g != int(g) for g in gains) else gains   # fit positions (half-steps stay untouched)
from scipy.interpolate import PchipInterpolator
def expand(path):
    return path if len(GF) == len(gains) else [float(v) for v in PchipInterpolator(GF, path)(gains)]

def quarantined(g, dim):
    ev = AUD[key(g)]["evidence"]
    return (dim == "level" and any(e.startswith("level:") for e in ev))

def cost_table(M, dis, dims, gl=None):
    gl = gl or GF
    C = np.zeros((len(gl), len(TF)))
    for i, g in enumerate(gl):
        for dm in dims:
            if quarantined(g, dm):
                continue
            fs = [f for f in FEAT if FEAT[f][0] == dm]
            for k, T in enumerate(TF):
                if T > Tmax: C[i, k] = 1e9; continue
                e = np.mean([abs(FEAT[f][3](M, di, T, 0.0) - FEAT[f][2](g, di, 0.0)) / SCALE[f] for f in fs for di in dis])
                C[i, k] += e / len(dims)
    return C

def monotone_path(C):
    n, m = C.shape; best = C[0].copy(); back = []
    for i in range(1, n):
        run = np.minimum.accumulate(best); arg = np.array([int(np.argmin(best[:k + 1])) for k in range(m)])
        back.append(arg); best = C[i] + run
    k = int(np.argmin(best)); path = [k]
    for arg in reversed(back):
        k = int(arg[k]); path.append(k)
    return expand([float(TF[k]) for k in path[::-1]])

def eval_map(M, mapping, dis, o):
    """per-feature mean abs error (physical units) per capture, at DI offset o."""
    out = {}
    for i, g in enumerate(gains):
        out[key(g)] = {}
        for f, (dm, u, rf, mf) in FEAT.items():
            out[key(g)][f] = float(np.mean([abs(mf(M, di, mapping[i], o) - rf(g, di, o)) for di in dis]))
    return out

def dim_err(errs, dm):
    fs = [f for f in FEAT if FEAT[f][0] == dm]
    return {k: float(np.mean([v[f] for f in fs])) for k, v in errs.items()}

fixed = [level_of(g) - REF for g in gains]
result = {"amp": amp, "gains": gains, "fit_gains": GF, "fixed_T": fixed, "models": {}}
for n in (10, 5, 3):
    M = Model(n)
    maps = {"fixed_4dB": fixed}
    allc = cost_table(M, FIT_DIS, ["tone", "saturation", "compression"])
    maps["measured_all"] = monotone_path(allc)
    for dm in ("tone", "saturation", "compression"):
        maps[f"only_{dm}"] = monotone_path(cost_table(M, FIT_DIS, [dm]))
    # RMS-only mapping (the naive one the brief warns against), for the trade-off illustration
    C = np.zeros((len(GF), len(TF)))
    for i, g in enumerate(GF):
        if quarantined(g, "level"): continue
        for k, T in enumerate(TF):
            C[i, k] = 1e9 if T > Tmax else np.mean([abs(FEAT["rms_db"][3](M, di, T, 0.0) - FEAT["rms_db"][2](g, di, 0.0)) for di in FIT_DIS])
    maps["rms_only"] = monotone_path(C)
    ev = {}
    for name, mp in maps.items():
        ev[name] = {"T": mp, "heldout_o0": eval_map(M, mp, HELD_OUT_DIS, 0.0), "fit_o0": eval_map(M, mp, FIT_DIS, 0.0),
                    "heldout_by_offset": {f"{o:g}": eval_map(M, mp, HELD_OUT_DIS, o) for o in (-12.0, -6.0, 6.0)}}
    # oracle per dimension ON HELD-OUT data (upper bound only)
    orc = {}
    for dm in ("tone", "saturation", "compression", "level"):
        Co = cost_table(M, HELD_OUT_DIS, [dm], gains) if dm != "level" else None
        if dm == "level":
            Co = np.zeros((len(gains), len(TF)))
            for i, g in enumerate(gains):
                for k, T in enumerate(TF):
                    Co[i, k] = 1e9 if T > Tmax else np.mean([abs(FEAT["rms_db"][3](M, di, T, 0.0) - FEAT["rms_db"][2](g, di, 0.0)) for di in HELD_OUT_DIS])
        mp = [float(TF[int(np.argmin(Co[i]))]) for i in range(len(gains))]   # un-ordered: best possible per capture
        orc[dm] = {"T": mp, "err": eval_map(M, mp, HELD_OUT_DIS, 0.0)}
    result["models"][str(n)] = {"mappings": ev, "oracle_per_dimension_ub": orc, "trained_gains": M.d["gains"]}
    print(amp, n, "mean held-out dim errors:", {nm: {dm: round(float(np.mean(list(dim_err(e['heldout_o0'], dm).values()))), 3) for dm in DIMS} for nm, e in ev.items() if nm in ("fixed_4dB", "measured_all")}, flush=True)
(D / "mapping.json").write_text(json.dumps(result))
