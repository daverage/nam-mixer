"""Phase 4D render check (no neural training): would a teacher built from a candidate subset (waveform blend of the two selected
neighbours, weight linear in knob position, timing aligned with the 4A corrections) reproduce the OMITTED real captures?
Compares music features (EQ, HF, crest, dynamic range, level) of the blend vs the real capture on the fit DIs. Usage: p4_select_teacher.py <amp>"""
import json
import os
import sys

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402

from p4_common import FIT_DIS, SR, capture, clip, load_json, out_dir, render  # noqa: E402
from cg_report import feats  # noqa: E402

amp = sys.argv[1]
D = out_dir(amp)
AUD = load_json(D / "audit.json")["captures"]
SEL = load_json(D / "selection.json")
key = lambda g: f"{g:g}"
ints = [g for g in [float(k) for k in AUD] if g == int(g)]
SECS = 12
R = {}
for g in ints:
    c = AUD[key(g)]["correction"]
    sh = -int(c["samples"]) if c and "samples" in c else 0     # music-derived delay of this capture vs the set (samples)
    R[g] = {}
    for d in FIT_DIS:
        y = render(capture(g), clip(d)[: SECS * SR], SR)
        R[g][d] = np.concatenate([y[sh:], np.zeros(sh, y.dtype)]) if sh > 0 else (np.concatenate([np.zeros(-sh, y.dtype), y[:sh]]) if sh < 0 else y)
FE = {g: {d: feats(R[g][d]) for d in FIT_DIS} for g in ints}
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]

def blend(sel, g, d):
    sel = sorted(sel)
    if g <= sel[0]: return R[sel[0]][d]
    if g >= sel[-1]: return R[sel[-1]][d]
    a = max(s for s in sel if s <= g); b = min(s for s in sel if s >= g)
    if a == b: return R[a][d]
    t = (g - a) / (b - a)
    n = min(len(R[a][d]), len(R[b][d]))
    return R[a][d][:n] * (1 - t) + R[b][d][:n] * t

def evaluate(sel):
    out = {}
    for g in ints:
        if g in sel: continue
        e = {"tone EQ (dB)": [], "HF>3k (dB)": [], "crest (dB)": [], "dyn range (dB)": [], "level (dB)": []}
        for d in FIT_DIS:
            f, r = feats(blend(sel, g, d)), FE[g][d]
            e["tone EQ (dB)"].append(np.mean([abs(f[b] - r[b]) for b in EQ]))
            e["HF>3k (dB)"].append(abs(f["hf3k_db"] - r["hf3k_db"])); e["crest (dB)"].append(abs(f["crest_db"] - r["crest_db"]))
            e["dyn range (dB)"].append(abs(f["dyn_range_db"] - r["dyn_range_db"])); e["level (dB)"].append(abs(f["rms_db"] - r["rms_db"]))
        out[key(g)] = {k: float(np.mean(v)) for k, v in e.items()}
    return out

cands = {}
kstar = SEL["k_star_all_within_tolerance"]
TOP = len(sys.argv) > 2 and sys.argv[2] == "top"
for k in (3, 5) + ((kstar,) if kstar and kstar not in (3, 5) else ()):
    cands[f"best k={k}"] = SEL["by_k"][str(k)]["best"]
if TOP:
    cands = {}
    for k in sorted({3, 5, kstar or 6}):
        for r, c in enumerate(SEL["top5_by_k"][str(k)]):
            cands[f"k={k} rank{r+1}"] = c["set"]
if kstar is None:
    cands["best k=7"] = SEL["by_k"]["7"]["best"]
if not TOP:
    cands["G1,G5,G10 (v3 C3)"] = [1.0, 5.0, 10.0]; cands["G1,G3,G5,G7,G10 (v3 C5)"] = [1.0, 3.0, 5.0, 7.0, 10.0]
res = {"amp": amp, "candidates": {}}
for name, sel in cands.items():
    ev = evaluate(sel)
    res["candidates"][name] = {"sel": sel, "per_position": ev,
        "mean": {m: float(np.mean([v[m] for v in ev.values()])) for m in next(iter(ev.values()))},
        "max": {m: float(np.max([v[m] for v in ev.values()])) for m in next(iter(ev.values()))}}
    print(amp, name, sel, {m: round(v, 2) for m, v in res["candidates"][name]["mean"].items()}, flush=True)
(D / ("teacher_top.json" if TOP else "teacher.json")).write_text(json.dumps(res))
