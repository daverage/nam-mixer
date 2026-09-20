"""Phase 4D: which captures carry distinct information? -> work/p4/<amp>/selection.json. No neural training.
Exhaustive search over subsets of the eligible integer captures; objective = how well the OMITTED captures are reproduced by
shape-preserving interpolation of the SELECTED captures' measured response along the knob axis, per dimension.
Eligible anchors: VALID or CORRECTED captures only (SUSPECT ones are analysis-only). Level is reported, not optimised.
Usage: p4_select.py <amp>"""
import itertools
import json
import os
import sys

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from p4_common import load_json, out_dir  # noqa: E402

amp = sys.argv[1]
D = out_dir(amp)
PR = load_json(D / "profile.json")
AUD = load_json(D / "audit.json")["captures"]
gains = PR["gains"]
key = lambda g: f"{g:g}"
GROUPS = {
    "tone": ["EQ sub", "EQ low", "EQ lowmid", "EQ mid", "EQ presence", "EQ air", "tilt (presence-low)"],
    "saturation": ["HF >3 kHz", "THD 440Hz @-30", "THD 440Hz @-18", "THD 440Hz @-6", "H2 440Hz @-18", "H3 440Hz @-18", "crest @0"],
    "compression": ["IO gain -54->-30", "IO gain -30->0", "dyn range @0", "dyn range @+6"],
    "level": ["music RMS @0 dB DI", "music RMS @-12 dB DI", "music RMS @+6 dB DI"],
}
SEL_DIMS = ["tone", "saturation", "compression"]
# physical-unit reporting groups and working tolerances (NOT perceptual measurements)
PHYS = {"tone EQ (dB)": (["EQ sub", "EQ low", "EQ lowmid", "EQ mid", "EQ presence", "EQ air"], 1.0), "HF>3k (dB)": (["HF >3 kHz"], 1.0),
        "crest (dB)": (["crest @0"], 1.0), "THD (dB)": (["THD 440Hz @-30", "THD 440Hz @-18", "THD 440Hz @-6"], 5.0),
        "IO slope (dB)": (["IO gain -54->-30", "IO gain -30->0"], 2.0), "dyn range (dB)": (["dyn range @0", "dyn range @+6"], 1.5),
        "level (dB)": (["music RMS @0 dB DI", "music RMS @-12 dB DI", "music RMS @+6 dB DI"], None)}
S = PR["series"]
FLOORED = [n for n in S if n.startswith(("THD", "H2", "H3"))]
def val(n):
    v = np.array(S[n]["values"], float)
    return np.maximum(v, -80.0) if n in FLOORED else v     # a sine THD/harmonic below -80 dB is numerical floor, not signal
_rng_cache = {}
def rng_(n):
    if n not in _rng_cache:
        v = val(n); _rng_cache[n] = max(float(np.ptp(v)), 1e-9)
    return _rng_cache[n]
gi_all = np.array(gains)
ints = [i for i, g in enumerate(gains) if g == int(g)]
half = [i for i, g in enumerate(gains) if g != int(g)]
eligible = [i for i in ints if AUD[key(gains[i])]["status"] in ("VALID", "CORRECTED")]
level_q = {i for i, g in enumerate(gains) if any(e.startswith("level:") for e in AUD[key(g)]["evidence"])}
G = np.array(gains)

def predict(sel, names, positions):
    """PCHIP through selected positions (flat hold outside the selected span); returns (len(positions), len(names))."""
    sel = sorted(sel); xs = G[sel]; out = np.zeros((len(positions), len(names)))
    for j, n in enumerate(names):
        y = val(n)[sel]
        f = PchipInterpolator(xs, y) if len(xs) > 1 else (lambda x, y=y: np.full_like(x, y[0], dtype=float))
        p = np.clip(G[positions], xs.min(), xs.max())
        out[:, j] = f(p)
    return out

def dim_errors(sel, positions, dim, normalised=True):
    names = GROUPS[dim]
    pred = predict(sel, names, positions)
    act = np.stack([val(n)[positions] for n in names], axis=1)
    e = np.abs(pred - act)
    if normalised:
        e = e / np.array([rng_(n) for n in names])
    return e.mean(axis=1)   # per position

def J(sel):
    omit = [i for i in ints if i not in sel]
    if not omit: return 0.0
    tot = 0.0
    for d in SEL_DIMS:
        e = dim_errors(sel, omit, d); tot += 0.5 * (e.sum() / len(ints)) + 0.5 * e.max()   # mean over ALL positions (selected = 0 error)
    return tot

def phys(sel, positions):
    out = {}
    for g, (names, tol) in PHYS.items():
        pred = predict(sel, names, positions); act = np.stack([val(n)[positions] for n in names], axis=1)
        e = np.abs(pred - act).mean(axis=1)
        keep = [k for k, p in enumerate(positions) if not (g.startswith("level") and p in level_q)]
        e = e[keep] if keep else e
        out[g] = {"mean": float(e.mean()), "max": float(e.max()), "tol": tol}
    return out

res = {"amp": amp, "eligible": [gains[i] for i in eligible], "ineligible": {key(gains[i]): AUD[key(gains[i])]["status"] for i in ints if i not in eligible}, "by_k": {}, "baselines": {}}
top5 = {}
allc = {}
for k in range(2, len(eligible) + 1):
    scored = sorted(((J(list(c)), c) for c in itertools.combinations(eligible, k)), key=lambda x: x[0])
    best = scored[0][1]
    top5[k] = [{"set": [gains[i] for i in c], "J": j} for j, c in scored[:5]]
    bestne = min((s for s in itertools.combinations(eligible, k) if not (ints[0] in s and ints[-1] in s)), key=lambda s: J(list(s)), default=None)   # endpoints not both forced in
    allc[k] = best
    omit = [i for i in ints if i not in best]
    res["by_k"][str(k)] = {"best": [gains[i] for i in best], "J": J(list(best)), "phys": phys(list(best), omit) if omit else {},
                           "best_without_both_endpoints": [gains[i] for i in bestne] if bestne else None, "J_without_both_endpoints": J(list(bestne)) if bestne else None}
    print(amp, k, [gains[i] for i in best], round(J(list(best)), 4), flush=True)
# baselines the brief mentions: standard G1/G5/G10, evenly spaced 3 / 5
def idx(gs): return [gains.index(float(g)) for g in gs]
for name, gs in {"G1,G5,G10 (v3 C3)": (1, 5, 10), "G1,G3,G5,G7,G10 (v3 C5)": (1, 3, 5, 7, 10)}.items():
    s = idx(gs); omit = [i for i in ints if i not in s]
    res["baselines"][name] = {"sel": list(gs), "J": J(s), "phys": phys(s, omit)}
# distinctness: leave-one-out and greedy marginal contribution
loo = {}
for i in ints:
    if i in (ints[0], ints[-1]):
        continue
    sel = [j for j in ints if j != i]
    loo[key(gains[i])] = {d: float(dim_errors(sel, [i], d)[0]) for d in SEL_DIMS}
res["loo_normalised_error"] = loo
endpoint_omitted = {}
for i in (ints[0], ints[-1]):
    sel = [j for j in ints if j != i]
    endpoint_omitted[key(gains[i])] = {d: float(dim_errors(sel, [i], d)[0]) for d in SEL_DIMS}
res["endpoint_omitted_normalised_error"] = endpoint_omitted
greedy, cur = [], []
for _ in range(len(eligible)):
    nxt = min((i for i in eligible if i not in cur), key=lambda i: J(cur + [i]))
    cur.append(nxt); greedy.append({"add": gains[nxt], "J_after": J(cur)})
res["greedy_order"] = greedy
# independent validation on genuine half-steps (never in the search): predict them from the chosen subsets
if half:
    hv = {}
    for k, best in allc.items():
        hv[str(k)] = phys(list(best), half)
    for nm, b in res["baselines"].items():
        hv[nm] = phys(idx(b["sel"]), half)
    res["half_step_validation"] = hv
# smallest k meeting every physical tolerance on every omitted position (max error) -- reported as evidence, not an imposed limit
kstar = None
for k in sorted(allc):
    ph = res["by_k"][str(k)]["phys"]
    if ph and all(v["max"] <= v["tol"] for g, v in ph.items() if v["tol"] is not None):
        kstar = k; break
res["k_star_all_within_tolerance"] = kstar
susp = [i for i in ints if i not in eligible]
if susp:
    res["errors_at_suspect_positions"] = {str(k): {key(gains[i]): {g: float(v) for g, v in zip(PHYS, [np.abs(predict(list(allc[k]), PHYS[g][0], [i]) - np.stack([val(n)[[i]] for n in PHYS[g][0]], axis=1)).mean() for g in PHYS])} for i in susp if i not in allc[k]} for k in allc}
# response-distance coordinate over ALL validated captures (analysis use): arc length of the normalised tone+saturation+compression profile
names = [n for d in SEL_DIMS for n in GROUPS[d]]
Z = np.stack([val(n) / rng_(n) for n in names], axis=1)
order = np.argsort(G)
seg = np.linalg.norm(np.diff(Z[order], axis=0), axis=1)
s = np.concatenate([[0], np.cumsum(seg)])
res["response_coordinate"] = {"gains": G[order].tolist(), "arc": (s / s[-1]).tolist(), "total": float(s[-1])}
res["top5_by_k"] = {str(k): v for k, v in top5.items()}
json.dump(res, open(D / "selection.json", "w"), indent=1)
print(amp, "eligible", res["eligible"], "k*:", kstar, "| greedy:", [g["add"] for g in greedy])
