"""Evaluate one model against the real references with each playback mapping (direct renders, no interpolation).
Usage: p4e_eval.py <amp> <model key>   -> work/p4e/<amp>/eval_<key>.json"""
import json, os, sys, time
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from p4e_common import *
from hybrid.validation import compute_esr_metrics
from p4e_fit import fit_mapping
from single_nam_common import render_file_nam
amp, key = sys.argv[1], sys.argv[2]
REAL = json.loads((P4E / amp / "real_ref.json").read_text()); gains = REAL["gains"]
nam, c = model_path(amp, key)
rf = lambda x: render_file_nam(nam, x.astype(np.float32)) / c
sweep = REPO / "work" / "p4" / amp / f"model_sweep_{key.split('C')[1]}.json" if key.startswith("v3_") else P4E / amp / f"sweep_{key}.json"
fitted, g_fit = fit_mapping(amp, sweep)
cfg = "B" if key.startswith("B") else "A"
maps = {"fixed": [fixed_T(g) for g in gains], "fitted": fitted}
if cfg == "B": maps["intended"] = intended_T(amp, "B", gains)
else: maps["intended"] = maps["fixed"]
AUD = json.loads((REPO / "work" / "p4" / amp / "audit.json").read_text())["captures"]
_real_cache = {}
def real_wave(g, d):
    if (g, d) not in _real_cache:
        y = render(capture(g), clip(d), SR); c_ = AUD[f"{g:g}"]["correction"]; sh = -int(c_["samples"]) if c_ and "samples" in c_ else 0
        _real_cache[(g, d)] = np.concatenate([y[sh:], np.zeros(sh, y.dtype)]) if sh > 0 else (np.concatenate([np.zeros(-sh, y.dtype), y[:sh]]) if sh < 0 else y)
    return _real_cache[(g, d)]
def lm_esr(cand, ref, warm=SR // 2):
    n = min(len(cand), len(ref)); a, b = cand[warm:n].astype(np.float64), ref[warm:n].astype(np.float64)
    a = a * np.sqrt(np.mean(b ** 2) / max(np.mean(a ** 2), 1e-20)); return float(compute_esr_metrics(a, b)["raw_esr"])
res = {"amp": amp, "model": key, "nam": str(nam.relative_to(REPO)), "gains": gains, "mappings": {k: v for k, v in maps.items()}, "regions": {f"{g:g}": r for g, r in regions(amp, gains).items()}, "results": {}}
t0 = time.time()
done = {}
for name, T in maps.items():
    if name == "intended" and cfg != "B": res["results"][name] = "same as fixed"; continue
    out = {"music": {}, "tones": {}}
    for g, Tg in zip(gains, T):
        for d in HELD:
            x = clip(d)
            for o in OFFS:
                k = f"{g:g}|{d}|{o:g}"; y = rf(x * db(Tg + o)); out["music"][k] = feats2(y)
                if o == 0: out["music"][k]["lm_esr"] = lm_esr(y, real_wave(g, d))
        for f0 in TONE_F0:
            for lv in TONE_LV:
                y = rf(tone(f0, lv + Tg)) if False else rf(tone(f0, lv + Tg))
                h = harmonics(y, f0); h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9))); out["tones"][f"{g:g}|{int(f0)}|{lv:g}"] = h
    res["results"][name] = out
    print(amp, key, name, f"{time.time()-t0:.0f}s", flush=True)
(P4E / amp / f"eval_{key}.json").write_text(json.dumps(res))
