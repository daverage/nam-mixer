"""Evaluate one model at ITS OWN intended mapping against the real captures (direct renders, same protocol and schema as p4e_eval.py). Usage: fc_eval.py <amp> <FC_s0|FC_s1|B_s0|B_s1|v3_C3>
-> work/p4e/final/<amp>/eval_<key>.json   (results under 'intended')"""
import json, os, sys, time
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from fc_common import *
from hybrid.validation import compute_esr_metrics
from single_nam_common import render_file_nam
amp, key = sys.argv[1], sys.argv[2]
REAL = json.loads((P4E / amp / "real_ref.json").read_text()); gains = REAL["gains"]; nam, c = fc_model_path(amp, key); rf = lambda x: render_file_nam(nam, x.astype(np.float32)) / c
T = fc_intended_T(amp, key, gains); AUD = audit(amp); cache = {}
def real_wave(g, d):
    if (g, d) not in cache: cache[(g, d)] = render_aligned(amp, g, clip(d))
    return cache[(g, d)]
def lm_esr(a, b, warm=SR // 2):
    n = min(len(a), len(b)); p, q = a[warm:n].astype(np.float64), b[warm:n].astype(np.float64); p = p * np.sqrt(np.mean(q ** 2) / max(np.mean(p ** 2), 1e-20)); return float(compute_esr_metrics(p, q)["raw_esr"])
res = {"amp": amp, "model": key, "nam": str(nam.relative_to(REPO)), "gains": gains, "mappings": {"intended": T}, "regions": {}, "results": {"intended": {"music": {}, "tones": {}}}}
out = res["results"]["intended"]; t0 = time.time()
for g, Tg in zip(gains, T):
    for d in HELD:
        x = clip(d)
        for o in OFFS:
            y = rf(x * db(Tg + o)); k = f"{g:g}|{d}|{o:g}"; out["music"][k] = feats2(y)
            if o == 0: out["music"][k]["lm_esr"] = lm_esr(y, real_wave(g, d))
    for f0 in TONE_F0:
        for lv in TONE_LV:
            y = rf(tone(f0, lv + Tg)); h = harmonics(y, f0); h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9))); out["tones"][f"{g:g}|{int(f0)}|{lv:g}"] = h
    print(amp, key, "G%g" % g, f"{time.time()-t0:.0f}s", flush=True)
(FCDIR / amp / f"eval_{key}.json").write_text(json.dumps(res, default=float))
