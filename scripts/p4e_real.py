"""Real-reference features on the held-out DIs (4A-verified alignment). -> work/p4e/<amp>/real_ref.json. Usage: p4e_real.py <amp>"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from p4e_common import *
amp = sys.argv[1]
AUD = json.loads((REPO / "work" / "p4" / amp / "audit.json").read_text())["captures"]
gains = sorted(float(k) for k in AUD)
def aligned(g, x):
    y = render(capture(g), x, SR); c = AUD[f"{g:g}"]["correction"]; sh = -int(c["samples"]) if c and "samples" in c else 0
    return np.concatenate([y[sh:], np.zeros(sh, y.dtype)]) if sh > 0 else (np.concatenate([np.zeros(-sh, y.dtype), y[:sh]]) if sh < 0 else y)
res = {"amp": amp, "gains": gains, "music": {}, "tones": {}}
for g in gains:
    for d in HELD:
        x = clip(d)
        for o in OFFS:
            res["music"][f"{g:g}|{d}|{o:g}"] = feats2(aligned(g, (x * db(o)).astype(np.float32)))
    for f0 in TONE_F0:
        for lv in TONE_LV:
            y = aligned(g, tone(f0, lv)); h = harmonics(y, f0); h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9)))
            res["tones"][f"{g:g}|{int(f0)}|{lv:g}"] = h
    print(amp, "G%g" % g, flush=True)
(P4E / amp / "real_ref.json").write_text(json.dumps(res))
