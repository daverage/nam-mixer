"""Supplementary (teacher- and NAM-independent): the intrinsic gain/level ambiguity between adjacent anchors, measured on the REAL captures.
Two situations deliver the SAME waveform to a standard NAM: capture a played at offset o+D and capture b played at offset o, where D = T_b - T_a is the gap between their
virtual-gain settings. Any single NAM output must sit between what the two real amps would do. Usage: pl_ambiguity.py <amp> -> work/p4e/playability/ambiguity_<amp>.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from pl_common import *
amp = sys.argv[1]; res = {"amp": amp, "pairs": []}
GS = {"jcm800": [1.0, 2.0, 4.0, 10.0], "vibrolux": [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]}[amp]
for rule in ("B", "A"):
    Ts = intended_T(amp, "B", GS) if rule == "B" else [fixed_T(g) for g in GS]
    for (ga, Ta), (gb, Tb) in zip(zip(GS, Ts), zip(GS[1:], Ts[1:])):
        D = Tb - Ta
        for o in (-12.0, -6.0, 0.0, 6.0):
            fa, fb = [], []
            for d in HELD:
                x = clip(d)
                fa.append(feats2(render_aligned(amp, ga, (x * db(o + D)).astype(np.float32)))); fb.append(feats2(render_aligned(amp, gb, (x * db(o)).astype(np.float32))))
            m = lambda L, f: float(np.mean([v[f] for v in L]))
            res["pairs"].append({"rule": rule, "a": ga, "b": gb, "gap_db": D, "offset_b": o, **{k: m(fa, f) - m(fb, f) for k, f in (("level", "rms_db"), ("HF", "hf3k_db"), ("crest", "crest_db"), ("dyn", "dyn_range_db"))},
                                  "eq_abs": float(np.mean([np.mean([abs(np.mean([v[b_] for v in fa]) - np.mean([v[b_] for v in fb])) for b_ in EQ])]))})
    print(amp, rule, "done", flush=True)
(PL / f"ambiguity_{amp}.json").write_text(json.dumps(res))
