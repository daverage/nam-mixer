"""Continuous Input-gain sweep through a model at nominal musical input (3 held-out DIs). Usage: fc_sweep.py <amp> <key> -> work/p4e/final/<amp>/sweep_<key>.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from fc_common import *
from single_nam_common import render_file_nam
amp, key = sys.argv[1], sys.argv[2]; nam, c = fc_model_path(amp, key); TS = list(range(-24, 25, 2)); res = {"amp": amp, "model": key, "T": TS, "feats": {}}
for d in HELD:
    x = clip(d); res["feats"][d] = []
    for T in TS:
        y = render_file_nam(nam, (x * db(T)).astype(np.float32)) / c; f = feats2(y.astype(np.float32)); res["feats"][d].append({k: f[k] for k in ("rms_db", "hf3k_db", "crest_db", "dyn_range_db", "peak_db")})
(FCDIR / amp / f"sweep_{key}.json").write_text(json.dumps(res, default=float)); print("done", amp, key)
