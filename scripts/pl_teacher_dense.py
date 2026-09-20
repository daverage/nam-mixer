"""Teacher-only analysis (no training, no NAM): the existing v3 C10 chain (all ten captures as anchors, fixed levels) rendered at the same cases, to test whether denser
anchors reduce the teacher's departure from the fixed-gain reference. Usage: pl_teacher_dense.py <amp> <pos>"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from pl_common import *
amp, g = sys.argv[1], float(sys.argv[2]); gains, levels, c = config(amp, "v3_C10")
res = {"amp": amp, "g": g, "cases": []}
for d in HELD:
    x = clip(d)
    for o in OFFS:
        x0 = (x * db(o)).astype(np.float32); real = render_aligned(amp, g, x0)
        tw, env, w = teacher(amp, (x0 * db(fixed_T(g))).astype(np.float32), gains, levels)
        act = env > env.max() - 30
        res["cases"].append({"di": d, "offset": o, "real": feats2(real), "teacher_C10": feats2(tw.astype(np.float32)), "mean_weight_active": w[:, act].mean(axis=1).tolist(), "gains": gains})
(PL / f"dense_{amp}_{sys.argv[2]}.json").write_text(json.dumps(res))
