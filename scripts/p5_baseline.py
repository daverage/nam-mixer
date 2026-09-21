"""Single-middle-capture Input-gain baseline (an UPPER BOUND for 'just turn the input of one capture'): for every real position, choose ONE Input gain on the middle capture (fc_common.base_capture)
on a 2 dB grid (-24..+24) that best matches the real position's tone/saturation/dynamics on the FIT DIs only (level excluded, captures are level-normalised). Frozen for all held-out use.
Usage: p5_baseline.py <amp> -> work/p4e/final/<amp>/baseline_T.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from fc_common import *
from single_nam_common import render_file_nam
amp = sys.argv[1]; FIT = ["clean_smooth", "moderate_hotrod"]; grid = list(range(-24, 25, 2)); nam, _ = fc_model_path(amp, "BASE"); mid = base_capture(amp)
real = json.loads((P4E / amp / "real_ref.json").read_text()); gains = real["gains"]
KEYS = EQ + ["hf3k_db", "crest_db", "dyn_range_db"]
cand = {d: {T: feats2(render_file_nam(nam, (clip(d) * db(T)).astype(np.float32)).astype(np.float32)) for T in grid} for d in FIT}
res = {"amp": amp, "middle_capture": mid, "grid_db": grid, "T_by_position": {}, "fit_DIs": FIT, "objective": "mean abs error over 6 EQ bands + HF>3k + crest + dyn range (level excluded), summed over fit DIs"}
for g in gains:
    rf = {d: feats2(render_aligned(amp, g, clip(d))) for d in FIT}
    cost = {T: sum(np.mean([abs(cand[d][T][k] - rf[d][k]) for k in KEYS]) for d in FIT) for T in grid}; res["T_by_position"][f"{g:g}"] = float(min(cost, key=cost.get))
    res.setdefault("cost", {})[f"{g:g}"] = float(min(cost.values()))
(FCDIR / amp / "baseline_T.json").write_text(json.dumps(res, indent=1)); print(amp, "middle capture G%g" % mid, res["T_by_position"])
