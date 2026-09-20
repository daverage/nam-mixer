"""Model sweep for a NEW model (same format as work/p4/<amp>/model_sweep_N.json) -> work/p4e/<amp>/sweep_<key>.json. Usage: p4e_sweep.py <amp> <key>"""
import json, os, sys, time
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from p4e_common import *
from p4_common import FIT_DIS, HELD_OUT_DIS
from single_nam_common import render_file_nam
amp, key = sys.argv[1], sys.argv[2]
nam, c = model_path(amp, key); rf = lambda x: render_file_nam(nam, x.astype(np.float32)) / c
T_GRID = list(range(-48, 37, 3)); IO = list(range(-96, 34, 3))
res = {"amp": amp, "model": key, "T": T_GRID, "music": {}, "io": {}}
for d in FIT_DIS + HELD_OUT_DIS:
    x = clip(d); res["music"][d] = [feats(rf(x * db(T))) for T in T_GRID]
for f0 in (110.0, 440.0):
    rows = []
    for lv in IO:
        y = rf(tone(f0, lv)); h = harmonics(y, f0); h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9))); rows.append(h)
    res["io"][str(int(f0))] = {"levels": IO, "stats": rows}
(P4E / amp / f"sweep_{key}.json").write_text(json.dumps(res)); print(amp, key, "sweep done")
