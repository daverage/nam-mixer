"""Phase 4C data: render the EXISTING v3 models (no retraining) across an input-gain sweep -> work/p4/<amp>/model_sweep_<n>.json.
T = total input change in dB applied to the native DI (= player input gain when the DI is at its native level). Usage: p4_model_sweep.py <amp>"""
import json
import os
import sys
import time

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np  # noqa: E402

from p4_common import FIT_DIS, HELD_OUT_DIS, REPO, SR, clip, db, feats, harmonics, load_json, music_extra, out_dir, tone  # noqa: E402
from single_nam_common import render_file_nam  # noqa: E402

amp = sys.argv[1]
D = out_dir(amp)
T_GRID = list(range(-48, 37, 3))
IO_LEVELS = list(range(-96, 34, 3))   # sine RMS dBFS at the model input
t0 = time.time()
for n in (10, 5, 3):
    ds = REPO / "work" / "cg" / amp / f"cg_{n}"
    nam = next(ds.glob("*/*.nam"))
    c = load_json(ds / "manifest.json")["output_scale_c"]
    rf = lambda x: render_file_nam(nam, x.astype(np.float32)) / c
    res = {"amp": amp, "n_captures": n, "model": str(nam.relative_to(REPO)), "gains": load_json(ds / "manifest.json")["gains"], "T": T_GRID, "music": {}, "io": {}}
    for d_ in FIT_DIS + HELD_OUT_DIS:
        x = clip(d_)
        res["music"][d_] = [{**feats(rf(x * db(T))), **music_extra(rf(x * db(T)))} if False else feats(rf(x * db(T))) for T in T_GRID]
    for f0 in (110.0, 440.0):
        rows = []
        for lv in IO_LEVELS:
            y = rf(tone(f0, lv))
            h = harmonics(y, f0); h["peak_db"] = float(20 * np.log10(max(np.max(np.abs(y[SR:])), 1e-9)))
            rows.append(h)
        res["io"][str(int(f0))] = {"levels": IO_LEVELS, "stats": rows}
    (D / f"model_sweep_{n}.json").write_text(json.dumps(res))
    print(amp, n, f"{time.time()-t0:.0f}s", flush=True)
