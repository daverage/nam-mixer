"""Teacher-only weight-band test, progression: continuous Input-gain sweep at nominal musical input. Usage: pl_band_sweep.py <amp> <cfg B|A> -> work/p4e/playability/band_sweep_<amp>_<cfg>.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from pl_common import *
from pl_band_common import *
amp, cfg = sys.argv[1], sys.argv[2]; gains, levels, c = config(amp, f"{cfg}_s0"); ch = GainChain(tuple(gains), tuple(levels), REF)
TS = list(range(-24, 17, 2)); res = {"amp": amp, "cfg": cfg, "T": TS, "phis": PHIS, "gains": gains, "levels": levels, "data": {}}
for d in HELD:
    x = clip(d); res["data"][d] = {str(phi): [] for phi in PHIS}
    for T in TS:
        x_in = (x * db(T)).astype(np.float32); env = bounded_causal_envelope_db(x_in, SR)
        renders = [render_aligned(amp, gg, (x_in * db(ch.input_scale_db(k))).astype(np.float32)) for k, gg in enumerate(gains)]
        n = min(len(env), *(len(r) for r in renders))
        for phi in PHIS:
            w = band_weights(env[:n], levels, phi); y = sum(w[k] * renders[k][:n] for k in range(len(gains))).astype(np.float32)
            f = feats2(y); res["data"][d][str(phi)].append({k: f[k] for k in ("rms_db", "hf3k_db", "crest_db", "dyn_range_db", "tilt_db", "rise_p95_db", *EQ)})
    print(amp, cfg, d, "done", flush=True)
(PL / f"band_sweep_{amp}_{cfg}.json").write_text(json.dumps(res))
