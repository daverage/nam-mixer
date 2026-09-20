"""Teacher-only weight-band test, cases: same positions/DIs/offsets as the playability diagnostic. For B and A anchor sets, render each anchor capture ONCE
per case and blend with each plateau fraction. Usage: pl_band_cases.py <amp> <pos> -> work/p4e/playability/band_cases_<amp>_<pos>.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from pl_common import *
from pl_band_common import *
from hybrid.validation import compute_esr_metrics
amp, g = sys.argv[1], float(sys.argv[2])
CFG = {"B": config(amp, "B_s0"), "A": config(amp, "A_s0")}
T_OF = {"B": intended_T(amp, "B", [g])[0], "A": fixed_T(g)}
def esr(a, b, warm=SR // 2):
    n = min(len(a), len(b)); p, q = a[warm:n].astype(np.float64), b[warm:n].astype(np.float64)
    p = p * np.sqrt(np.mean(q ** 2) / max(np.mean(p ** 2), 1e-20)); return float(compute_esr_metrics(p, q)["raw_esr"])
res = {"amp": amp, "g": g, "phis": PHIS, "cases": []}
for d in HELD:
    x = clip(d)
    for o in OFFS:
        x0 = (x * db(o)).astype(np.float32); real = render_aligned(amp, g, x0); rf = feats2(real); rh = hf10k_db(real, SR)
        case = {"di": d, "offset": o, "real": rf, "real_hf10k": rh, "cfg": {}}
        for cfg, (gains, levels, c) in CFG.items():
            x_in = (x0 * db(T_OF[cfg])).astype(np.float32); env = bounded_causal_envelope_db(x_in, SR)
            ch = GainChain(tuple(gains), tuple(levels), REF)
            renders = [render_aligned(amp, gg, (x_in * db(ch.input_scale_db(k))).astype(np.float32)) for k, gg in enumerate(gains)]
            n = min(len(env), *(len(r) for r in renders)); act = env[:n] > env[:n].max() - 30; per = {}
            for phi in PHIS:
                w = band_weights(env[:n], levels, phi); y = sum(w[k] * renders[k][:n] for k in range(len(gains))).astype(np.float32)
                per[str(phi)] = {"feats": feats2(y), "esr": esr(y, real), "hf10k": hf10k_db(y, SR),
                                 "mean_weight_active": w[:, act].mean(axis=1).tolist(), "nearest_anchor_weight": float(w[int(np.argmin(np.abs(np.array(gains) - g))), act].mean()),
                                 "cross": crossfade_stats(w, env[:n], SR)}
            case["cfg"][cfg] = {"T": T_OF[cfg], "gains": gains, "levels": levels, "phi": per}
        res["cases"].append(case)
(PL / f"band_cases_{amp}_{sys.argv[2]}.json").write_text(json.dumps(res)); print(amp, g, "done")
