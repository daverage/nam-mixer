"""FC students vs FC teacher vs real at fixed Input gain (same positions/DIs/offsets as the playability diagnostic). Usage: fc_cases.py <amp> <pos> -> work/p4e/final/<amp>/cases_fc_<pos>.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from fc_common import *
from hybrid.validation import compute_esr_metrics
from single_nam_common import render_file_nam
amp, g = sys.argv[1], float(sys.argv[2]); gains, Tanch = FC_CFG[amp]; levels = [t + REF for t in Tanch]; Tg = T_of_position(amp, gains, Tanch, g)
models = {k: fc_model_path(amp, k) for k in ("FC_s0", "FC_s1") if list((FCDIR / amp / "FC_bundle").glob(f"{amp}_FC_s{k[-1]}/*.nam"))}; real = json.loads((P4E / amp / "real_ref.json").read_text())
def esr(a, b, warm=SR // 2):
    n = min(len(a), len(b)); p, q = a[warm:n].astype(np.float64), b[warm:n].astype(np.float64); p = p * np.sqrt(np.mean(q ** 2) / max(np.mean(p ** 2), 1e-20)); return float(compute_esr_metrics(p, q)["raw_esr"])
res = {"amp": amp, "g": g, "T": Tg, "cases": []}
for d in HELD:
    x = clip(d)
    for o in OFFS:
        x0 = (x * db(o)).astype(np.float32); x_in = (x0 * db(Tg)).astype(np.float32); tw, env, w = teacher(amp, x_in, gains, levels)
        case = {"di": d, "offset": o, "T": Tg, "real": real["music"][f"{g:g}|{d}|{o:g}"], "teacher": feats2(tw.astype(np.float32)), "student": {}, "esr": {}}
        for k, (nam, c) in models.items():
            y = render_file_nam(nam, x_in) / c; case["student"][k] = feats2(y.astype(np.float32)); case["esr"][f"{k}_vs_teacher"] = esr(y, tw)
        res["cases"].append(case)
(FCDIR / amp / f"cases_fc_{sys.argv[2]}.json").write_text(json.dumps(res, default=float)); print(amp, g, "done")
