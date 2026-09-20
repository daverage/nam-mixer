"""For ONE amp and a list of positions: at FIXED virtual gain (intended mapping) render REAL capture, TEACHER (envelope-driven target, analysis only) and STUDENTS
(exported Phase 4E / v3 NAMs) for each held-out DI at each musical-input offset. Records features, pairwise level-matched ESR, teacher blend weights,
and within-recording block levels (offset 0). Usage: pl_cases.py <amp> <pos,pos,...> -> work/p4e/playability/cases_<amp>_<pos>.json"""
import json, os, sys, time
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from pl_common import *
from hybrid.validation import compute_esr_metrics
from single_nam_common import render_file_nam
amp = sys.argv[1]; positions = [float(p) for p in sys.argv[2].split(",")]
STUDENTS = {"B_s0": "B", "B_s1": "B", "A_s0": "A", "A_s1": "A", "v3_C3": "C3"}
TEACH = {"B": "B_s0", "A": "A_s0", "C3": "v3_C3"}
cfgs = {t: config(amp, k) for t, k in TEACH.items()}
models = {k: model_path(amp, k) for k in STUDENTS}
def esr(a, b, warm=SR // 2):
    n = min(len(a), len(b)); p, q = a[warm:n].astype(np.float64), b[warm:n].astype(np.float64)
    p = p * np.sqrt(np.mean(q ** 2) / max(np.mean(p ** 2), 1e-20)); return float(compute_esr_metrics(p, q)["raw_esr"])
BL = SR // 4
def blocks(y): n = len(y) // BL; return (10 * np.log10(np.mean(y[: n * BL].astype(np.float64).reshape(n, BL) ** 2, axis=1) + 1e-12)).tolist()
res = {"amp": amp, "cases": []}; t0 = time.time()
for g in positions:
    T_of = {"B": intended_T(amp, "B", [g])[0], "A": fixed_T(g), "C3": fixed_T(g)}
    for d in HELD:
        x = clip(d)
        for o in OFFS:
            x0 = (x * db(o)).astype(np.float32)                      # musical input (DI at the offset), before the player's Input gain
            real = render_aligned(amp, g, x0)
            case = {"g": g, "di": d, "offset": o, "T": T_of, "real": feats2(real), "teacher": {}, "student": {}, "weights": {}, "esr": {}}
            wave = {"real": real}
            for t, (gains, levels, c) in cfgs.items():
                x_in = (x0 * db(T_of[t])).astype(np.float32)            # what the player delivers to the NAM = musical input * Input gain
                tw, env, w = teacher(amp, x_in, gains, levels)
                act = env > env.max() - 30
                case["teacher"][t] = feats2(tw.astype(np.float32)); wave["T_" + t] = tw
                case["weights"][t] = {"gains": gains, "levels_db": levels, "mean_weight_active": w[:, act].mean(axis=1).tolist(),
                                      "env_active_p10_p50_p90": [float(v) for v in np.percentile(env[act], [10, 50, 90])],
                                      "nearest_anchor_weight_active": float(w[int(np.argmin(np.abs(np.array(gains) - g))), act].mean())}
                case["esr"]["teacher_vs_real_" + t] = esr(tw, real)
            for k, t in STUDENTS.items():
                nam, c = models[k]; x_in = (x0 * db(T_of[t])).astype(np.float32); y = render_file_nam(nam, x_in) / c
                case["student"][k] = feats2(y.astype(np.float32)); wave[k] = y
                case["esr"][f"{k}_vs_real"] = esr(y, real); case["esr"][f"{k}_vs_teacher"] = esr(y, wave["T_" + t])
            if o == 0.0:
                case["blocks"] = {"di_in_db": blocks(x0), **{k: blocks(v) for k, v in wave.items() if k in ("real", "T_B", "B_s0", "T_A", "A_s0", "T_C3", "v3_C3")}}
            res["cases"].append(case)
    print(amp, "G%g" % g, f"{time.time()-t0:.0f}s", flush=True)
(PL / f"cases_{amp}_{'_'.join(sys.argv[2].split(','))}.json").write_text(json.dumps(res))
