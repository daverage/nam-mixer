"""Teacher-only evaluation of ONE candidate capture set + mapping for one amp: at every physical position (intended Input gain from the candidate's own mapping) render the
envelope-driven teacher for each held-out DI and playing offset, compare with the real capture, and sweep the Input gain continuously. Usage:
  fc_teacher_eval.py <amp> <name> <gains comma list> [t_lo] [t_hi]  -> work/p4e/final/teacher_<amp>_<name>.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from fc_common import *
amp, name = sys.argv[1], sys.argv[2]; gains = [float(x) for x in sys.argv[3].split(",")]
t_lo = float(sys.argv[4]) if len(sys.argv) > 4 else -20.0; t_hi = float(sys.argv[5]) if len(sys.argv) > 5 else 14.0
gains, Tanch = anchor_levels(amp, gains, t_lo, t_hi); levels = [t + REF for t in Tanch]
real = json.loads((P4E / amp / "real_ref.json").read_text()); positions = [g for g in real["gains"] if g == int(g)]
F = ("rms_db", "hf3k_db", "crest_db", "dyn_range_db", "tilt_db", *EQ)
out = {"amp": amp, "name": name, "gains": gains, "anchor_T": Tanch, "positions": positions, "cases": {}, "sweep": {}}
for g in positions:
    Tg = T_of_position(amp, gains, Tanch, g)
    for d in HELD:
        x = clip(d)
        for o in OFFS:
            x_in = (x * db(o + Tg)).astype(np.float32); tw, env, w = teacher(amp, x_in, gains, levels); f = feats2(tw.astype(np.float32)); r = real["music"][f"{g:g}|{d}|{o:g}"]
            out["cases"][f"{g:g}|{d}|{o:g}"] = {"T": Tg, "err": {k: f[k] - r[k] for k in F}, "eq_abs": float(np.mean([abs(f[b] - r[b]) for b in EQ]))}
    print(name, "G%g" % g, flush=True)
TS = list(range(int(np.floor(t_lo)), int(np.ceil(t_hi)) + 1, 2)); out["sweep"]["T"] = TS; out["sweep"]["feats"] = {}
for d in HELD:
    x = clip(d); rows = []
    for T in TS:
        tw, env, w = teacher(amp, (x * db(T)).astype(np.float32), gains, levels); f = feats2(tw.astype(np.float32)); rows.append({k: f[k] for k in ("rms_db", "hf3k_db", "crest_db", "dyn_range_db")})
    out["sweep"]["feats"][d] = rows
(FCDIR / f"teacher_{amp}_{name}.json").write_text(json.dumps(out, default=float))
print("done", amp, name, "anchors", dict(zip(gains, [round(t, 1) for t in Tanch])))
