"""Render the TEACHER (envelope-driven training target, analysis only) and the dry/musical DI files for the shortlist items of ONE amp, using the same windows, offsets, Input gains
and makeup/common gains as pl_listen_stage.py. Usage: pl_listen_teacher_stage.py <amp>  -> work/p4e/listening_fixed_gain/_stage/<amp>/<stage_id>/{TEACHER_B,TEACHER_v3C3,DI_dry_original,DI_musical_input}__*.wav + teacher_info.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np, soundfile as sf
from pl_common import *
import pl_listen_short as S
amp = sys.argv[1]; LG = P4E / "listening_fixed_gain"
stage = json.loads((LG / "_stage" / f"stage_{amp}.json").read_text()); di_name = stage["di"]
PRE = 10 ** (4.5 / 20); GAP = np.zeros(SR // 2, np.float32)
CFG = {"B": config(amp, "B_s0"), "v3C3": config(amp, "v3_C3")}
def T_of(t, g): return intended_T(amp, "B", [g])[0] if t == "B" else fixed_T(g)
def teach(x0, g, t):
    gains, levels, c = CFG[t]; tw, env, w = teacher(amp, (x0 * db(T_of(t, g))).astype(np.float32), gains, levels); act = env > env.max() - 30
    return tw.astype(np.float32), w[:, act].mean(axis=1), gains
def wr(path, y): sf.write(path, np.clip(y, -1, 1).astype(np.float32), SR, subtype="PCM_16")
full = clip(di_name); info = {}
for (a, g, pk) in S.SHORT:
    if a != amp: continue
    st = next(s for s in stage["stimuli"] if s["virtual_gain"] == g and (s["pick"] == pk or (pk.startswith("soft then") and s["kind"] == "sequence"))); d = LG / "_stage" / amp / st["sid"]
    seq = st["kind"] == "sequence"; s0 = st["window_start_s"]
    if seq:
        segs = [full[int(s0 * SR): int((s0 + 4.0) * SR)] * db(o) for o in (-12.0, 0.0, 6.0)]; musical = np.concatenate([segs[0], GAP, segs[1], GAP, segs[2]]).astype(np.float32); dry = full[int(s0 * SR): int((s0 + 4.0) * SR)]
        parts = {t: [teach(x, g, t) for x in segs] for t in CFG}; teach_w = {t: np.concatenate([p[0] for p in parts[t]][:1] + [GAP, parts[t][1][0], GAP, parts[t][2][0]]) for t in CFG}
        wmean = {t: np.mean([p[1] for p in parts[t]], axis=0).tolist() for t in CFG}; gl = {t: parts[t][0][2] for t in CFG}
        ref = np.concatenate([render_aligned(amp, g, segs[0].astype(np.float32)), GAP, render_aligned(amp, g, segs[1].astype(np.float32)), GAP, render_aligned(amp, g, segs[2].astype(np.float32))])
    else:
        x0 = (full[int(s0 * SR): int((s0 + 5.0) * SR)] * db(st["pick_offset_db"])).astype(np.float32); musical = x0; dry = full[int(s0 * SR): int((s0 + 5.0) * SR)]
        r = {t: teach(x0, g, t) for t in CFG}; teach_w = {t: r[t][0] for t in CFG}; wmean = {t: r[t][1].tolist() for t in CFG}; gl = {t: r[t][2] for t in CFG}; ref = render_aligned(amp, g, x0)
    for mode in ("native", "levelmatched"):
        cg = st["levels"][mode]["common_gain"]
        for name, y in (("DI_dry_original", dry), ("DI_musical_input", musical)):
            if mode == "native": wr(d / f"{name}__native.wav", y * PRE * cg)
        for t in CFG:
            w = teach_w[t] * PRE
            if mode == "levelmatched": w = w * np.sqrt(np.mean((ref * PRE)[SR // 2:] ** 2) / max(np.mean(w[SR // 2:] ** 2), 1e-20))
            wr(d / f"TEACHER_{t}__{mode}.wav", w * cg)
    info[st["sid"]] = {"weights_mean_active": wmean, "anchors": gl, "input_gain_db": {"B": T_of("B", g), "v3C3": T_of("v3C3", g)}}
    print(amp, st["sid"], "done", flush=True)
(LG / "_stage" / f"teacher_info_{amp}.json").write_text(json.dumps(info, indent=1, default=float))
