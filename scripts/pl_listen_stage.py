"""Fixed-virtual-gain listening set, stage 1 (one amp). Renders REAL reference, v3 C3, Phase 4E B seeds 0/1 at a FIXED Input gain per setting for soft / normal / hard playing
and a soft->normal->hard sequence. No training, no new models. Usage: pl_listen_stage.py <amp> -> work/p4e/listening_fixed_gain/_stage/<amp>.json + audio"""
import json, os, random, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np, soundfile as sf
from pl_common import *
from single_nam_common import render_file_nam
amp = sys.argv[1]
OUT = P4E / "listening_fixed_gain" / "_stage" / amp; OUT.mkdir(parents=True, exist_ok=True)
CFG = {"jcm800": {"di": "moderate_brit", "settings": [2.0, 5.0, 8.0, 10.0]}, "vibrolux": {"di": "clean_mayer", "settings": [3.0, 5.0, 7.0, 10.0]}}[amp]
PICKS = {"soft": -12.0, "normal": 0.0, "hard": 6.0}
MODELS = ["v3_C3", "B_s0", "B_s1"]
rng = random.Random(f"fixedgain-listening-{amp}-20260921")
def T_of(key, g): return intended_T(amp, "B", [g])[0] if key.startswith("B") else fixed_T(g)
def window(x, secs, step=0.5):
    """Deterministic clip choice: the window with the widest spread of 250 ms block levels (p90-p10) among windows with >=80% active blocks."""
    n = int(secs * SR); best = None
    for s in np.arange(0, len(x) / SR - secs + 1e-9, step):
        seg = x[int(s * SR): int(s * SR) + n]; b = 10 * np.log10(np.mean(seg[: len(seg) // (SR // 4) * (SR // 4)].reshape(-1, SR // 4) ** 2, axis=1) + 1e-12)
        act = b > b.max() - 45
        if act.mean() < 0.8: continue
        sp = np.percentile(b, 90) - np.percentile(b, 10)
        if best is None or sp > best[0]: best = (sp, float(s))
    return best[1], best[0]
x_full = clip(CFG["di"]); s1, sp1 = window(x_full, 5.0); s2, sp2 = window(x_full, 4.0)
nams = {k: model_path(amp, k) for k in MODELS}
def render_all(x0, g):
    ref = render_aligned(amp, g, x0.astype(np.float32)); out = {"REFERENCE": ref}
    for k, (nam, c) in nams.items(): out[k] = render_file_nam(nam, (x0 * db(T_of(k, g))).astype(np.float32)) / c
    return out
def write(sid, waves):
    """waves: dict name->array. Writes native and level-matched versions with ONE common gain per stimulus and mode (relative levels preserved)."""
    PRE = 10 ** (4.5 / 20); waves = {k: v * PRE for k, v in waves.items()}   # fixed +4.5 dB makeup so quiet clips are audible; identical for every item and candidate
    ref = waves["REFERENCE"]; d = OUT / sid; d.mkdir(exist_ok=True); info = {"pre_gain_db": 4.5}
    for mode in ("native", "levelmatched"):
        w = {k: (v if mode == "native" or k == "REFERENCE" else v * np.sqrt(np.mean(ref[SR // 2:] ** 2) / max(np.mean(v[SR // 2:] ** 2), 1e-20))) for k, v in waves.items()}
        peak = max(float(np.max(np.abs(v))) for v in w.values()); gain = min(1.0, 0.95 / peak); info[mode] = {"common_gain": float(gain), "peak_before": float(peak)}
        for k, v in w.items(): sf.write(d / f"{k}__{mode}.wav", np.clip(v * gain, -1, 1).astype(np.float32), SR, subtype="PCM_16")
    return info
stim = []
for g in CFG["settings"]:
    for pk, off in PICKS.items():
        x0 = clip(CFG["di"])[int(s1 * SR): int((s1 + 5.0) * SR)] * db(off); w = render_all(x0, g); sid = f"{amp}_G{g:g}_{pk}"
        stim.append({"sid": sid, "kind": "single", "amp": amp, "virtual_gain": g, "pick": pk, "pick_offset_db": off, "di": CFG["di"], "window_start_s": float(s1), "duration_s": 5.0, "levels": write(sid, w),
                     "input_gain_db": {k: T_of(k, g) for k in MODELS}})
    segs = {}; gap = np.zeros(SR // 2, np.float32)
    for pk, off in PICKS.items():
        xs = clip(CFG["di"])[int(s2 * SR): int((s2 + 4.0) * SR)] * db(off); segs[pk] = render_all(xs, g)
    seq = {k: np.concatenate([segs["soft"][k], gap, segs["normal"][k], gap, segs["hard"][k]]) for k in segs["soft"]}; sid = f"{amp}_G{g:g}_sequence"
    stim.append({"sid": sid, "kind": "sequence", "amp": amp, "virtual_gain": g, "pick": "soft then normal then hard", "pick_offset_db": [-12.0, 0.0, 6.0], "di": CFG["di"], "window_start_s": float(s2), "duration_s": 13.0,
                 "levels": write(sid, seq), "input_gain_db": {k: T_of(k, g) for k in MODELS}})
    print(amp, "G%g" % g, "done", flush=True)
(OUT.parent / f"stage_{amp}.json").write_text(json.dumps({"amp": amp, "di": CFG["di"], "window_single": [s1, sp1], "window_sequence": [s2, sp2], "stimuli": stim}, indent=1, default=float))
