"""Verify, from the actual code and data, the signal-path claims the playability diagnostic depends on. -> work/p4e/playability/path_verification.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = "vibrolux"
import numpy as np, soundfile as sf
from pl_common import *
from single_nam_common import render_file_nam
out = {}
amp = "vibrolux"
# 1. The envelope is a function of the waveform that reaches the NAM: gain in dB shifts it one-for-one; DI offset and player gain are indistinguishable.
x = clip("moderate_brit")
e0 = bounded_causal_envelope_db(x, SR); act = e0 > e0.max() - 60
res = {}
for g in (-12.0, -6.0, 6.0, 14.0):
    e1 = bounded_causal_envelope_db((x * db(g)).astype(np.float32), SR)
    d = (e1 - e0)[act]; res[f"{g:+g} dB"] = {"median_shift_db": float(np.median(d)), "max_abs_deviation_from_gain_db": float(np.max(np.abs(d - g)))}
out["envelope_shift_vs_gain"] = res
a = bounded_causal_envelope_db((x * db(-6.0) * db(-8.4)).astype(np.float32), SR); b = bounded_causal_envelope_db((x * db(-14.4)).astype(np.float32), SR)
out["player_gain_and_musical_offset_commute"] = {"max_abs_envelope_difference_db (offset -6 then gain -8.4 vs offset -14.4)": float(np.max(np.abs(a - b)))}
# 2. Teacher reconstruction from the frozen Phase 4E bundle (proves teacher() == the builder's signal path, including scaling)
for cfg in ("A", "B"):
    d = P4E / amp / f"{cfg}_bundle"; m = json.loads((d / "manifest.json").read_text())
    X, _ = sf.read(d / "input.wav", dtype="float32"); Y, _ = sf.read(d / "target.wav", dtype="float32")
    n = 12 * SR; xin = X[:n]
    T, env, w = teacher(amp, xin, m["gains"], m["levels_db"])
    diff = np.abs(T * m["output_scale_c"] - Y[:n])
    out[f"teacher_reconstruction_{cfg}"] = {"seconds_compared": 12, "output_scale_c": m["output_scale_c"], "max_abs_diff_vs_frozen_target": float(diff.max()), "rms_target": float(np.sqrt(np.mean(Y[:n] ** 2)))}
# 3. Evaluation path == ordinary player path: model input = DI * offset * player Input gain; reference capture sees DI * offset only.
nam, c = model_path(amp, "B_s0"); Tg = intended_T(amp, "B", [3.0])[0]
xo = (clip("clean_mayer")[: 6 * SR] * db(6.0)).astype(np.float32)
y1 = render_file_nam(nam, (xo * db(Tg)).astype(np.float32)); y2 = render_file_nam(nam, (clip("clean_mayer")[: 6 * SR] * db(6.0 + Tg)).astype(np.float32))
out["evaluation_path"] = {"vibrolux_G3_intended_input_gain_db": Tg, "model input = (DI*offset)*gain vs DI*(offset+gain): max abs diff": float(np.max(np.abs(y1 - y2))),
                          "reference capture input": "DI*offset only (no player gain); model output divided by the single global constant c", "c_B_bundle": c}
(PL / "path_verification.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
