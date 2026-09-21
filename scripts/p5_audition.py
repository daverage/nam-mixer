"""Short whole-amp audition per amp (no training). Files under work/p4e/final/audition/<amp>/ (through the repo test IR unless 'noIR'; that is NOT your IR):
 low / mid / top position: real capture vs FC vs middle-capture baseline (native + level-matched); a continuous Input-gain glide (-20 -> +14 dB) for FC and the baseline;
 the real-capture staircase vs the FC staircase over the selected anchors; soft-to-hard (DI level ramp -24 -> +8 dB) at the middle anchor's Input gain, real vs FC vs baseline.
Usage: p5_audition.py <amp>"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np, soundfile as sf
from fc_common import *
from single_nam_common import render_file_nam, load_di
from hybrid.cab_ir import get_prepared_cab_ir, apply_cab_ir
amp = sys.argv[1]; sel = FC_CFG[amp][0]; Ta = FC_CFG[amp][1]; DI = "clean_mayer" if amp in ("twin", "supersonic", "vibrolux") else "moderate_brit"
IR = REPO / "assets" / "nam_models" / "V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav"; prep = get_prepared_cab_ir(str(IR), SR)
out = REPO / "work" / "p4e" / "final" / "audition" / amp; out.mkdir(parents=True, exist_ok=True); files = []; m = {}
nam_fc, c_fc = fc_model_path(amp, "FC_s0"); nam_b, _ = fc_model_path(amp, "BASE"); mid_pos = sel[len(sel) // 2]; Tb = json.loads((FCDIR / amp / "baseline_T.json").read_text())["T_by_position"]
x = clip(DI)
def rms(y):
    y = y[SR:]; e = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(4800) / 4800, "same")); k = e > e.max() * 0.01; return float(np.sqrt(np.mean(y[k].astype(np.float64) ** 2)))
ir = lambda y: apply_cab_ir(y.astype(np.float32), prep)
store = {}
def put(name, pair): store[name] = (pair, None)
for tag, g in (("low", min(sel)), ("mid", mid_pos), ("top", max(sel))):
    Tf = T_of_position(amp, *FC_CFG[amp], g); real = render_aligned(amp, g, x); fc = (render_file_nam(nam_fc, (x * db(Tf)).astype(np.float32)) / c_fc).astype(np.float32); b = render_file_nam(nam_b, (x * db(float(Tb[f"{g:g}"]))).astype(np.float32)).astype(np.float32)
    m[tag] = {"position": g, "fc_input_gain_db": Tf, "baseline_capture": base_capture(amp), "baseline_input_gain_db": float(Tb[f"{g:g}"])}
    for k, y in (("real", real), ("FC", fc), ("baseline", b)): put(f"{amp}_{tag}_G{g:g}_{k}", (ir(y), ir(real)))
    if tag == "top":
        for k, y in (("real", real), ("FC", fc), ("baseline", b)): put(f"{amp}_{tag}_G{g:g}_{k}_noIR", (y, real))
# continuous glide (input gain ramp in dB, causal) over the DI looped to 40 s
long_x = np.tile(load_di(DI), 4)[: 40 * SR]; ramp = np.linspace(-20, 14, len(long_x))
xg = (long_x * db(0) * (10 ** (ramp / 20))).astype(np.float32)
put(f"{amp}_sweep_-20_to_+14dB_FC", (ir((render_file_nam(nam_fc, xg) / c_fc).astype(np.float32)), None)); put(f"{amp}_sweep_-20_to_+14dB_baseline_G{base_capture(amp):g}", (ir(render_file_nam(nam_b, xg).astype(np.float32)), None))
# staircase: each selected anchor position for 8 s: real capture vs FC at its anchor gain
seg = 8 * SR; st_r, st_f = [], []
for g, Tf in zip(sel, Ta):
    xs = np.tile(load_di(DI), 2)[:seg].astype(np.float32); st_r.append(render_aligned(amp, g, xs)); st_f.append((render_file_nam(nam_fc, (xs * db(Tf)).astype(np.float32)) / c_fc).astype(np.float32))
put(f"{amp}_staircase_real_anchors_" + "-".join(f"G{g:g}" for g in sel), (ir(np.concatenate(st_r)), None)); put(f"{amp}_staircase_FC_" + "-".join(f"{t:+.0f}dB" for t in Ta), (ir(np.concatenate(st_f)), None))
# soft-to-hard at the middle anchor's Input gain
Tm = T_of_position(amp, *FC_CFG[amp], mid_pos); xs = np.tile(load_di(DI), 3)[: 30 * SR].astype(np.float32); lv = np.linspace(-24, 8, len(xs)); xh = (xs * 10 ** (lv / 20)).astype(np.float32)
for k, y in (("real", render_aligned(amp, mid_pos, xh)), ("FC", (render_file_nam(nam_fc, (xh * db(Tm)).astype(np.float32)) / c_fc).astype(np.float32)), ("baseline", render_file_nam(nam_b, (xh * db(float(Tb[f"{mid_pos:g}"]))).astype(np.float32)).astype(np.float32))): put(f"{amp}_soft_to_hard_at_G{mid_pos:g}_FC{Tm:+.1f}dB_{k}", (ir(y), None))
m["soft_to_hard"] = {"position": mid_pos, "fc_input_gain_db": Tm, "di_level_ramp_db": [-24, 8]}
def get(v): return v[0][0]
pk = max(np.max(np.abs(get(v))) for v in store.values()); common = min(1.0, 10 ** (-1 / 20) / pk)
for name, v in store.items():
    y = get(v); ref = v[0][1]
    sf.write(out / f"{name}_native.wav", y * common, SR, subtype="PCM_24"); files.append(f"{name}_native.wav")
    if ref is not None:
        yl = y * (rms(ref) / rms(y)); g2 = min(1.0, 10 ** (-1 / 20) / np.max(np.abs(yl))); sf.write(out / f"{name}_levelmatched.wav", yl * g2, SR, subtype="PCM_24"); files.append(f"{name}_levelmatched.wav")
(out / "manifest.json").write_text(json.dumps({"amp": amp, "di": DI, "ir": IR.name, "note": "repo test IR, NOT the user's IR; 'noIR' files have none", "native_common_gain": float(common), "positions": m, "files": files}, indent=1, default=float)); print(amp, len(files), "files")
