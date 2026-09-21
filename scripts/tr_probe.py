"""EQ probe audition for the JCM800 (no training): FC seed 0 through a small predefined set of audible post-NAM EQ shapes at G1, G5, G10, next to the real capture.
Level-matched to the real capture; with and without the repo test IR. Usage: tr_probe.py -> work/tr/probe/*.wav, work/tr/probe/manifest.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = "jcm800"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np, soundfile as sf
from fc_common import *
from single_nam_common import render_file_nam
import tr_eq as EQ
from hybrid.cab_ir import get_prepared_cab_ir, apply_cab_ir
PROBES = {"P1_hishelf_+1.5dB_4k": [{"type": "high_shelf", "f": 4000, "gain_db": 1.5, "s": 0.7}], "P2_hishelf_+3dB_4k": [{"type": "high_shelf", "f": 4000, "gain_db": 3.0, "s": 0.7}],
          "P3_lowshelf_+1.5dB_150": [{"type": "low_shelf", "f": 150, "gain_db": 1.5, "s": 0.7}], "P4_lowshelf_+1.5_150_and_hishelf_+1.5_4k": [{"type": "low_shelf", "f": 150, "gain_db": 1.5, "s": 0.7}, {"type": "high_shelf", "f": 4000, "gain_db": 1.5, "s": 0.7}],
          "P5_peak_+2dB_400_Q0.7": [{"type": "peak", "f": 400, "gain_db": 2.0, "q": 0.7}]}
amp = "jcm800"; Fg, Ta = FC_CFG[amp]; nam, c = fc_model_path(amp, "FC_s0"); x = clip("moderate_brit")
IR = REPO / "assets" / "nam_models" / "V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav"; prep = get_prepared_cab_ir(str(IR), SR)
out = REPO / "work" / "tr" / "probe"; out.mkdir(parents=True, exist_ok=True)
def act_rms(y):
    y = y[SR:]; e = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(4800) / 4800, "same")); m = e > e.max() * 0.01; return float(np.sqrt(np.mean(y[m].astype(np.float64) ** 2)))
sigs, files = {}, []
for g in (1, 5, 10):
    T = T_of_position(amp, Fg, Ta, g); real = render_aligned(amp, g, x); fc = (render_file_nam(nam, (x * db(T)).astype(np.float32)) / c).astype(np.float32)
    sigs[(g, "real")] = real; sigs[(g, "FC-noEQ")] = fc
    for n, p in PROBES.items(): sigs[(g, n)] = EQ.apply(EQ.build(p), fc)
for tag, f in (("IR", lambda y: apply_cab_ir(y.astype(np.float32), prep)), ("noIR", lambda y: y)):
    proc = {k: f(v) for k, v in sigs.items()}; sc = {}
    for (g, k), y in proc.items(): sc[(g, k)] = y * (act_rms(proc[(g, "real")]) / act_rms(y))
    common = min(1.0, 10 ** (-1 / 20) / max(np.max(np.abs(v)) for v in sc.values()))
    for (g, k), y in sc.items(): nm = f"jcm800_G{g}_{k}_levelmatched_{tag}.wav"; sf.write(out / nm, y * common, SR, subtype="PCM_24"); files.append(nm)
(out / "manifest.json").write_text(json.dumps({"di": "moderate_brit", "model": "FC seed 0", "probes": PROBES, "files": files, "ir": IR.name}, indent=1)); print(len(files), "files")
