"""Tonal refinement, step 3: focused audition package + time-domain peak/headroom check. No training. Usage: tr_audition.py -> work/tr/audition/<amp>/*.wav, work/tr/audition/manifest.json
Real = verified-aligned capture render; FC = existing FC seed 0 (no seed preference reported) at the frozen intended Input gain, / c; EQ candidates = temporary post-NAM biquads (tr_analyze.py).
Every file: same DI, same test cabinet IR (repo IR, NOT the user's IR) when 'IR' in the name; 'native' = raw levels (one common gain per amp keeps peaks <= -1 dBFS), 'lm' = level-matched to the real capture."""
import json, os, sys
import numpy as np, soundfile as sf
amp_env = None
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
POS = {"jcm800": [1, 2, 5, 8, 10], "vibrolux": [1, 3, 5, 7, 10]}; DI = {"jcm800": "moderate_brit", "vibrolux": "clean_mayer"}
res = {}; A = json.loads((__import__("pathlib").Path(__file__).resolve().parent.parent / "work" / "tr" / "analysis.json").read_text())
for amp in ("jcm800", "vibrolux"):
    os.environ["SINGLE_NAM_AMP"] = amp
    import importlib, single_nam_common; importlib.reload(single_nam_common)
    import fc_common; importlib.reload(fc_common)
    from fc_common import *
    from single_nam_common import render_file_nam
    import tr_eq as EQ
    from hybrid.cab_ir import get_prepared_cab_ir, apply_cab_ir
    IR = REPO / "assets" / "nam_models" / "V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav"
    prep = get_prepared_cab_ir(str(IR), SR) if True else None
    Fg, Ta = FC_CFG[amp]; models = {k: fc_model_path(amp, k) for k in ("FC_s0", "FC_s1")}
    sosB = EQ.build(A[amp]["eq"]["B"]["params"]); sosC = EQ.build(A[amp]["eq"]["C"]["params"]) if amp == "vibrolux" else None
    out = REPO / "work" / "tr" / "audition" / amp; out.mkdir(parents=True, exist_ok=True); x = clip(DI[amp]); items = {}; peaks = {}
    def act_rms(y):
        y = y[SR:]; e = np.sqrt(np.convolve(y.astype(np.float64) ** 2, np.ones(4800) / 4800, "same")); m = e > e.max() * 10 ** (-40 / 20); return float(np.sqrt(np.mean(y[m].astype(np.float64) ** 2)))
    for g in POS[amp]:
        T = T_of_position(amp, Fg, Ta, g); real = render_aligned(amp, g, x)
        fc = {k: (render_file_nam(nam, (x * db(T)).astype(np.float32)) / c).astype(np.float32) for k, (nam, c) in models.items()}
        sig = {"real": real, "FC-noEQ": fc["FC_s0"], "FC-EQ-B": EQ.apply(sosB, fc["FC_s0"])}
        if sosC is not None: sig["FC-EQ-C"] = EQ.apply(sosC, fc["FC_s0"])
        if amp == "vibrolux" and g in (1, 3): sig["FC-noEQ-seed1"] = fc["FC_s1"]
        for k, y in sig.items(): items[(g, k)] = y
        # time-domain peak check at the hardest musical input (+6 dB DI)
        xh = (x * db(6.0)).astype(np.float32); fh = render_file_nam(models["FC_s0"][0], (xh * db(T)).astype(np.float32)) / models["FC_s0"][1]
        peaks[f"G{g}"] = {"real_peak_dbfs": float(20 * np.log10(np.max(np.abs(render_aligned(amp, g, xh))))), "FC_peak_dbfs": float(20 * np.log10(np.max(np.abs(fh)))), "FC_B_peak_dbfs": float(20 * np.log10(np.max(np.abs(EQ.apply(sosB, fh.astype(np.float32)))))),
                          **({"FC_C_peak_dbfs": float(20 * np.log10(np.max(np.abs(EQ.apply(sosC, fh.astype(np.float32))))))} if sosC is not None else {})}
    def with_ir(y): return apply_cab_ir(y.astype(np.float32), prep)
    cab = {kk: with_ir(v) for kk, v in items.items()}
    def write(store, tag, ir_tag, subset=None):
        # one common gain per amp/tag keeps peaks <= -1 dBFS; lm = per-file rms match to the real capture of the same position
        pk = max(np.max(np.abs(v)) for v in store.values()); common = min(1.0, 10 ** (-1 / 20) / pk); names = []
        for (g, k), y in store.items():
            if subset and g not in subset: continue
            yn = y * common; nm = f"{amp}_G{g}_normal_{k}_{tag}_{ir_tag}.wav"; sf.write(out / nm, yn, SR, subtype="PCM_24"); names.append(nm)
            if tag == "native":
                r = store[(g, "real")]; ylm = y * (act_rms(r) / act_rms(y)); pk2 = np.max(np.abs(ylm)); ylm = ylm * min(1.0, 10 ** (-1 / 20) / pk2) if pk2 > 10 ** (-1 / 20) else ylm
                nm2 = f"{amp}_G{g}_normal_{k}_levelmatched_{ir_tag}.wav"; sf.write(out / nm2, ylm, SR, subtype="PCM_24"); names.append(nm2)
        return float(common), names
    c1, n1 = write(cab, "native", "IR"); c2, n2 = write(items, "native", "noIR", subset=[10])
    res[amp] = {"di": DI[amp], "positions": POS[amp], "native_common_gain_IR": c1, "native_common_gain_noIR": c2, "files": n1 + n2, "time_domain_peaks_hard_+6dB_no_cab_dbfs": peaks, "ir": IR.name}
    print(amp, len(n1 + n2), "files", flush=True)
(REPO / "work" / "tr" / "audition" / "manifest.json").write_text(json.dumps(res, indent=1, default=float))
