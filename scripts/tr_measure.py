"""Tonal refinement, step 1 (NO training): class-resolved power spectra of real fixed-gain captures vs the existing FC models (native NAM, no cab/EQ/CLO).
Usage: tr_measure.py <amp> <fit|held> -> work/tr/<amp>/spec_<set>.npz
Same conventions as fc_cases/fc_eval: real = verified-aligned capture render of x*db(offset); FC = render of x*db(offset)*db(T(g)) / c with T from the FROZEN FC anchor mapping.
Frames are classified from the REAL signal: attack (>= +4 dB over 43 ms earlier), decay (<= -4 dB), sustain (else); frames below max-55 dB and the first 0.5 s are dropped."""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from scipy.signal import stft
from fc_common import *
from single_nam_common import render_file_nam
amp, which = sys.argv[1], sys.argv[2]
FIT_DIS = ["clean_smooth", "high_metalcore", "high_thrash", "moderate_hotrod"]; FIT_OFFS = [-6.0, 0.0]
dis, offs = (HELD, OFFS) if which == "held" else (FIT_DIS, FIT_OFFS)
gains_all = [g for g in json.loads((P4E / amp / "real_ref.json").read_text())["gains"] if float(g).is_integer()]
Fg, Ta = FC_CFG[amp]
models = {k: fc_model_path(amp, k) for k in ("FC_s0", "FC_s1")}
NP, HOP = 2048, 512
def spec(y):
    y = y[SR // 2:].astype(np.float64); f, t, Z = stft(y, SR, window="hann", nperseg=NP, noverlap=NP - HOP, boundary=None, padded=False); return (np.abs(Z) ** 2).T
def classes(P):
    e = 10 * np.log10(P.sum(axis=1) + 1e-20); lag = 4; d = np.full_like(e, 0.0); d[lag:] = e[lag:] - e[:-lag]; ok = e > e.max() - 55; ok[:lag] = False
    return {"all": ok, "attack": ok & (d >= 4), "decay": ok & (d <= -4), "sustain": ok & (np.abs(d) < 4)}
keys, out = [], {}
for g in gains_all:
    T = T_of_position(amp, Fg, Ta, g)
    for d in dis:
        x = clip(d)
        for o in offs:
            x0 = (x * db(o)).astype(np.float32); real = render_aligned(amp, g, x0); sr_ = spec(real); cl = classes(sr_)
            sig = {"real": real}
            for k, (nam, c) in models.items(): sig[k] = (render_file_nam(nam, (x0 * db(T)).astype(np.float32)) / c).astype(np.float32)
            tag = f"{g:g}|{d}|{o:g}"; keys.append(tag)
            for k, y in sig.items():
                P = sr_ if k == "real" else spec(y)
                for cn, m in cl.items(): out[f"{tag}|{k}|{cn}"] = P[m].mean(axis=0).astype(np.float32) if m.any() else np.full(NP // 2 + 1, np.nan, np.float32)
                out[f"{tag}|{k}|rms"] = np.float32(20 * np.log10(np.sqrt(np.mean(y[SR // 2:].astype(np.float64) ** 2)) + 1e-9)); out[f"{tag}|{k}|peak"] = np.float32(np.max(np.abs(y)))
            for cn, m in cl.items(): out[f"{tag}|nframes|{cn}"] = np.int32(m.sum())
            out[f"{tag}|T"] = np.float32(T)
    print(amp, which, "G%g" % g, flush=True)
(REPO / "work" / "tr" / amp).mkdir(parents=True, exist_ok=True)
np.savez_compressed(REPO / "work" / "tr" / amp / f"spec_{which}.npz", keys=np.array(keys), **out)
