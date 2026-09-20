"""Evaluate a Continuous Gain v3 model against the REAL captures on held-out DIs (frozen designed mapping).
Level-matched diagnostics are primary (captures are normalised); raw level delta reported for information.
Baseline: the real G5 capture driven by the SAME designed input gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from single_nam_common import ALL_GAINS, SR, TEST_DIS, capture_lag, db, load_di, metrics, render_capture, render_file_nam
from cg_build import LO, REF, STEP, level_of

BANDS = 1000 * 2 ** (np.arange(-6, 4.5, 1 / 3))


def band_db(x):
    x = x[SR // 2:].astype(np.float64)
    s = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    b = np.array([s[(f >= lo) & (f < hi)].sum() for lo, hi in zip(BANDS[:-1], BANDS[1:])])
    return 10 * np.log10(np.maximum(b, 1e-20) / max(b.sum(), 1e-20))


def lm(cand, ref, warm=SR // 2):
    n = min(len(cand), len(ref))
    c, r = cand[warm:n].astype(np.float64), ref[warm:n].astype(np.float64)
    return c * np.sqrt(np.mean(r ** 2) / max(np.mean(c ** 2), 1e-20))


def score(cand, ref):
    m = metrics(cand, ref)
    cl = lm(cand, ref)
    mm = metrics(cl.astype(np.float32), ref[SR // 2:len(cl) + SR // 2].astype(np.float32), warm=0)
    return {"lm_esr": mm["raw_esr"], "raw_esr": m["raw_esr"], "level_db": m["level_delta_db"], "spec_corr": m["spectral_corr"],
            "band_db": float(np.mean(np.abs(band_db(cand) - band_db(ref)))), "crest_d": m["crest_delta_db"], "hf_d": m["hf_delta_db"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--listen", type=Path)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    c, trained = man["output_scale_c"], set(man["gains"])
    dis = {n: load_di(n)[: a.seconds * SR] for n in TEST_DIS}
    rows = []
    for g in ALL_GAINS:
        gd = level_of(g) - REF
        acc = {"new": [], "g5": []}
        for n, x in dis.items():
            ref = render_capture(g, x)
            new = render_file_nam(a.model, (x * db(gd)).astype(np.float32)) / c
            g5 = render_capture(5.0, x, gd)
            acc["new"].append(score(new, ref)); acc["g5"].append(score(g5, ref))
            if a.listen and g in (1.0, 3.0, 5.0, 7.0, 10.0) and n == "moderate_brit":
                a.listen.mkdir(parents=True, exist_ok=True)
                for tag, y in (("real", ref), ("new", new), ("g5", g5)):
                    sf.write(a.listen / f"{a.label}_G{g:g}_{tag}.wav", y.astype(np.float32), SR)
                    sf.write(a.listen / f"{a.label}_G{g:g}_{tag}_levelmatched.wav", (y / max(np.sqrt(np.mean(y**2)),1e-9) * np.sqrt(np.mean(ref**2))).astype(np.float32), SR)
        rows.append({"gain": g, "input_db": gd, "trained": g in trained,
                     **{f"{k}_{m}": float(np.mean([r[m] for r in v])) for k, v in acc.items() for m in v[0]}})
    (a.manifest.parent / f"eval_{a.label}.json").write_text(json.dumps(rows, indent=2))
    print(f"{'G':>4} {'in dB':>6} | new: lmESR bandΔ crest hf lvl | G5same: lmESR bandΔ")
    for r in rows:
        print(f"{r['gain']:4g}{'*' if r['trained'] else ' '}{r['input_db']:+6.1f} | {r['new_lm_esr']:.4f} {r['new_band_db']:5.2f} {r['new_crest_d']:+5.2f} {r['new_hf_d']:+5.2f} {r['new_level_db']:+5.1f} | {r['g5_lm_esr']:.4f} {r['g5_band_db']:5.2f}")
    ints = [r for r in rows if r["gain"] == int(r["gain"])]
    for k in ("new", "g5"):
        print(k, "mean lm_esr int", round(np.mean([r[f"{k}_lm_esr"] for r in ints]), 4), "band", round(np.mean([r[f"{k}_band_db"] for r in ints]), 2))


if __name__ == "__main__":
    main()
