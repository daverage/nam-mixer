"""Evaluate an exported single-NAM model against real captures + G5 baselines, on held-out DIs.

Signal path for the new model is exactly a NAM player's: DI * frozen-law input gain -> exported .nam (NAMCore),
then ONE global constant 1/c (the training-target scale recorded in the manifest). No per-gain fitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from single_nam_common import (ALL_GAINS, CALIBRATED_G5_DB, INT_GAINS, OUT, SR, TEST_DIS, db, load_di, load_law,
                               metrics, render_capture, render_file_nam, save_json)

SECONDS = 20
LEVELS_DB = [-12.0, 0.0, 6.0]
LAG_8_5 = 690  # click-latency excess of the G8.5 capture (701 vs ~11), analysis-only alignment


def new_render(nam: Path, c: float, di: np.ndarray, law_db: float) -> np.ndarray:
    return render_file_nam(nam, di * db(law_db)) / c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nam", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--trained-gains", type=float, nargs="+", default=INT_GAINS)
    ap.add_argument("--wavs", action="store_true")
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    c = man["output_scale_c"]
    law = load_law()
    res = {"label": a.label, "trained_gains": a.trained_gains, "per_di": {}, "levels": {}}

    for name in TEST_DIS:
        di = load_di(name)[: SECONDS * SR]
        rows = {}
        for g in ALL_GAINS:
            ref = render_capture(g, di)
            row = {"trained": g in a.trained_gains,
                   "new": metrics(new_render(a.nam, c, di, law[g]), ref),
                   "g5_same_law": metrics(render_capture(5.0, di, law[g]), ref)}
            if g in CALIBRATED_G5_DB:
                row["g5_calibrated_old"] = metrics(render_capture(5.0, di, CALIBRATED_G5_DB[g]), ref)
            if g == 8.5:
                n = new_render(a.nam, c, di, law[g])
                row["new_lag_aligned"] = metrics(n[:-LAG_8_5], ref[LAG_8_5:])
            rows[str(g)] = row
            if a.wavs and name == "moderate_brit" and g in (3.0, 6.0, 9.0):
                w = OUT / "listening" / a.label
                w.mkdir(parents=True, exist_ok=True)
                sf.write(w / f"g{g:g}_real.wav", ref[: 10 * SR], SR, subtype="FLOAT")
                sf.write(w / f"g{g:g}_new.wav", new_render(a.nam, c, di, law[g])[: 10 * SR], SR, subtype="FLOAT")
                sf.write(w / f"g{g:g}_g5_same_law.wav", render_capture(5.0, di, law[g])[: 10 * SR], SR, subtype="FLOAT")
        res["per_di"][name] = rows

    di = load_di("moderate_brit")[: SECONDS * SR]
    for lv in LEVELS_DB:
        rows = {}
        for g in INT_GAINS:
            ref = render_capture(g, di * db(lv))
            rows[str(g)] = {"new": metrics(new_render(a.nam, c, di * db(lv), law[g]), ref),
                            "g5_same_law": metrics(render_capture(5.0, di * db(lv), law[g]), ref)}
        res["levels"][str(lv)] = rows
    save_json(f"eval_{a.label}.json", res)

    for name, rows in res["per_di"].items():
        print(f"\n== {name}: raw ESR / level dB / spec corr (new | G5 same law | G5 old-calibrated)")
        for g, r in rows.items():
            cal = r.get("g5_calibrated_old")
            print(f"G{float(g):4.1f}{'*' if r['trained'] else ' '} {r['new']['raw_esr']:7.4f} {r['new']['level_delta_db']:+5.1f} "
                  f"{r['new']['spectral_corr']:.3f} | {r['g5_same_law']['raw_esr']:7.4f} {r['g5_same_law']['level_delta_db']:+5.1f} | "
                  + (f"{cal['raw_esr']:7.4f} {cal['level_delta_db']:+5.1f}" if cal else "   n/a"))
    print("\n== input-level test (moderate_brit), integer gains: raw ESR new | G5 same law")
    for lv, rows in res["levels"].items():
        print(f"DI {lv:>5} dB: " + " ".join(f"{r['new']['raw_esr']:.3f}/{r['g5_same_law']['raw_esr']:.3f}" for r in rows.values()))


if __name__ == "__main__":
    main()
