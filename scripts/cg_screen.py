"""Capture pre-screen (prototype): flag captures that break the smooth gain curve of an amp sweep.
Uses only the real-capture rows of report_<amp>.csv (music at reference level) + capture_qa_<amp>.json.
Per capture: (1) timing vs the set, (2) leave-one-out curve residual on level-independent features (EQ bands, HF, crest),
(3) noise floor. Status OK / WARN / EXCLUDE with a user-facing reason. Curve = amp gain structure: smooth, no violent peaks."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "work" / "cg"
FEATS = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db", "hf3k_db", "crest_db"]
LAT_TOL = 20          # samples vs the set median
RES_WARN, RES_EXCL = 1.5, 3.0   # LOO residual relative to the local slope of the curve


def load(amp):
    rows = [r for r in csv.DictReader(open(ROOT / amp / f"report_{amp}.csv"))
            if r["kind"] == "music" and r["source"] == "real" and float(r["offset_db"]) == 0.0]
    g = sorted({float(r["gain"]) for r in rows})
    m = np.array([[np.mean([float(r[f]) for r in rows if float(r["gain"]) == gg]) for f in FEATS] for gg in g])
    return g, m


def screen(amp):
    g, m = load(amp)
    qa = json.loads((ROOT / amp / f"capture_qa_{amp}.json").read_text())
    lat = np.array([qa["click_latency_samples"][str(x)] for x in g], float)
    med_lat = float(np.median(lat))
    # feature scale: robust per-feature spread of the adjacent-capture step, so every feature counts comparably
    step = np.abs(np.diff(m, axis=0))
    scale = np.maximum(np.median(step, axis=0), 0.15)
    z = m / scale
    n = len(g)
    res = np.zeros(n)
    local = np.zeros(n)
    for i in range(n):
        if 0 < i < n - 1:
            t = (g[i] - g[i - 1]) / (g[i + 1] - g[i - 1])
            res[i] = float(np.sqrt(np.mean((z[i] - (z[i - 1] * (1 - t) + z[i + 1] * t)) ** 2)))
            local[i] = float(np.linalg.norm(z[i + 1] - z[i - 1]) / np.sqrt(len(FEATS)) / 2)
        else:  # ends anchor the curve: extrapolate from the two nearest inner captures, half weight
            j, k = (1, 2) if i == 0 else (n - 2, n - 3)
            pred = z[j] + (z[j] - z[k]) * (g[i] - g[j]) / (g[j] - g[k])
            res[i] = 0.5 * float(np.sqrt(np.mean((z[i] - pred) ** 2)))
            local[i] = float(np.linalg.norm(z[j] - z[k]) / np.sqrt(len(FEATS)))
    typical = float(np.median(local[1:-1])) or 1.0
    ratio = res / np.maximum(local, 0.5 * typical)   # deviation relative to the curve's local slope
    out = []
    for i, gg in enumerate(g):
        why, status = [], "OK"
        dl = lat[i] - med_lat
        if abs(dl) > LAT_TOL:
            why.append(f"timing offset {dl:+.0f} samples vs the set (correctable by shifting; only some pipelines apply it automatically)")
            status = "NOTE"
        r = float(ratio[i])
        end = i in (0, n - 1)
        if r > RES_EXCL and not end:
            why.append(f"off the gain curve ({r:.1f}x typical deviation)")
            status = "EXCLUDE"
        elif r > RES_WARN and (not end or r > RES_EXCL):
            why.append(f"bends the gain curve ({r:.1f}x typical deviation)")
            status = max(status, "WARN", key=["OK", "NOTE", "WARN", "EXCLUDE"].index)
        nf = qa["silence_noise_db"].get(str(gg)) if gg == int(gg) else None
        if nf is not None and nf > -50:
            why.append(f"noisy capture (silence renders at {nf:.0f} dBFS)")
            status = max(status, "WARN", key=["OK", "NOTE", "WARN", "EXCLUDE"].index)
        out.append((gg, status, r, "; ".join(why)))
    return out


if __name__ == "__main__":
    for amp in sys.argv[1:]:
        print(f"\n== {amp}")
        for gg, st, r, why in screen(amp):
            if st != "OK":
                print(f"  G{gg:g}: {st:7s} (curve dev {r:.1f}x)  {why}")
        print("  all others OK")
