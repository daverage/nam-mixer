"""Continuous Gain v3 comparison data for ONE amp: real captures vs the new single NAMs (10/5/3) vs the G5-pushed
baseline, at the frozen designed input gains. Writes report_<amp>.csv (long format, absolute features for EVERY
source, so tone/EQ/saturation/dynamics/level can be compared directly) + capture_qa_<amp>.json.
Rows: kind=music (held-out DIs at level offsets) and kind=tone (sine harmonic-distortion / compression probes)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from single_nam_common import ALL_GAINS, INT_GAINS, SR, TEST_DIS, capture, capture_lag, crest_db, db, hf_ratio_db, load_di, render, render_capture, render_file_nam
from cg_build import REF, level_of
from cg_eval import BANDS, band_db, lm
from hybrid.envelope import bounded_causal_envelope_db
from hybrid.audio_metrics import spectral_magnitude_correlation
from hybrid.validation import compute_esr_metrics

EQ_BANDS = {"sub": (20, 100), "low": (100, 250), "lowmid": (250, 800), "mid": (800, 2500), "presence": (2500, 6000), "air": (6000, 16000)}


def feats(y: np.ndarray) -> dict:
    y = y[SR // 2:].astype(np.float64)
    rms = float(np.sqrt(np.mean(y ** 2)))
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    tot = max(s.sum(), 1e-20)
    out = {"rms_db": 20 * np.log10(max(rms, 1e-9)), "peak_db": 20 * np.log10(max(np.max(np.abs(y)), 1e-9)),
           "crest_db": crest_db(y), "hf3k_db": hf_ratio_db(y)}
    for k, (lo, hi) in EQ_BANDS.items():
        out[f"eq_{k}_db"] = 10 * np.log10(max(s[(f >= lo) & (f < hi)].sum(), 1e-20) / tot)
    out["tilt_db"] = out["eq_presence_db"] - out["eq_low_db"]
    e = bounded_causal_envelope_db(y.astype(np.float32), SR)
    a = e[e > np.max(e) - 50]
    out["dyn_range_db"] = float(np.percentile(a, 95) - np.percentile(a, 10)) if len(a) else 0.0
    return out


def sim(cand: np.ndarray, ref: np.ndarray) -> dict:
    n = min(len(cand), len(ref)); w = SR // 2
    c, r = cand[w:n].astype(np.float64), ref[w:n].astype(np.float64)
    cl = c * np.sqrt(np.mean(r ** 2) / max(np.mean(c ** 2), 1e-20))
    return {"lm_esr": float(compute_esr_metrics(cl, r)["raw_esr"]), "raw_esr": float(compute_esr_metrics(c, r)["raw_esr"]),
            "band_err_db": float(np.mean(np.abs(band_db(cand) - band_db(ref)))),
            "spec_corr": float(spectral_magnitude_correlation(c, r))}


def tone(f0: float, level_db: float) -> np.ndarray:
    t = np.arange(3 * SR) / SR
    x = np.sin(2 * np.pi * f0 * t) * db(level_db) * np.sqrt(2)
    x[: SR // 20] *= np.linspace(0, 1, SR // 20)
    return x.astype(np.float32)


def harmonics(y: np.ndarray, f0: float) -> dict:
    y = y[SR:].astype(np.float64)  # last 2 s: integer number of cycles
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    frq = np.fft.rfftfreq(len(y), 1 / SR)
    h = [sp[(frq > k * f0 - 6) & (frq < k * f0 + 6)].max() for k in range(1, 11)]
    thd = np.sqrt(sum(v ** 2 for v in h[1:])) / max(h[0], 1e-12)
    return {"thd_db": 20 * np.log10(max(thd, 1e-9)), "h2_db": 20 * np.log10(max(h[1] / h[0], 1e-9)),
            "h3_db": 20 * np.log10(max(h[2] / h[0], 1e-9)), "out_rms_db": 20 * np.log10(max(np.sqrt(np.mean(y ** 2)), 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--offsets", type=float, nargs="+", default=[-12.0, -6.0, 0.0, 6.0])
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--configs", type=int, nargs="+", default=[10, 5, 3])
    a = ap.parse_args()
    models = {}
    for n in a.configs:
        p = list((a.root / f"cg_{n}").glob("*/*.nam"))
        if p:
            models[f"new{n}"] = (p[0], json.loads((a.root / f"cg_{n}" / "manifest.json").read_text())["output_scale_c"])
    print("models:", list(models))
    gains = ALL_GAINS
    dis = {n: load_di(n)[: a.seconds * SR] for n in TEST_DIS}
    rows = []

    def add(kind, source, g, di, off, ft, extra=None):
        rows.append({"amp": a.amp, "kind": kind, "source": source, "gain": g, "di": di, "offset_db": off,
                     "designated_gain_db": level_of(g) - REF, **ft, **(extra or {})})

    for g in gains:
        gd = level_of(g) - REF
        offs = a.offsets if g == int(g) else [0.0]
        for n, x in dis.items():
            for off in offs:
                xo = (x * db(off)).astype(np.float32)
                real = render_capture(g, xo)
                add("music", "real", g, n, off, feats(real))
                outs = {"g5": render_capture(5.0, xo, gd)}
                for k, (mp, c) in models.items():
                    outs[k] = render_file_nam(mp, (xo * db(gd)).astype(np.float32)) / c
                for k, y in outs.items():
                    add("music", k, g, n, off, feats(y), sim(y, real))
        if g == int(g):
            for f0 in (110.0, 440.0):
                for off in (-12.0, 0.0, 12.0):
                    x = tone(f0, REF + off)
                    ys = {"real": render_capture(g, x), "g5": render_capture(5.0, x, gd)}
                    for k, (mp, c) in models.items():
                        ys[k] = render_file_nam(mp, (x * db(gd)).astype(np.float32)) / c
                    for k, y in ys.items():
                        add("tone", k, g, f"sine{int(f0)}", off, harmonics(y, f0))
        print("gain", g, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("amp", "kind", "source", "gain", "di", "offset_db", "designated_gain_db"), k))
    with open(a.root / f"report_{a.amp}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    click = np.zeros(SR, np.float32); click[SR // 2] = 0.05
    qa = {"click_latency_samples": {str(g): int(np.argmax(np.abs(render(capture(g), click, SR))) - SR // 2) for g in gains},
          "applied_lag_vs_g5": {str(g): capture_lag(g) for g in gains},
          "silence_noise_db": {str(g): float(20 * np.log10(max(np.sqrt(np.mean(render(capture(g), np.zeros(SR, np.float32), SR).astype(np.float64) ** 2)), 1e-9))) for g in INT_GAINS}}
    real = [r for r in rows if r["kind"] == "music" and r["source"] == "real" and r["offset_db"] == 0.0 and r["gain"] == int(r["gain"])]
    for m in ("rms_db", "hf3k_db", "crest_db"):
        v = [np.mean([r[m] for r in real if r["gain"] == g]) for g in INT_GAINS]
        qa[f"real_{m}_by_gain"] = [round(float(x), 2) for x in v]
        qa[f"real_{m}_reversals"] = int(sum(np.sign(np.diff(v)) [:-1] * np.sign(np.diff(v))[1:] < 0))
    (a.root / f"capture_qa_{a.amp}.json").write_text(json.dumps(qa, indent=2))
    print("wrote", a.root / f"report_{a.amp}.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
