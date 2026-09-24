"""Continuous Gain response profile (Phase 4B): measured level / tone / saturation / compression / dynamics / noise
series over the physical gain positions. Behaviour-preserving extraction of `scripts/p4_profile.py` (archived; no plots).

Dimensions stay separate; nothing here is a perceptual score. Captures flagged for a dimension (level / noise)
by the audit are quarantined from THAT dimension's interpolation estimate only.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .probe import FIT_DIS

EQ = ["sub", "low", "lowmid", "mid", "presence", "air"]


def build_profile(probes: dict[float, dict], audit: dict) -> dict:
    A = audit["captures"]
    gains = sorted(float(g) for g in probes)
    key = lambda g: f"{g:g}"
    mus = lambda g, f, off="0": float(np.mean([probes[g]["music"][f"{n}@{off}"][f] for n in FIT_DIS]))
    ton = lambda g, f0, lv, f: probes[g]["tones"][f"{f0}@{lv}"][f]
    quar = {"level": {g for g in gains if any(e.startswith("level:") for e in A[key(g)]["evidence"])},
            "noise": {g for g in gains if any("noise spike" in e for e in A[key(g)]["evidence"])}}
    series = {
        "music RMS @0 dB DI": ("level", "dBFS", lambda g: mus(g, "rms_db")),
        "music RMS @-12 dB DI": ("level", "dBFS", lambda g: mus(g, "rms_db", "-12")),
        "music RMS @+6 dB DI": ("level", "dBFS", lambda g: mus(g, "rms_db", "6")),
        "music peak @0": ("level", "dBFS", lambda g: mus(g, "peak_db")),
        **{f"EQ {b}": ("tone", "dB rel. total", (lambda b: lambda g: mus(g, f"eq_{b}_db"))(b)) for b in EQ},
        "tilt (presence-low)": ("tone", "dB", lambda g: mus(g, "tilt_db")),
        "HF >3 kHz": ("saturation", "dB rel. total", lambda g: mus(g, "hf3k_db")),
        "THD 440Hz @-30": ("saturation", "dB", lambda g: ton(g, 440, -30, "thd_db")),
        "THD 440Hz @-18": ("saturation", "dB", lambda g: ton(g, 440, -18, "thd_db")),
        "THD 440Hz @-6": ("saturation", "dB", lambda g: ton(g, 440, -6, "thd_db")),
        "H2 440Hz @-18": ("saturation", "dB", lambda g: ton(g, 440, -18, "h2_db")),
        "H3 440Hz @-18": ("saturation", "dB", lambda g: ton(g, 440, -18, "h3_db")),
        "crest @0": ("saturation", "dB", lambda g: mus(g, "crest_db")),
        "IO gain -54->-30": ("compression", "dB", lambda g: ton(g, 440, -30, "out_rms_db") - ton(g, 440, -54, "out_rms_db") - 24.0),
        "IO gain -30->0": ("compression", "dB", lambda g: ton(g, 440, 0, "out_rms_db") - ton(g, 440, -30, "out_rms_db") - 30.0),
        "dyn range @0": ("dynamics", "dB", lambda g: mus(g, "dyn_range_db")),
        "dyn range @+6": ("dynamics", "dB", lambda g: mus(g, "dyn_range_db", "6")),
        "noise floor (silence)": ("noise", "dBFS", lambda g: probes[g]["silence"]["rms_db"]),
    }
    prof = {"gains": gains, "quarantine": {k: sorted(v) for k, v in quar.items()}, "series": {}}
    ints = [g for g in gains if g == int(g)]
    for name, (dim, unit, fn) in series.items():
        y = np.array([fn(g) for g in gains])
        bad = quar.get(dim, set())
        ok = np.array([g not in bad for g in gains])
        gi = np.array(gains)[ok]
        yi = y[ok]
        loo = [abs(yi[k] - PchipInterpolator(np.delete(gi, k), np.delete(yi, k))(gi[k])) for k in range(1, len(gi) - 1)] if len(gi) >= 3 else []
        rng = float(np.ptp(yi)) or 1.0
        entry = {"dimension": dim, "unit": unit, "values": y.tolist(), "reliable": ok.tolist(),
                 "loo_median_abs_err": float(np.median(loo)) if loo else None, "range": rng, "rate": None}
        half = [g for g in gains if g != int(g)]
        if half and len(ints) > 2:
            gi2 = np.array([g for g in ints if g not in bad])
            yi2 = np.array([y[gains.index(g)] for g in gi2])
            pr = PchipInterpolator(gi2, yi2)
            errs = [abs(y[gains.index(g)] - float(pr(g))) for g in half if g not in bad]
            if errs:
                entry["half_step_true_err_median"] = float(np.median(errs))
                entry["half_step_true_err_max"] = float(np.max(errs))
        dg = np.diff(gi)
        dy = np.abs(np.diff(yi)) / rng / dg
        entry["rate_intervals"] = [[float(gi[k]), float(gi[k + 1]), float(dy[k])] for k in range(len(dy))]
        prof["series"][name] = entry
    regions = {}
    grid = ints if len(ints) >= 2 else gains
    for dim in ("level", "tone", "saturation", "compression", "dynamics"):
        ser = [v for v in prof["series"].values() if v["dimension"] == dim]
        rates = []
        for a, b in zip(grid[:-1], grid[1:]):
            vals = []
            for v in ser:
                gi = [g for g, o in zip(gains, v["reliable"]) if o]
                yy = [y for y, o in zip(v["values"], v["reliable"]) if o]
                f = PchipInterpolator(gi, yy)
                vals.append(abs(float(f(b)) - float(f(a))) / v["range"])
            rates.append(float(np.mean(vals)))
        med = float(np.median(rates)) if rates else 0.0
        regions[dim] = {"intervals": [[a, b] for a, b in zip(grid[:-1], grid[1:])], "rate": rates,
                        "fast": [[a, b] for (a, b), r in zip(zip(grid[:-1], grid[1:]), rates) if r > 1.5 * med],
                        "stable": [[a, b] for (a, b), r in zip(zip(grid[:-1], grid[1:]), rates) if r < 0.5 * med]}
    prof["regions"] = regions
    pl = []
    lr, sr_, tr = regions["level"]["rate"], regions["saturation"]["rate"], regions["tone"]["rate"]
    for k, (a, b) in enumerate(regions["level"]["intervals"]):
        if lr[k] < 0.5 * np.median(lr) and (sr_[k] > np.median(sr_) or tr[k] > np.median(tr)):
            pl.append([a, b])
    prof["level_plateau_but_character_changing"] = pl
    return prof
