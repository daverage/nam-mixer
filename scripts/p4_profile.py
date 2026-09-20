"""Phase 4B: amp-specific response profile from the probe bank -> work/p4/<amp>/profile.json + profile.png.
Dimensions kept separate: level, tone/EQ, saturation, compression, dynamics, noise. Measured points are markers, estimates dashed.
Captures flagged SUSPECT for a dimension (level / noise) are quarantined from THAT dimension's estimate only. Usage: p4_profile.py <amp>"""
import json
import os
import sys

os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from p4_common import FIT_DIS, load_json, out_dir  # noqa: E402

amp = sys.argv[1]
D = out_dir(amp)
P = load_json(D / "captures.json")["captures"]
A = load_json(D / "audit.json")["captures"]
gains = sorted(float(k) for k in P)
key = lambda g: f"{g:g}"
EQ = ["sub", "low", "lowmid", "mid", "presence", "air"]
mus = lambda g, f, off="0": float(np.mean([P[key(g)]["probe"]["music"][f"{n}@{off}"][f] for n in FIT_DIS]))
ton = lambda g, f0, lv, f: P[key(g)]["probe"]["tones"][f"{f0}@{lv}"][f]

quar = {"level": {g for g in gains if any(e.startswith("level:") for e in A[key(g)]["evidence"])},
        "noise": {g for g in gains if any("noise spike" in e for e in A[key(g)]["evidence"])}}
SERIES = {  # name: (dimension, unit, fn(g))
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
    "noise floor (silence)": ("noise", "dBFS", lambda g: P[key(g)]["probe"]["silence"]["rms_db"]),
}
prof = {"amp": amp, "gains": gains, "quarantine": {k: sorted(v) for k, v in quar.items()}, "series": {}}
ints = [g for g in gains if g == int(g)]
for name, (dim, unit, fn) in SERIES.items():
    y = np.array([fn(g) for g in gains])
    bad = quar.get(dim, set())
    ok = np.array([g not in bad for g in gains])
    # uncertainty of the interpolation: leave-one-out residual over interior reliable points; on JCM800 also true half-step error
    gi = np.array(gains)[ok]; yi = y[ok]
    loo = [abs(yi[k] - PchipInterpolator(np.delete(gi, k), np.delete(yi, k))(gi[k])) for k in range(1, len(gi) - 1)]
    rng = float(np.ptp(yi)) or 1.0
    entry = {"dimension": dim, "unit": unit, "values": y.tolist(), "reliable": ok.tolist(), "loo_median_abs_err": float(np.median(loo)) if loo else None,
             "range": rng, "rate": None}
    half = [g for g in gains if g != int(g)]
    if half:
        gi2 = np.array([g for g in ints if g not in bad]); yi2 = np.array([y[gains.index(g)] for g in gi2])
        pr = PchipInterpolator(gi2, yi2)
        errs = [abs(y[gains.index(g)] - float(pr(g))) for g in half if g not in bad]
        entry["half_step_true_err_median"] = float(np.median(errs))
        entry["half_step_true_err_max"] = float(np.max(errs))
    # local rate of change between adjacent reliable captures, relative to the series' own range (per unit knob)
    dg = np.diff(gi); dy = np.abs(np.diff(yi)) / rng / dg
    entry["rate_intervals"] = [[float(gi[k]), float(gi[k + 1]), float(dy[k])] for k in range(len(dy))]
    prof["series"][name] = entry
# regions where important characteristics change quickly / are stable, per dimension (mean normalised rate across the dimension's series)
regions = {}
for dim in ("level", "tone", "saturation", "compression", "dynamics"):
    ser = [v for v in prof["series"].values() if v["dimension"] == dim]
    grid = ints if not any(g != int(g) for g in gains) else ints
    rates = []
    for a, b in zip(grid[:-1], grid[1:]):
        vals = []
        for v in ser:
            gi = [g for g, o in zip(gains, v["reliable"]) if o]; yy = [y for y, o in zip(v["values"], v["reliable"]) if o]
            f = PchipInterpolator(gi, yy)
            vals.append(abs(float(f(b)) - float(f(a))) / v["range"])
        rates.append(float(np.mean(vals)))
    med = float(np.median(rates))
    regions[dim] = {"intervals": [[a, b] for a, b in zip(grid[:-1], grid[1:])], "rate": rates,
                    "fast": [[a, b] for (a, b), r in zip(zip(grid[:-1], grid[1:]), rates) if r > 1.5 * med],
                    "stable": [[a, b] for (a, b), r in zip(zip(grid[:-1], grid[1:]), rates) if r < 0.5 * med]}
prof["regions"] = regions
# plateau check: intervals where output level is ~flat but tone/saturation keep changing
pl = []
lv = prof["series"]["music RMS @0 dB DI"]
for k, (a, b) in enumerate(regions["level"]["intervals"]):
    dl = abs(regions["level"]["rate"][k]) * 1.0
    if regions["level"]["rate"][k] < 0.5 * np.median(regions["level"]["rate"]) and (regions["saturation"]["rate"][k] > np.median(regions["saturation"]["rate"]) or regions["tone"]["rate"][k] > np.median(regions["tone"]["rate"])):
        pl.append([a, b])
prof["level_plateau_but_character_changing"] = pl
(D / "profile.json").write_text(json.dumps(prof, indent=1))

# ---- plots
def panel(ax, names, title, ylabel, cols=None):
    for i, n in enumerate(names):
        e = prof["series"][n]; y = np.array(e["values"]); ok = np.array(e["reliable"]); g = np.array(gains)
        c = (cols or plt.cm.tab10.colors)[i % 10]
        f = PchipInterpolator(g[ok], y[ok]); xs = np.linspace(g.min(), g.max(), 200)
        ax.plot(xs, f(xs), "--", color=c, lw=1, alpha=0.7)
        if e["loo_median_abs_err"]:
            ax.fill_between(xs, f(xs) - e["loo_median_abs_err"], f(xs) + e["loo_median_abs_err"], color=c, alpha=0.08)
        ax.plot(g[ok], y[ok], "o", color=c, ms=4, label=n)
        if (~ok).any():
            ax.plot(g[~ok], y[~ok], "o", mfc="none", mec="red", ms=8)
    ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8); ax.set_xlabel("physical gain position", fontsize=8)
    ax.legend(fontsize=6, ncol=1); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

fig, axs = plt.subplots(3, 3, figsize=(17, 12))
panel(axs[0, 0], ["music RMS @-12 dB DI", "music RMS @0 dB DI", "music RMS @+6 dB DI", "music peak @0"], "LEVEL: output level (music, DI level offsets)", "dBFS")
panel(axs[0, 1], [f"EQ {b}" for b in EQ], "TONE: EQ band energy (relative)", "dB rel. total")
panel(axs[0, 2], ["tilt (presence-low)", "HF >3 kHz"], "TONE / brightness", "dB")
panel(axs[1, 0], ["THD 440Hz @-30", "THD 440Hz @-18", "THD 440Hz @-6"], "SATURATION: sine THD at three input levels", "dB")
panel(axs[1, 1], ["H2 440Hz @-18", "H3 440Hz @-18", "crest @0"], "SATURATION: harmonic distribution / crest", "dB")
panel(axs[1, 2], ["IO gain -54->-30", "IO gain -30->0"], "COMPRESSION: level change above linear (0 = linear)", "dB")
panel(axs[2, 0], ["dyn range @0", "dyn range @+6"], "DYNAMICS: dynamic range of music", "dB")
panel(axs[2, 1], ["noise floor (silence)"], "NOISE (silence render)", "dBFS")
ax = axs[2, 2]
for dim, c in zip(("level", "tone", "saturation", "compression", "dynamics"), plt.cm.tab10.colors):
    r = regions[dim]; mid = [(a + b) / 2 for a, b in r["intervals"]]
    ax.plot(mid, np.array(r["rate"]) / (np.median(r["rate"]) or 1), "-o", color=c, ms=3, label=dim)
ax.axhline(1, color="k", lw=0.5); ax.set_title("Where each dimension changes (rate / its median; >1.5 = fast)", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
fig.suptitle(f"{amp}: measured response (markers) and PCHIP estimate (dashed, band = median leave-one-out error); red rings = quarantined for that dimension", fontsize=10)
fig.tight_layout(); fig.savefig(D / "profile.png", dpi=110); plt.close(fig)
print(amp, "profile written; fast:", {k: v["fast"] for k, v in regions.items()}, "plateau-but-changing:", pl)
