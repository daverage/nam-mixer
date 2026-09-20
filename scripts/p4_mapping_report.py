"""Render docs/CONTINUOUS_GAIN_PHASE4_MAPPING.md (+ work/p4/mapping_curves.png) from work/p4/<amp>/mapping.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "work" / "p4"
NAMES = {"jcm800": "JCM800", "twin": "Twin", "supersonic": "Super-Sonic Bassman", "vibrolux": "Super-Sonic Vibrolux",
         "peavey": "Peavey 5150", "peavey6505": "Peavey 6505+", "mesa": "Mesa Dual Rectifier", "orange": "Orange Dual Terror"}
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]
GROUPS = {"tone (EQ, dB)": EQ, "HF>3k (dB)": ["hf3k_db"], "crest (dB)": ["crest_db"], "THD@-30 (dB)": ["thd@-30"],
          "IO slope (dB)": ["io_-54->-30", "io_-30->0"], "dyn range (dB)": ["dyn_range_db"], "level (dB)": ["rms_db"]}
TOL = {"tone (EQ, dB)": 0.15, "HF>3k (dB)": 0.2, "crest (dB)": 0.2, "THD@-30 (dB)": 1.5, "IO slope (dB)": 0.5, "dyn range (dB)": 0.3, "level (dB)": 0.4}
HIGH = {"tone (EQ, dB)": 1.0, "HF>3k (dB)": 1.0, "crest (dB)": 1.0, "THD@-30 (dB)": 5.0, "IO slope (dB)": 2.0, "dyn range (dB)": 1.5, "level (dB)": 1.5}
ORACLE_DIM = {"tone (EQ, dB)": "tone", "HF>3k (dB)": "saturation", "crest (dB)": "saturation", "THD@-30 (dB)": "saturation", "IO slope (dB)": "compression", "dyn range (dB)": "compression", "level (dB)": "level"}
A = {a: json.loads((R / a / "mapping.json").read_text()) for a in NAMES}
AUD = {a: json.loads((R / a / "audit.json").read_text())["captures"] for a in NAMES}

def gerr(errs, group, amp, subset=None):
    vals = []
    for k, fe in errs.items():
        g = float(k)
        if subset is not None and not subset(g): continue
        if group == "level (dB)" and any(e.startswith("level:") for e in AUD[amp][k]["evidence"]): continue
        vals.append(np.mean([fe[f] for f in GROUPS[group]]))
    return float(np.mean(vals)) if vals else float("nan")

def row(amp, n, name, subset=None):
    m = A[amp]["models"][str(n)]
    if name == "oracle (UB)":
        return {g: gerr(m["oracle_per_dimension_ub"][ORACLE_DIM[g]]["err"], g, amp, subset) for g in GROUPS}
    return {g: gerr(m["mappings"][name]["heldout_o0"], g, amp, subset) for g in GROUPS}

out = ["# Phase 4C: can a better input-gain mapping make an existing NAM behave more like the amp? (no retraining)\n",
"Uses the existing v3 models (C3 = G1/G5/G10, C5 = G1/G3/G5/G7/G10, C10 = all) unchanged and changes ONLY the playback input-gain mapping. Code: `scripts/p4_model_sweep.py` (renders each model over -48..+36 dB input gain plus a sine input/output sweep), `scripts/p4_mapping.py` (fit + evaluation), `scripts/p4_mapping_report.py`; data `work/p4/<amp>/mapping.json`.\n",
"## Method\n",
"- **Fixed** = the v3 design: gain N at input level `-22 + 4(N-1)` dB (G1 -22 ... G10 +14) for every amp.\n"
"- **Measured** = one ordered (non-decreasing) mapping per amp and model, fitted on the fit DIs only (clean_smooth, moderate_hotrod, high_thrash, high_metalcore) by minimising, per real capture, a scale-free distance in **tone** (six EQ bands), **saturation** (HF>3 kHz, crest, sine THD) and **compression** (two input/output slopes, dynamic range). **Output level is deliberately excluded from the fit** and reported separately; it is not an RMS match. One mapping for all DIs and levels: nothing is tuned per recording or per metric.\n"
"- **Held-out validation** uses moderate_brit, clean_mayer and bass_rollin, which were never used for any fit or mapping. On the JCM800 the mapping is fitted on the 10 integer positions only, so its 9 genuine half-step captures are untouched reference positions (their mapping values are PCHIP-interpolated).\n"
"- **Quarantine:** captures flagged SUSPECT for level (Peavey G4, Orange G2) are excluded from the level column; timing does not affect these measurements.\n"
"- **Trade-off mappings:** the same fit using only one dimension, and an RMS-only fit (the naive approach the brief warns against), are shown so trade-offs are visible.\n"
"- **Oracle (upper bound):** best per-capture input gain per dimension chosen ON the held-out DIs, unordered. It is a ceiling that shows how much any mapping could ever fix; it is **not** a valid mapping and is never reported as a result.\n"
"- Thresholds below (`material` = smallest change worth noting; `high` = error still likely audible) are **working values, not perceptual measurements**: tone 0.15/1.0 dB, HF 0.2/1.0, crest 0.2/1.0, THD 1.5/5, IO slope 0.5/2, dynamic range 0.3/1.5, level 0.4/1.5.\n"
"- Single seed models; small differences are not established.\n"]
COLS = list(GROUPS)
def table(amp, n, subset=None, names=("fixed_4dB", "measured_all", "rms_only", "oracle (UB)")):
    o = ["| Mapping | " + " | ".join(COLS) + " |", "|---|" + "---:|" * len(COLS)]
    for nm in names:
        r = row(amp, n, nm, subset)
        o.append(f"| {nm} | " + " | ".join(f"{r[c]:.2f}" for c in COLS) + " |")
    return o

verdict = {}
trade = []
for amp in NAMES:
    for n in (3, 10):
        f, m = row(amp, n, "fixed_4dB"), row(amp, n, "measured_all")
        for c in COLS:
            d = m[c] - f[c]
            verdict[(amp, n, c)] = ("helps" if d < -TOL[c] and m[c] < 0.8 * f[c] else "worsens" if d > TOL[c] and m[c] > 1.25 * f[c] else "same")
        better = [c for c in COLS if verdict[(amp, n, c)] == "helps"]; worse = [c for c in COLS if verdict[(amp, n, c)] == "worsens"]
        if better and worse:
            trade.append(f"- **{NAMES[amp]}, C{n}:** measured mapping improves {', '.join(f'{c} ({f[c]:.2f} to {m[c]:.2f})' for c in better)} but worsens {', '.join(f'{c} ({f[c]:.2f} to {m[c]:.2f})' for c in worse)}.")

def io_row(amp, n):
    m = A[amp]["models"][str(n)]
    f = lambda e: float(np.mean([np.mean([fe[k] for k in ("io_-54->-30", "io_-30->0")]) for fe in e.values()]))
    return f(m["mappings"]["fixed_4dB"]["heldout_o0"]), f(m["mappings"]["measured_all"]["heldout_o0"]), f(m["oracle_per_dimension_ub"]["compression"]["err"])
out += ["## Findings (mapping-only; existing v3 models; held-out DIs)\n",
"1. **Compression (input/output slope) is the dimension a measured mapping helps most, and the one it cannot finish.** Static input/output slope error (dB, mean of two slopes) for fixed / measured / oracle-ceiling:\n",
"| Amp | C3 fixed | C3 measured | C3 oracle UB | C10 fixed | C10 measured | C10 oracle UB |", "|---|---:|---:|---:|---:|---:|---:|"]
for amp in NAMES:
    a3, a10 = io_row(amp, 3), io_row(amp, 10)
    out.append(f"| {NAMES[amp]} | " + " | ".join(f"{v:.2f}" for v in (*a3, *a10)) + " |")
out += ["",
"   The measured mapping lowers C3 slope error clearly on five amps (JCM800, Twin, Vibrolux, Peavey 5150, 6505+), marginally on Orange, and not on Mesa or Super-Sonic Bassman (Bassman's oracle ceiling, 1.25, is well below the fitted 3.56, so headroom exists there that my joint tone+saturation+compression fit did not find; a compression-weighted fit might). What remains after even the oracle ceiling (about 1-5 dB) cannot be reached by any input-gain mapping of that model, and C10 does not consistently reduce it (Twin, Mesa and Peavey 5150 are worse at C10). That points to the level-driven design itself (a louder input is read as a higher gain setting, so the model compresses differently from a fixed-gain amp), not to mapping or to capture coverage.",
"2. **Tone is mostly reachable with either mapping.** Fixed-mapping tone error is already 0.4-1.0 dB; the measured mapping helps on Bassman, Mesa and Orange and worsens nothing at C3. Tone is not where the mapping matters.",
"3. **Saturation and dynamics gains are amp-specific.** The measured mapping helps crest/dynamic range on JCM800, Peavey 5150 (but HF worsens, 0.64 to 1.03 dB), Mesa and Orange, and HF on Bassman, Mesa and Orange; on the clean Fenders (Twin, Vibrolux) it worsens crest (and HF/THD on the Twin).",
"4. **Character versus output level is a genuine conflict on the clean Fenders.** Level is excluded from the fit; the measured mapping raises level error on Twin (1.47 to 2.91 dB) and Vibrolux (0.78 to 3.80 dB) at C3 while improving compression. A single mapping cannot give both the real amp's character progression and its level progression there. This must not be reported as a general improvement.",
"5. **The measured mapping is compressive at the top.** On the JCM800 G5-G10 spans about 8 dB of input gain instead of the fixed 20 dB, consistent with the real amp's plateau (see the 4B profile): equal 4 dB steps over-spend range where the real amp has stopped changing.",
"6. **Independent check (JCM800 half-steps, never used in the fit):** C3 with the measured mapping is better than fixed in all seven columns (tone 0.35 to 0.29, HF 0.46 to 0.40, crest 0.49 to 0.32, THD 1.23 to 0.70, IO slope 4.43 to 3.20, dyn range 0.91 to 0.46, level 0.74 to 0.55). C10 is mixed (IO slope and dyn range better; THD 0.57 to 0.92 and tone 0.33 to 0.36 slightly worse).",
"7. **Not shown here:** these are technical metrics on level-matched-to-native playback with a single training seed per model; no audible improvement has been demonstrated, and a model TRAINED with the measured mapping (Phase 4E) may behave differently from the same mapping applied to a model trained on fixed spacing.\n"]
out += ["## Cross-amp verdict for the mapping-only baseline (C3, held-out DIs, DI level 0)\n",
        "`helps` / `worsens` = measured mapping vs fixed 4 dB, beyond the working thresholds; `-` = no material change. Add `*` where the error is still above the `high` threshold under the best mapping (measured or fixed) and the oracle ceiling is no better than 70% of it, i.e. mapping alone cannot fix it for this model.\n",
        "| Amp | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
for amp in NAMES:
    f, m, o_ = row(amp, 3, "fixed_4dB"), row(amp, 3, "measured_all"), row(amp, 3, "oracle (UB)")
    cells = []
    for c in COLS:
        v = verdict[(amp, 3, c)]; best = min(f[c], m[c])
        star = "*" if best > HIGH[c] and o_[c] > 0.7 * best else ""
        cells.append(("-" if v == "same" else v) + star)
    out.append(f"| {NAMES[amp]} | " + " | ".join(cells) + " |")
out += ["", "### Trade-offs (improving one characteristic worsens another)\n"] + (sorted(set(trade)) or ["- none"]) + [""]

out += ["## Coverage check: does a richer model (C10) reach what C3 cannot, under the same measured mapping?\n",
        "Where C10 is much better than C3 with the SAME method (error < 60% and materially lower), the limit is capture coverage, not mapping. Otherwise it is not fixed by more captures either.\n",
        "| Amp | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
for amp in NAMES:
    m3, m10 = row(amp, 3, "measured_all"), row(amp, 10, "measured_all")
    cells = [("coverage" if m10[c] < 0.6 * m3[c] and m3[c] - m10[c] > TOL[c] else "-") for c in COLS]
    out.append(f"| {NAMES[amp]} | " + " | ".join(cells) + " |")
out.append("")

out += ["## Per-amp results (mean absolute error vs the real capture, held-out DIs, DI level 0; lower is better)\n"]
for amp in NAMES:
    out.append(f"### {NAMES[amp]}\n")
    for n in (3, 5, 10):
        out.append(f"**C{n}** (trained on G{', G'.join(f'{g:g}' for g in A[amp]['models'][str(n)]['trained_gains'])})\n")
        out += table(amp, n); out.append("")
    if any(g != int(g) for g in A[amp]["gains"]):
        out.append("**Independent half-step captures only (untouched by the fit)**, C3 and C10:\n")
        for n in (3, 10):
            out.append(f"C{n}:\n"); out += table(amp, n, subset=lambda g: g != int(g), names=("fixed_4dB", "measured_all", "rms_only")); out.append("")
    g = A[amp]["gains"]; ints = [x for x in g if x == int(x)]
    out.append("Input-gain mapping (dB) by physical gain position:\n")
    out.append("| Mapping | " + " | ".join(f"G{x:g}" for x in ints) + " |"); out.append("|---|" + "---:|" * len(ints))
    for n in (3, 10):
        for nm in ("fixed_4dB", "measured_all", "rms_only"):
            T = A[amp]["models"][str(n)]["mappings"][nm]["T"]
            out.append(f"| C{n} {nm} | " + " | ".join(f"{T[g.index(x)]:+.0f}" for x in ints) + " |")
    out.append("")
    out.append("Robustness to guitar level: error with the SAME mapping when the DI is quieter/hotter (C3, held-out DIs; music-based groups only):\n")
    out.append("| DI offset | mapping | tone | HF>3k | crest | dyn range | level |"); out.append("|---:|---|---:|---:|---:|---:|---:|")
    for o in ("-12", "-6", "0", "6"):
        for nm in ("fixed_4dB", "measured_all"):
            e = A[amp]["models"]["3"]["mappings"][nm]["heldout_o0" if o == "0" else "heldout_by_offset"]
            e = e if o == "0" else e[f"{float(o):g}"]
            out.append(f"| {o} | {nm} | " + " | ".join(f"{gerr(e, c, amp):.2f}" for c in ("tone (EQ, dB)", "HF>3k (dB)", "crest (dB)", "dyn range (dB)", "level (dB)")) + " |")
    out.append("")

# curves
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
for ax, amp in zip(axs.ravel(), NAMES):
    g = A[amp]["gains"]
    ax.plot(g, A[amp]["fixed_T"], "k--", label="fixed 4 dB")
    for n, c in ((3, "tab:red"), (10, "tab:blue")):
        ax.plot(g, A[amp]["models"][str(n)]["mappings"]["measured_all"]["T"], "-o", ms=3, color=c, label=f"measured, C{n}")
    ax.plot(g, A[amp]["models"]["3"]["mappings"]["rms_only"]["T"], ":", color="tab:green", label="RMS-only, C3")
    ax.set_title(NAMES[amp], fontsize=10); ax.set_xlabel("physical gain position", fontsize=8); ax.set_ylabel("input gain (dB)", fontsize=8); ax.grid(alpha=.25); ax.legend(fontsize=6)
fig.suptitle("Input-gain mapping per physical gain position: fixed 4 dB vs measured (fit on tone+saturation+compression) vs RMS-only", fontsize=10)
fig.tight_layout(); fig.savefig(R / "mapping_curves.png", dpi=110); plt.close(fig)
(ROOT / "docs" / "CONTINUOUS_GAIN_PHASE4_MAPPING.md").write_text("\n".join(out))
print(len(out), "lines"); print("\n".join(trade))
