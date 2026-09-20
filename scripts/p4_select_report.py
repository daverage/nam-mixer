"""Render docs/CONTINUOUS_GAIN_PHASE4_SELECTION.md (+ docs/phase4/selection_curves.png) from work/p4/<amp>/{selection,teacher,mapping,audit}.json."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "work" / "p4"
NAMES = {"jcm800": "JCM800", "twin": "Twin", "supersonic": "Super-Sonic Bassman", "vibrolux": "Super-Sonic Vibrolux",
         "peavey": "Peavey 5150", "peavey6505": "Peavey 6505+", "mesa": "Mesa Dual Rectifier", "orange": "Orange Dual Terror"}
L = lambda a, f: json.loads((R / a / f).read_text())
SEL = {a: L(a, "selection.json") for a in NAMES}; TEA = {a: L(a, "teacher.json") for a in NAMES}
MAP = {a: L(a, "mapping.json") for a in NAMES}; AUD = {a: L(a, "audit.json")["captures"] for a in NAMES}
T_LO, T_HI, MIN_SEP = -22.0, 14.0, 4.0
fmt = lambda s: "G" + ", G".join(f"{g:g}" for g in s)

def proposed_T(a, sel):
    rc = SEL[a]["response_coordinate"]; g, arc = rc["gains"], rc["arc"]
    at = lambda x: float(np.interp(x, g, arc))
    lo, hi = at(min(sel)), at(max(sel)); span = (hi - lo) or 1.0
    T = [T_LO + (T_HI - T_LO) * (at(x) - lo) / span for x in sorted(sel)]
    for _ in range(6):   # enforce a minimum separation between anchors so adjacent captures stay separable by level
        for i in range(1, len(T)):
            T[i] = max(T[i], T[i - 1] + MIN_SEP)
        T = [T_LO + (T_HI - T_LO) * (t - T[0]) / ((T[-1] - T[0]) or 1.0) for t in T]
    return T

out = ["# Phase 4D: which captures carry distinct information? (selection without an imposed count)\n",
"Non-neural analysis on the validated captures; nothing here was trained. Code: `scripts/p4_select.py` (exhaustive subset search on measured profiles), `scripts/p4_select_teacher.py` (audio-domain blend check), `scripts/p4_select_report.py`; data `work/p4/<amp>/selection.json`, `teacher.json`.\n",
"## Method\n",
"- **Question asked of each subset S:** how well are the OTHER captures reproduced from S? Each omitted position is predicted by shape-preserving (PCHIP) interpolation of the selected captures' measured response along the knob axis (held flat outside the selected span), separately for **tone** (six EQ bands + tilt), **saturation** (HF>3 kHz, sine THD at three levels, H2, H3, crest) and **compression** (two input/output slopes, dynamic range). **Level is reported, not optimised** (captures are normalised). THD/harmonics are floored at -80 dB (below that is numerical floor, not signal).\n"
"- **Objective:** per dimension, half the mean and half the worst-case normalised error over all integer positions (selected positions count as zero), summed over the three dimensions. Every subset of the eligible captures is evaluated (at most 1024 per amp); no count is fixed in advance and no neural training is involved.\n"
"- **Eligible anchors:** captures whose 4A status is VALID or CORRECTED. SUSPECT captures are analysis-only: they are still scored as omitted positions, and how well their neighbours predict them is reported separately.\n"
"- **How many captures?** Not imposed. The report shows the whole error-versus-count curve; `k*` is the smallest count at which EVERY omitted position is within a working tolerance in EVERY physical group (tone 1.0 dB, HF 1.0, crest 1.0, THD 5, IO slope 2.0, dynamic range 1.5). These tolerances are working values, not perceptual measurements, so `k*` is evidence, not a rule.\n"
"- **JCM800:** searched on the 10 integer positions only; the 9 genuine half-steps are an independent check of the chosen sets.\n"
"- **Audio check:** for chosen sets the real captures (timing-aligned with the 4A corrections) are blended in the waveform domain with weight linear in knob position, and the result is compared with the real omitted captures on the fit DIs. This tests the teacher-construction step itself, and is not affected by any NAM.\n"
"- **Hypothesis test, not the criterion:** the v3 NAM's own per-position errors are compared with the profile analysis at the end; they were not used to choose subsets.\n"]

out += ["## Summary across amps\n",
"| Amp | eligible anchors | k* | best set at k* | best 3 | best 5 | v3 C3 (G1,G5,G10) J | best-3 J | v3 C5 J | best-5 J |", "|---|---|---:|---|---|---|---:|---:|---:|---:|"]
for a, nm in NAMES.items():
    s = SEL[a]; ks = s["k_star_all_within_tolerance"]
    b = s["baselines"]
    out.append(f"| {nm} | {len(s['eligible'])} | {ks if ks else 'none'} | {fmt(s['by_k'][str(ks)]['best']) if ks else '-'} | {fmt(s['by_k']['3']['best'])} | {fmt(s['by_k']['5']['best'])} | "
               f"{b['G1,G5,G10 (v3 C3)']['J']:.3f} | {s['by_k']['3']['J']:.3f} | {b['G1,G3,G5,G7,G10 (v3 C5)']['J']:.3f} | {s['by_k']['5']['J']:.3f} |")
out += ["", "J = objective above (lower is better). The v3 fixed sets G1/G5/G10 and G1/G3/G5/G7/G10 are compared with the best set of the same size for that amp.\n",
        "![selection error vs count](phase4/selection_curves.png)\n"]

fig, axs = plt.subplots(2, 4, figsize=(18, 7.5))
for ax, (a, nm) in zip(axs.ravel(), NAMES.items()):
    s = SEL[a]; ks = sorted(int(k) for k in s["by_k"])
    ax.plot(ks, [s["by_k"][str(k)]["J"] for k in ks], "-o", ms=4, label="best subset")
    ax.plot(ks, [s["by_k"][str(k)]["J_without_both_endpoints"] for k in ks], "--", color="tab:orange", label="best without both endpoints")
    ax.scatter([3], [s["baselines"]["G1,G5,G10 (v3 C3)"]["J"]], color="tab:red", zorder=5, label="v3 G1,G5,G10")
    ax.scatter([5], [s["baselines"]["G1,G3,G5,G7,G10 (v3 C5)"]["J"]], color="tab:green", zorder=5, label="v3 G1,G3,G5,G7,G10")
    if s["k_star_all_within_tolerance"]: ax.axvline(s["k_star_all_within_tolerance"], color="k", ls=":", lw=1)
    ax.set_title(nm, fontsize=10); ax.set_xlabel("number of training captures", fontsize=8); ax.set_ylabel("reconstruction error J", fontsize=8); ax.grid(alpha=.25); ax.legend(fontsize=6)
fig.suptitle("Reconstruction error of the omitted captures vs number of selected captures (dotted line = smallest count with everything within tolerance)", fontsize=10)
fig.tight_layout(); fig.savefig(ROOT / "docs" / "phase4" / "selection_curves.png", dpi=110); plt.close(fig)

# hypothesis test: does profile-based distinctness predict where the v3 NAM struggles?
xs, ys, rows = [], [], []
for a in NAMES:
    m3 = MAP[a]["models"]["3"]["mappings"]["measured_all"]["heldout_o0"]; loo = SEL[a]["loo_normalised_error"]
    trained = set(MAP[a]["models"]["3"]["trained_gains"])
    for k, v in loo.items():
        g = float(k)
        if g in trained or k not in m3: continue
        fe = m3[k]
        nam = {"tone": np.mean([fe[f] for f in ("eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db")]),
               "saturation": np.mean([fe["hf3k_db"], fe["crest_db"]]), "compression": np.mean([fe["io_-54->-30"], fe["io_-30->0"], fe["dyn_range_db"]])}
        for d in ("tone", "saturation", "compression"):
            rows.append((a, d, v[d], nam[d]))
out += ["## Hypothesis test: does the profile analysis predict where the v3 NAM (C3, measured mapping) struggles?\n",
        "For interior positions the C3 model was NOT trained on: rank correlation between the capture's leave-one-out error (how badly its neighbours predict it, from the profile) and the C3 model's own held-out error at that position, per dimension. Positive = positions that are hard to interpolate are also where the NAM is worse. Not a selection criterion; a check that the profile analysis is measuring something the NAM also feels.\n",
        "| Dimension | n positions | Spearman rho | p |", "|---|---:|---:|---:|"]
for d in ("tone", "saturation", "compression"):
    x = [r[2] for r in rows if r[1] == d]; y = [r[3] for r in rows if r[1] == d]
    rho, p = spearmanr(x, y)
    out.append(f"| {d} | {len(x)} | {rho:+.2f} | {p:.3f} |")
out.append("\nPositions are pooled over the eight amps (single seed, one model per amp), so treat as indicative.\n")

out += ["## Per-amp results\n"]
for a, nm in NAMES.items():
    s, t = SEL[a], TEA[a]
    out.append(f"### {nm}\n")
    if s["ineligible"]:
        out.append(f"Not eligible as training anchors (analysis-only, 4A status): {', '.join(f'G{k} {v}' for k, v in s['ineligible'].items())}.\n")
    out.append("**Best subset per count** (evaluated over every subset of the eligible captures; `fails` = physical groups where some omitted position exceeds its working tolerance):\n")
    out.append("| k | best set | J | best set if not both endpoints are kept | J | fails (max error dB) |"); out.append("|---:|---|---:|---|---:|---|")
    for k in sorted(int(x) for x in s["by_k"]):
        b = s["by_k"][str(k)]
        fails = {g: v["max"] for g, v in b["phys"].items() if v["tol"] is not None and v["max"] > v["tol"]} if b["phys"] else {}
        jne = "-" if b["J_without_both_endpoints"] is None else "%.3f" % b["J_without_both_endpoints"]
        sne = fmt(b["best_without_both_endpoints"]) if b["best_without_both_endpoints"] else "-"
        fs = ", ".join("%s %.1f" % (g, v) for g, v in fails.items()) or "none"
        out.append(f"| {k} | {fmt(b['best'])} | {b['J']:.3f} | {sne} | {jne} | {fs} |")
    out.append("")
    out.append("**Greedy contribution order** (each step adds the capture that reduces J most): " + " > ".join(f"G{g['add']:g} ({g['J_after']:.3f})" for g in s["greedy_order"]) + ".\n")
    loo = s["loo_normalised_error"]; ep = s["endpoint_omitted_normalised_error"]
    out.append("**Distinctness** (leave-one-out error, normalised by each dimension's range; large = the capture is not predictable from its neighbours = carries distinct information):\n")
    out.append("| Position | tone | saturation | compression |"); out.append("|---|---:|---:|---:|")
    for k, v in {**{list(ep)[0]: ep[list(ep)[0]]}, **loo, **{list(ep)[1]: ep[list(ep)[1]]}}.items():
        tag = " (endpoint, extrapolated)" if k in ep else ""
        out.append(f"| G{k}{tag} | {v['tone']:.3f} | {v['saturation']:.3f} | {v['compression']:.3f} |")
    out.append("")
    if s.get("errors_at_suspect_positions"):
        k0 = str(s["k_star_all_within_tolerance"] or 5)
        e = s["errors_at_suspect_positions"][k0]
        out.append(f"**How well neighbours predict the SUSPECT captures** (best set at k={k0}; physical units; large = irregular capture that cannot be inferred, so the flag may matter to the training plan): " + "; ".join(f"G{g}: " + ", ".join(f"{m} {v:.1f}" for m, v in x.items() if not m.startswith('level')) for g, x in e.items()) + ".\n")
    if s.get("half_step_validation"):
        hv = s["half_step_validation"]
        out.append("**Independent check on the genuine half-step captures** (never in the search): mean error when predicted from each chosen set (tone EQ / HF / crest / IO slope / dyn range, dB):\n")
        out.append("| Set | tone | HF | crest | IO slope | dyn range |"); out.append("|---|---:|---:|---:|---:|---:|")
        for name, ph in hv.items():
            label = f"best k={name}" if name.isdigit() else name
            if name.isdigit() and int(name) not in (3, 4, 5, 6): continue
            out.append(f"| {label} | " + " | ".join(f"{ph[g]['mean']:.2f}" for g in ("tone EQ (dB)", "HF>3k (dB)", "crest (dB)", "IO slope (dB)", "dyn range (dB)")) + " |")
        out.append("")
    out.append("**Audio-domain check** (waveform blend of the selected captures vs the real omitted captures, fit DIs, mean absolute error in dB; lower is better):\n")
    out.append("| Set | tone EQ | HF>3k | crest | dyn range | level |"); out.append("|---|---:|---:|---:|---:|---:|")
    for name, c in t["candidates"].items():
        out.append(f"| {name}: {fmt(c['sel'])} | " + " | ".join(f"{c['mean'][m]:.2f}" for m in ("tone EQ (dB)", "HF>3k (dB)", "crest (dB)", "dyn range (dB)", "level (dB)")) + " |")
    out.append("")
    for k in sorted({3, 5, s["k_star_all_within_tolerance"] or 5}):
        sel = s["by_k"][str(k)]["best"]
        out.append(f"Proposed input-gain anchors for the best k={k} set ({fmt(sel)}), from the measured response distance between ALL validated captures (analysis-only captures included; range {T_LO:+.0f} to {T_HI:+.0f} dB as in v3, minimum {MIN_SEP:.0f} dB between anchors): " + ", ".join(f"G{g:g} {x:+.1f}" for g, x in zip(sorted(sel), proposed_T(a, sel))) + " dB.")
    out.append("")
(ROOT / "docs" / "CONTINUOUS_GAIN_PHASE4_SELECTION.md").write_text("\n".join(out))
print(len(out), "lines")
for d in ("tone", "saturation", "compression"):
    x = [r[2] for r in rows if r[1] == d]; y = [r[3] for r in rows if r[1] == d]; print(d, len(x), spearmanr(x, y))
