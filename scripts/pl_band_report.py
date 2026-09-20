"""Render docs/CONTINUOUS_GAIN_WEIGHT_BAND_TEST.md from the teacher-only weight-band data. Findings text: docs/phase4e/weight_band_findings.md"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pl_band_eval as E

ROOT = E.ROOT; PL = E.PL; DOC = ROOT / "docs"; IMG = DOC / "phase4e" / "playability"
AMPS = {"jcm800": "Marshall JCM800 2203 (High)", "vibrolux": "Fender Super-Sonic Vibrolux"}
out = ["# Teacher-only weight-band test\n",
"**No NAM was trained and no exported model was used.** Scope, variant and the criteria for calling a variant promising were committed BEFORE any result: `docs/phase4e/weight_band_criteria.md` (commit `d688755`). Code: `scripts/pl_band_common.py` (variant weights; verified equal to `hybrid.multi_blend.chain_weights` at plateau fraction 0, maximum difference 0.0), `pl_band_cases.py`, `pl_band_sweep.py`, `pl_band_eval.py`, `pl_ambiguity.py`, `pl_band_report.py`; data `work/p4e/playability/band_*.json`. Cases, DIs and offsets are the same as the fixed-virtual-gain playability diagnostic (`docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md`). `hybrid/multi_blend.py` was not modified. Human listening remains pending.\n",
"**Variant.** Plateau fraction phi holds each anchor's capture at full weight over a band and concentrates the crossfade in the middle of the gap between adjacent anchors (`t = smoothstep(clip((p-(0.5-w/2))/w, 0, 1))`, `p` the position inside the gap, `w = 1-phi`). phi = 0 is the current teacher; phi = 1 is a hard switch at the midpoint. Swing = error at +6 minus error at -12 dB musical input (teacher minus real capture, same fixed virtual gain).\n"]
def fm(v, f="{:.2f}"): return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)
for amp, nm in AMPS.items():
    res, base = E.criteria(amp, "B"); cs = E.load(amp)
    out += [f"## {nm}\n", "### Primary configuration B (frozen Phase 4E anchors): mean over positions\n",
            "| phi | |swing| level | HF | crest | dyn range | EQ | fixed |offset| level | HF | crest | dyn range | centre-of-mass swing (positions) | nearest-anchor weight at nominal | median crossfade (ms) | switches/s | HF>10 kHz excess vs real (dB) | mean ESR |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for phi in E.PHIS:
        m = E.metrics(amp, "B", phi, cs)
        out.append(f"| {phi:g} | " + " | ".join(fm(m["swing"][k]) for k in ("level", "HF", "crest", "dyn range")) + f" | {m['eq_swing']:.2f} | " + " | ".join(fm(m["fixed"][k]) for k in ("level", "HF", "crest", "dyn range")) + f" | {m['com_swing']:.2f} | {m['nearest_weight_nominal']:.2f} | {fm(m['median_crossfade_ms'], '{:.1f}')} | {m['switches_per_s']:.1f} | {m['hf10k_excess']:+.2f} | {m['esr_mean']:.3f} |")
    out += ["", "### Predeclared criteria (B)\n", "| phi | 1: swing reduction level / HF / crest / dyn (need >= 40% each) | 2: omitted-position mean error change level / HF / crest / dyn (need <= +0.5 dB each; worst position <= +1.5) | 3: median crossfade ms (>= 5) / HF>10k excess vs phi 0 (<= 1.5 dB) | 1 | 2 | 3 | promising |", "|---:|---|---|---|---|---|---|---|"]
    for phi, r in res.items():
        out.append(f"| {phi:g} | " + " / ".join(f"{v*100:+.0f}%" for v in r["c1_swing_reduction"].values()) + " | " + " / ".join(f"{v:+.2f}" for v in r["c2_omitted_mean_increase"].values()) + f" (worst pos {max(r['c2_worst_position_increase'].values()):+.1f}) | {fm(r['c3']['median_crossfade_ms'], '{:.1f}')} / {r['c3']['hf10k_excess_vs_phi0']:+.2f} | {'pass' if r['pass1'] else 'FAIL'} | {'pass' if r['pass2'] else 'FAIL'} | {'pass' if r['pass3'] else 'FAIL'} | {'YES' if r['promising'] else 'no'} |")
    out.append("")
    gains = sorted({c["g"] for c in cs})
    out += ["### Swing by position, phi = 0 (current) versus phi = 0.75 and 1.0 (dB; teacher B minus real capture)\n", "| Position | level phi 0 / 0.75 / 1.0 | HF | crest | dyn range |", "|---|---|---|---|---|"]
    mm = {phi: E.metrics(amp, "B", phi, cs) for phi in (0.0, 0.75, 1.0)}
    for g in gains:
        out.append(f"| G{g:g} | " + " | ".join(" / ".join(f"{mm[phi]['swing_by_pos'][k][g]:+.1f}" for phi in (0.0, 0.75, 1.0)) for k in ("level", "HF", "crest", "dyn range")) + " |")
    out.append("")
    out += ["### Same test on configuration A (fixed anchors), for comparison: swing reduction relative to phi 0 (level / HF / crest / dyn range)\n", "| phi | reduction |", "|---:|---|"]
    resA, baseA = E.criteria(amp, "A")
    for phi, r in resA.items(): out.append(f"| {phi:g} | " + " / ".join(f"{v*100:+.0f}%" for v in r["c1_swing_reduction"].values()) + " |")
    out.append("")
    # progression sweep
    out += ["### Progression cost: continuous Input-gain sweep, nominal musical input (teacher B and A)\n", "Steps of the teacher's features between Input gains 2 dB apart over the tested range (-22 to +14 dB), mean over three held-out DIs. Larger maximum steps or a higher max/median ratio mean a more stair-stepped progression.\n", "| config | phi | max step level / HF / crest / dyn (dB per 2 dB) | max/median step (worst of the four) |", "|---|---:|---|---:|"]
    for cfg in ("B", "A"):
        sw = json.loads((PL / f"band_sweep_{amp}_{cfg}.json").read_text()); T = np.array(sw["T"]); msk = (T >= -22) & (T <= 14)
        for phi in E.PHIS:
            ms, ratio = [], 0.0
            for f in ("rms_db", "hf3k_db", "crest_db", "dyn_range_db"):
                y = np.mean([[r[f] for r in sw["data"][d][str(phi)]] for d in sw["data"]], axis=0)[msk]; st = np.abs(np.diff(y)); ms.append(st.max()); ratio = max(ratio, st.max() / max(np.median(st), 0.05))
            out.append(f"| {cfg} | {phi:g} | " + " / ".join(f"{v:.2f}" for v in ms) + f" | {ratio:.1f} |")
    out.append("")

# figure: swing vs phi
fig, axs = plt.subplots(2, 4, figsize=(18, 7))
for i, amp in enumerate(AMPS):
    cs = E.load(amp); ms = {phi: E.metrics(amp, "B", phi, cs) for phi in E.PHIS}
    for ax, k in zip(axs[i], ("level", "HF", "crest", "dyn range")):
        ax.plot(E.PHIS, [ms[p]["swing"][k] for p in E.PHIS], "-o", color="tab:red", label="mean |swing| (over 18 dB of playing)")
        ax.plot(E.PHIS, [ms[p]["omitted_err"][k]["mean"] for p in E.PHIS], "-s", color="tab:blue", label="mean |error| at omitted positions, nominal")
        ax.axhline(0.6 * ms[0.0]["swing"][k], ls=":", color="gray", label="60% of phi=0 swing (criterion 1 line)"); ax.set_title(f"{AMPS[amp]}: {k} (dB)", fontsize=9); ax.set_xlabel("plateau fraction phi"); ax.grid(alpha=.25); ax.legend(fontsize=5)
fig.tight_layout(); fig.savefig(IMG / "weight_band_swing_vs_phi.png", dpi=100); plt.close(fig)
out += ["![swing vs phi](phase4e/playability/weight_band_swing_vs_phi.png)\n"]

# supplementary ambiguity
out += ["## Supplementary: the intrinsic gain/level ambiguity measured on the real captures (no teacher, no NAM)\n",
"Two different situations hand the SAME waveform to a standard NAM: capture a played at offset `o + gap` and capture b played at offset `o`, where gap is the difference between their virtual-gain settings. Whatever a single NAM outputs must sit between what the two real amps would do. The table is the difference between the two real amps for identical input (mean absolute over four offsets and three DIs).\n",
"| Amp | rule | adjacent anchors | gap (dB) | |level| | |HF| | |crest| | |dyn range| | EQ (mean abs) |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
for amp, nm in AMPS.items():
    r = json.loads((PL / f"ambiguity_{amp}.json").read_text())["pairs"]
    for rule in ("B", "A"):
        for a, b in sorted({(p["a"], p["b"]) for p in r if p["rule"] == rule}):
            ps = [p for p in r if p["rule"] == rule and p["a"] == a and p["b"] == b]
            out.append(f"| {nm.split(' (')[0]} | {rule} | G{a:g} to G{b:g} | {ps[0]['gap_db']:.1f} | " + " | ".join(f"{np.mean([abs(p[k]) for p in ps]):.2f}" for k in ("level", "HF", "crest", "dyn", "eq_abs")) + " |")
out.append("")
fp = DOC / "phase4e" / "weight_band_findings.md"
out.append(fp.read_text() if fp.exists() else "## Findings\n\n(pending)\n")
(DOC / "CONTINUOUS_GAIN_WEIGHT_BAND_TEST.md").write_text("\n".join(out)); print("written", len(out))
