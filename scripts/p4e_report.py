"""Render docs/CONTINUOUS_GAIN_PHASE4E_RESULTS.md (+ docs/phase4e/*.png) from the frozen Phase 4E data. Comparison rules are applied mechanically."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import p4e_metrics as M

ROOT = M.ROOT; P4E = M.P4E
AMPS = {"jcm800": "Marshall JCM800 2203 (High)", "vibrolux": "Fender Super-Sonic Vibrolux"}
DIS_T = {k: json.loads((ROOT / "work" / "cg" / a / f"cg_{k}" / "manifest.json").read_text())["gains"] for a in AMPS for k in (3, 5, 10)}
fmtc = lambda s: f"{s['mean']:.2f} ({s['worst']:.2f})"
SHOW = ["level_signed", "level_abs", "eq", "hf", "tilt", "thd", "h23", "io_slope", "di_intensity", "crest", "dyn", "transient", "esr"]
TOL = {"level_abs": 0.4, "eq": 0.15, "hf": 0.2, "tilt": 0.2, "thd": 1.5, "h23": 1.5, "io_slope": 0.5, "di_intensity": 0.5, "crest": 0.2, "dyn": 0.3, "transient": 0.2, "esr": 0.01}
out = ["# Phase 4E results: selected captures and response-distance training anchors (JCM800, Super-Sonic Vibrolux)\n"]
DATA = {a: M.load(a) for a in AMPS}
avail = {a: list(DATA[a][1]) for a in AMPS}
new_ready = all(all(k in avail[a] for k in ("A_s0", "A_s1", "B_s0", "B_s1")) for a in AMPS)
man = M.MAN
fc = (P4E / "freeze_commit.txt").read_text().strip() if (P4E / "freeze_commit.txt").exists() else "?"
out.append(f"**Status:** {'primary pilot complete (8 new models evaluated)' if new_ready else 'INCOMPLETE - new-model evaluations missing'}. Perceptual (listening) evaluation: **PENDING**, no human listening has happened. Frozen manifest: `docs/phase4e/manifest_frozen.json`, freeze commit `{fc}`. Brief: `docs/CONTINUOUS_GAIN_PHASE4E_REVISED.md`.\n")

# --- 1 design
out += ["## 1. What was run\n",
"Two pilot amps, four new configurations (A: selected captures at the original fixed physical-position anchors; B: the same captures at response-distance anchors), two predeclared seeds (0, 1) each = 8 new standard A2 `.nam` files. Existing v3 C3 (G1,G5,G10), C5 and C10 models are the baselines and were not retrained.\n",
"| Amp | training captures | A anchors (dB) | B anchors (dB) |", "|---|---|---|---|"]
for a, nm in AMPS.items():
    am = man["amps"][a]; g = am["selected_positions"]
    out.append(f"| {nm} | G{', G'.join(str(x) for x in g)} | " + ", ".join(f"{x:+.1f}" for x in am["anchors_designated_input_gain_db"]["A_fixed_physical_position_spacing"].values()) + " | " + ", ".join(f"{x:+.1f}" for x in am["anchors_designated_input_gain_db"]["B_response_distance_spacing"].values()) + " |")
out += ["", "Anchors are designated input gain in dB (level = anchor + -30 dBFS reference). The B anchors are a hypothesis, not an established knob-to-drive mapping.\n",
"**Comparison rules (frozen):** A vs v3 C3 compares a new capture SET (count and placement both change) under the original anchor rule and does not isolate placement; B vs A isolates the anchor rule within each set. A difference is conclusive only if it is larger than the seed-to-seed difference and has the same sign in both seeds; otherwise inconclusive. No composite score. Working thresholds only flag differences for examination.\n",
"**Playback mappings evaluated:** *intended* (A and v3: fixed rule `-22+4(N-1)` dB; B: linear in the response coordinate between its anchors), *fixed* (fixed rule for every model; for B this is a deliberate mismatch), *fitted* (one ordered mapping per model fitted ONLY on the fit DIs by the Phase 4C objective, level excluded, frozen for all held-out DIs and levels).\n"]

# --- 2 training
out += ["## 2. Training budget and measured time\n", "| Model | wall-clock (concurrent) | optimiser updates (final checkpoint) | epochs run | val ESR (own teacher) | train audio (s) | audio per selected capture (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
STEPS = json.loads((P4E / 'train_steps.json').read_text()) if (P4E / 'train_steps.json').exists() else {}
for a, nm in AMPS.items():
    for cfg in "AB":
        for s in (0, 1):
            d = P4E / a / f"{cfg}_bundle" / f"{a}_P4E_{cfg}_s{s}"
            wp = P4E / "logs" / f"wall_{a}_{cfg}_s{s}.json"
            if not d.exists() or not wp.exists(): continue
            ti = json.loads((d / "train_info.json").read_text()); wall = json.loads(wp.read_text())["wall_seconds"]
            last = STEPS.get(d.name, {}).get("global_step")
            best = json.loads((d / "packed_best.json").read_text()) if (d / "packed_best.json").exists() else {}
            bm = json.loads((P4E / a / f"{cfg}_bundle" / "manifest.json").read_text()); ng = len(bm["gains"])
            out.append(f"| {nm} {cfg} seed {s} | {wall/60:.1f} min | {last} | {ti.get('epochs')} | {ti['validation_esr']:.4f} | {bm['train_seconds']:.0f} | {bm['train_seconds']/ng:.0f} |")
out += ["", "Validation ESR is measured against each configuration's own teacher and validation clip, so it is NOT comparable between A and B or with v3; the exported file is the stock best-validation checkpoint for every model. All eight models trained concurrently on one machine (MPS), so wall-clock is inflated relative to a solo run (v3 solo timings were about 17-24 minutes per model). Every configuration receives the same number of optimiser updates because the training audio has the same length; the number of captures only changes exposure per capture, which is a disclosed confound against the v3 C3/C5/C10 baselines (which have the same audio length: 467 s for every v3 model as well).\n"]

# --- 3 verification
vp = P4E / "verify_standard_nam.json"
out += ["## 3. Standard NAM verification\n"]
if vp.exists():
    v = json.loads(vp.read_text()); out.append(v["note"] + "\n")
    out += ["| Model | same structure as stock v3 export | extra metadata keys | finite over -48..+36 dB input gain | max raw peak (dBFS) | input gains (dB) where raw peak > 0 dBFS |", "|---|---|---|---|---:|---|"]
    for k, r in v["models"].items():
        out.append(f"| {k} | {r['structure_matches_v3_C3']} | {', '.join(r['metadata_keys_not_in_v3']) or 'none'} | {r['all_finite']} | {r['max_raw_peak_dbfs']:.1f} | {r['input_gain_range_with_raw_peak_over_0dbfs_db'] or 'none'} |")
    out.append("\nRaw model output is scaled by a single training constant `c` (the peak-ceiling gain applied to the targets, identical for every configuration of an amp); evaluation divides it back out exactly as v3 did. In a player this constant is the output-level setting, not hidden DSP.\n")
else:
    out.append("Not run yet.\n")

def gtab(a, key, variant):
    real, ev = DATA[a]; e = ev[key]
    trained = DIS_T.get(int(key.split("C")[1])) if key.startswith("v3_") else man["amps"][a]["selected_positions"]
    tab = M.model_table(real, e, variant)
    return tab, M.summarise(tab, real["gains"], e["regions"], [float(x) for x in trained])

def variant_of(key, kind): return "intended" if kind == "intended" else kind

def verdict(diffs, spread, tol):
    if all(d < 0 for d in diffs) and min(abs(d) for d in diffs) > spread: return "better" + (" (<tol)" if max(abs(d) for d in diffs) < tol else "")
    if all(d > 0 for d in diffs) and min(abs(d) for d in diffs) > spread: return "worse" + (" (<tol)" if max(abs(d) for d in diffs) < tol else "")
    return "inconclusive"

def cmp_table(a, base, news, vb, vn, title, paired=False):
    """base: list of model keys (1 for v3, 2 seeds for paired); news: 2 seed keys. Group value = mean over all positions at o=0."""
    rows = ["", f"**{title}** (mean over all positions, held-out DIs, DI level 0; verdict per the frozen seed rule)\n", "| Group | base | new s0 | new s1 | seed spread | verdict |", "|---|---:|---:|---:|---:|---|"]
    S = {k: gtab(a, k, vb if k in base else vn)[1]["all"] for k in set(base + news)}
    for grp in [x for x in SHOW if x != "level_signed"]:
        g = grp
        val = lambda k: S[k][g]["mean"]
        if paired:
            diffs = [val(n) - val(b) for n, b in zip(news, base)]; spread = max(abs(val(news[0]) - val(news[1])), abs(val(base[0]) - val(base[1])))
            b = np.mean([val(x) for x in base])
        else:
            diffs = [val(n) - val(base[0]) for n in news]; spread = abs(val(news[0]) - val(news[1])); b = val(base[0])
        rows.append(f"| {M.LABEL[g]} | {b:.2f} | {val(news[0]):.2f} | {val(news[1]):.2f} | {spread:.2f} | {verdict(diffs, spread, TOL.get(g, 0.2))} |")
    return rows

for a, nm in AMPS.items():
    real, ev = DATA[a]; gains = real["gains"]
    out += [f"## 4. {nm}\n"]
    if not new_ready:
        out.append("New-model evaluations are not available yet.\n"); continue
    reg = ev["A_s0"]["regions"]
    out.append("Regions (frozen): " + "; ".join(f"{r}: " + ", ".join(f"G{float(g):g}" for g, x in reg.items() if x == r) for r in ("low", "transition", "high", "plateau") if any(x == r for x in reg.values())) + ".\n")
    # 4.1 main tables
    for variant, label in (("intended", "intended mapping"), ("fixed", "fixed physical-position mapping"), ("fitted", "model-specific FITTED mapping (fit DIs only)")):
        out += [f"### {label}: mean (worst-position) error against the real capture, held-out DIs, DI level 0 dB, all positions\n",
                "| Model | " + " | ".join(M.LABEL[g] for g in SHOW) + " |", "|---|" + "---:|" * len(SHOW)]
        for k in M.MODELS:
            if k not in ev: continue
            vv = variant if (variant in M.variants(k, ev[k])) else None
            if variant == "intended" and not k.startswith("B"): vv = "fixed"
            if vv is None: continue
            s = gtab(a, k, vv)[1]["all"]
            cells = [(f"{s[g]['mean']:+.2f}" if g == "level_signed" else (fmtc(s[g]) if g != "esr" else f"{s[g]['mean']:.3f}")) for g in SHOW]
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out.append("")
    # regions under intended
    out += ["### By region (intended mapping): mean error per group\n", "| Model | region | " + " | ".join(M.LABEL[g] for g in ("level_abs", "eq", "hf", "thd", "io_slope", "di_intensity", "crest", "dyn", "transient")) + " |", "|---|---|" + "---:|" * 9]
    for k in M.MODELS:
        if k not in ev: continue
        vv = "intended" if k.startswith("B") else "fixed"; sm = gtab(a, k, vv)[1]
        for r in ("low", "transition", "high", "plateau"):
            if r in sm: out.append(f"| {k} | {r} | " + " | ".join(f"{sm[r][g]['mean']:.2f}" for g in ("level_abs", "eq", "hf", "thd", "io_slope", "di_intensity", "crest", "dyn", "transient")) + " |")
    out.append("")
    if any(g != int(g) for g in gains):
        out += ["### Independent JCM800 half-step captures only (intended mapping): mean error\n", "| Model | " + " | ".join(M.LABEL[g] for g in ("level_abs", "eq", "hf", "thd", "io_slope", "crest", "dyn", "transient")) + " |", "|---|" + "---:|" * 8]
        for k in M.MODELS:
            if k not in ev: continue
            sm = gtab(a, k, "intended" if k.startswith("B") else "fixed")[1]
            if "half_steps" in sm: out.append(f"| {k} | " + " | ".join(f"{sm['half_steps'][g]['mean']:.2f}" for g in ("level_abs", "eq", "hf", "thd", "io_slope", "crest", "dyn", "transient")) + " |")
        out.append("")
    # trained vs omitted
    out += ["### Trained versus omitted (neural-training holdout) integer positions, intended mapping: mean EQ / HF / THD / IO slope / crest / dyn error\n", "| Model | set | EQ | HF | THD | IO slope | crest | dyn |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k in M.MODELS:
        if k not in ev: continue
        sm = gtab(a, k, "intended" if k.startswith("B") else "fixed")[1]
        for st in ("trained", "omitted"):
            if st in sm: out.append(f"| {k} | {st} | " + " | ".join(f"{sm[st][g]['mean']:.2f}" for g in ("eq", "hf", "thd", "io_slope", "crest", "dyn")) + " |")
    if a == "vibrolux": out.append("\nVibrolux omitted integer positions were used in the 4D subset selection, so they are a neural-training holdout but NOT fully independent research validation.\n")
    else: out.append("")
    # DI-level regressions
    out += ["### DI-level dependence (intended mapping): mean error by DI offset\n", "| Model | DI offset | level |err| | EQ | HF | crest | dyn | transient |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in M.MODELS:
        if k not in ev: continue
        vv = "intended" if k.startswith("B") else "fixed"
        for o in M.OFFS:
            tab = M.model_table(real, ev[k], vv, o); s = M.summarise(tab, gains)["all"]
            out.append(f"| {k} | {o:+g} | " + " | ".join(f"{s[g]['mean']:.2f}" for g in ("level_abs", "eq", "hf", "crest", "dyn", "transient")) + " |")
    out.append("")
    # comparisons
    out += ["### Frozen comparisons\n"]
    out += cmp_table(a, ["v3_C3"], ["A_s0", "A_s1"], "fixed", "fixed", "A vs v3 C3 (both under the fixed anchor rule; capture SET differs, so this does not isolate placement)")
    out += cmp_table(a, ["A_s0", "A_s1"], ["B_s0", "B_s1"], "fixed", "intended", "B (intended response-distance mapping) vs A (intended fixed mapping): the training-anchor rule within the same captures", paired=True)
    out += cmp_table(a, ["A_s0", "A_s1"], ["B_s0", "B_s1"], "fitted", "fitted", "B vs A, both with their own FITTED playback mapping (training effect after equalising the playback fit)", paired=True)
    out += cmp_table(a, ["v3_C3"], ["B_s0", "B_s1"], "fixed", "intended", "ADDITIONAL (not one of the two frozen paired comparisons; same mechanical rule): B intended vs v3 C3 fixed")
    out += cmp_table(a, ["v3_C10"], ["B_s0", "B_s1"], "fixed", "intended", "ADDITIONAL (same rule): B intended vs v3 C10 fixed (the ten-capture single NAM)")
    out += ["", "**Remapping-only effect** (same model; mean error, intended mapping -> its own fitted mapping; fit on the fit DIs only; level excluded from the fit)\n", "| Model | " + " | ".join(M.LABEL[g] for g in ("level_signed", "level_abs", "eq", "hf", "thd", "io_slope", "di_intensity", "crest", "dyn", "transient")) + " |", "|---|" + "---:|" * 10]
    for k in M.MODELS:
        if k not in ev: continue
        v0 = "intended" if k.startswith("B") else "fixed"; s0_ = gtab(a, k, v0)[1]["all"]; s1_ = gtab(a, k, "fitted")[1]["all"]
        out.append(f"| {k} | " + " | ".join((f"{s0_[g]['mean']:+.2f} -> {s1_[g]['mean']:+.2f}" if g == "level_signed" else f"{s0_[g]['mean']:.2f} -> {s1_[g]['mean']:.2f}") for g in ("level_signed", "level_abs", "eq", "hf", "thd", "io_slope", "di_intensity", "crest", "dyn", "transient")) + " |")
    out.append("")
    # plots
    real_, ev_ = DATA[a]
    fig, axs = plt.subplots(2, 4, figsize=(19, 8)); g = np.array(gains)
    pan = [("Level (native RMS, dBFS, DI 0)", "rms_db"), ("EQ presence band (dB rel. total)", "eq_presence_db"), ("HF >3 kHz (dB rel.)", "hf3k_db"), ("Crest (dB)", "crest_db"), ("Dynamic range (dB)", "dyn_range_db"), ("Transient rise p95 (dB/5 ms)", "rise_p95_db")]
    for ax, (t, f) in zip(axs.ravel(), pan):
        ax.plot(g, [np.mean([real_["music"][f"{x:g}|{d}|0"][f] for d in M.HELD]) for x in gains], "k-", lw=2, label="real amp")
        for k, c in (("v3_C3", "tab:gray"), ("A_s0", "tab:blue"), ("B_s0", "tab:red")):
            e = ev_[k]; vv = "intended" if k.startswith("B") else "fixed"
            r = e["results"]["fixed" if e["results"].get("intended") == "same as fixed" and vv == "intended" else vv]["music"]
            ax.plot(g, [np.mean([r[f"{x:g}|{d}|0"][f] for d in M.HELD]) for x in gains], "-o", ms=3, color=c, label=k)
        ax.set_title(t, fontsize=9); ax.set_xlabel("physical gain position", fontsize=8); ax.grid(alpha=.25); ax.legend(fontsize=6)
    ax = axs[1, 2]
    for x0, c in ((2, "tab:green"), (7, "tab:purple")):
        ax.plot(M.OFFS, [np.mean([real_["music"][f"{x0:g}|{d}|{o:g}"]["rms_db"] for d in M.HELD]) for o in M.OFFS], "-", color=c, lw=2, label=f"real G{x0}")
        for k, ls in (("v3_C3", ":"), ("A_s0", "--"), ("B_s0", "-.")):
            e = ev_[k]; vv = "intended" if k.startswith("B") else "fixed"
            r = e["results"]["fixed" if e["results"].get("intended") == "same as fixed" and vv == "intended" else vv]["music"]
            ax.plot(M.OFFS, [np.mean([r[f"{x0:g}|{d}|{o:g}"]["rms_db"] for d in M.HELD]) for o in M.OFFS], ls, color=c, alpha=.7, label=f"{k} G{x0}")
    ax.set_title("Output level vs DI offset (compression), G2 and G7", fontsize=9); ax.set_xlabel("DI offset dB", fontsize=8); ax.legend(fontsize=5); ax.grid(alpha=.25)
    ax = axs[1, 3]
    ax.plot(g, [fixed for fixed in ev_["A_s0"]["mappings"]["fixed"]], "b--", label="fixed rule"); ax.plot(g, ev_["B_s0"]["mappings"]["intended"], "r-", label="B intended (response distance)")
    ax.plot(g, ev_["A_s0"]["mappings"]["fitted"], "c:", label="A fitted"); ax.plot(g, ev_["B_s0"]["mappings"]["fitted"], "m:", label="B fitted")
    ax.set_title("Playback input gain by physical position (dB)", fontsize=9); ax.legend(fontsize=6); ax.grid(alpha=.25)
    fig.suptitle(f"{AMPS[a]}: real amp vs v3 C3 (fixed), A seed 0 (fixed), B seed 0 (intended); held-out DIs", fontsize=10); fig.tight_layout()
    (ROOT / "docs" / "phase4e").mkdir(exist_ok=True); fig.savefig(ROOT / "docs" / "phase4e" / f"curves_{a}.png", dpi=105); plt.close(fig)
    out.append(f"![{a} curves](phase4e/curves_{a}.png)\n")
gt = P4E / "_timing"
if (gt / "stages.json").exists():
    st = json.loads((gt / "stages.json").read_text()); solo = json.loads((gt / "wall_solo.json").read_text()) if (gt / "wall_solo.json").exists() else None
    vt = json.loads((gt / "verify_time.json").read_text()) if (gt / "verify_time.json").exists() else None
    walls = [json.loads(p_.read_text())["wall_seconds"] for p_ in (P4E / "logs").glob("wall_*.json")]
    rows = ["", "## 6. Generation time (Super-Sonic Vibrolux, 10 captures, one amp, solo on a quiet machine)\n", "| Stage | seconds |", "|---|---:|"]
    tot = 0.0
    for k, v in st.items(): rows.append(f"| {k} | {v:.0f} |"); tot += v
    if vt: rows.append(f"| standard-NAM export check and NAMCore inference (per model) | {vt['per_model_seconds']:.0f} |"); tot += vt["per_model_seconds"]
    if solo: rows.append(f"| **training, one model, 60 epochs, running alone (includes the trainer's export)** | **{solo['wall_seconds']:.0f}** ({solo['wall_seconds']/60:.1f} min) |"); tot += solo["wall_seconds"]
    rows.append(f"| **total for one amp, one model** | **{tot:.0f}** ({tot/60:.1f} min) |")
    rows.append("")
    rows.append(f"For the pilot the eight models trained concurrently in {max(walls)/60:.0f} minutes of wall-clock ({max(walls)/8/60:.1f} minutes of machine time per model when eight share the GPU), so the solo figure is the user-relevant one for a single amp. QA probing scales with the number of captures (about {st.get('QA probe, 10 captures (p4_probe_captures)', 0)/10:.0f} s per capture); everything before training is under about {sum(st.values())/60:.0f} minutes. Evaluation (sweeps, mapping fits, direct renders for every position and DI level) is research cost, not part of a user workflow.")
    tj = gt / "vibrolux_timing_solo_train_info.json"
    if tj.exists():
        ti = json.loads(tj.read_text()); ref = json.loads((P4E / "vibrolux" / "B_bundle" / "vibrolux_P4E_B_s0" / "train_info.json").read_text())
        rows.append(f"\n**Repeatability datapoint:** the solo timing run used the same bundle and seed 0 as the concurrent B seed-0 model; final validation ESR {ti['validation_esr']:.4f} vs {ref['validation_esr']:.4f} for the concurrent run (same seed, different concurrency). This is one pair, not a determinism proof.")
    GEN_ROWS = rows
else:
    GEN_ROWS = []
fp = ROOT / "docs" / "phase4e" / "findings.md"
out.append(fp.read_text() if fp.exists() else "## Findings\n\n(pending)\n")
out += GEN_ROWS
(ROOT / "docs" / "CONTINUOUS_GAIN_PHASE4E_RESULTS.md").write_text("\n".join(out)); print("report written", len(out), "lines; new models ready:", new_ready)
