"""Final-candidate evaluation: applies the FROZEN criteria (docs/final/manifest_frozen.json) mechanically, applies the selection rule, writes docs/CONTINUOUS_GAIN_FINAL_CANDIDATES.md
(+ docs/final/*.png) and work/p4e/final/selection.json. Usage: python scripts/fc_report.py"""
import glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4e_metrics as M
import pl_load as PLL
ROOT = Path(__file__).resolve().parent.parent; P4E = ROOT / "work" / "p4e"; FC = P4E / "final"; DOC = ROOT / "docs"; IMG = DOC / "final"
FROZEN = json.loads((IMG / "manifest_frozen.json").read_text()); CRIT = FROZEN["criteria"]
AMPS = {"jcm800": "Marshall JCM800 2203 (High)", "vibrolux": "Fender Super-Sonic Vibrolux"}
KEYS = ["v3_C3", "B_s0", "B_s1", "FC_s0", "FC_s1"]; HELD = M.HELD; OFFS = M.OFFS
ANCH_B = {"jcm800": [1.0, 2.0, 4.0, 10.0], "vibrolux": [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]}
def load_eval(amp, k):
    p = (FC / amp / f"eval_{k}.json") if k.startswith("FC_") else (P4E / amp / f"eval_{k}.json"); return json.loads(p.read_text())
def T_of(amp, k, gains):
    e = load_eval(amp, k); return e["mappings"]["intended"] if k.startswith("FC_") else (e["mappings"]["intended"] if k.startswith("B_") else e["mappings"]["fixed"])
R = {}; out = ["# Continuous Gain: the best practical single NAM for each amp (final candidates)\n"]
sel_all = {}
for amp, nm in AMPS.items():
    real = json.loads((P4E / amp / "real_ref.json").read_text()); gains = real["gains"]; ev = {k: load_eval(amp, k) for k in KEYS}
    tabs = {k: M.model_table(real, ev[k], "intended") for k in KEYS}
    def mean_abs(k, f): return float(np.mean([abs(tabs[k][g][f]) for g in gains]))
    d = {"amp": amp, "tabs": {k: {f"{g:g}": tabs[k][g] for g in gains} for k in KEYS}}
    # ---- C2 (native level)
    c2 = {}
    for k in KEYS: c2[k] = {"level": mean_abs(k, "level_abs"), "HF": mean_abs(k, "hf"), "crest": mean_abs(k, "crest"), "EQ": mean_abs(k, "eq"), "dyn": mean_abs(k, "dyn"), "worst_level": max(abs(tabs[k][g]["level_abs"]) for g in gains), "esr": float(np.mean([tabs[k][g]["esr"] for g in gains if "esr" in tabs[k][g]]))}
    d["c2_values"] = c2
    bm = {m: np.mean([c2["B_s0"][m], c2["B_s1"][m]]) for m in c2["B_s0"]}
    tol = {"level": 0.3, "HF": 0.3, "crest": 0.3, "EQ": 0.3, "dyn": 0.5}
    d["C2"] = {k: all(c2[k][m] <= bm[m] + tol[m] for m in tol) and c2[k]["worst_level"] <= 1.5 for k in ("FC_s0", "FC_s1")}
    d["C4"] = all(abs(c2["FC_s0"][m] - c2["FC_s1"][m]) <= max(0.3, abs(c2["B_s0"][m] - c2["B_s1"][m])) for m in ("level", "HF", "crest", "EQ", "dyn"))
    # ---- C1 and C3 from the fixed-setting cases (same positions for B and FC)
    fcc = []
    for f in sorted(glob.glob(str(FC / amp / "cases_fc_*.json"))): fcc.append(json.loads(Path(f).read_text()))
    bc = PLL.load_cases(amp)
    def esr_hi(cases, keyfmt, get_total):
        v = [np.mean([c["esr"][keyfmt.format(s)] for s in ("s0", "s1")]) for c in cases if get_total(c) >= 8.0]; return float(np.mean(v)) if v else float("nan")
    def esr_hi_seed(cases, seed, tot):
        v = [c["esr"][seed] for c in cases if tot(c) >= 8.0]; return float(np.mean(v)) if v else float("nan")
    fc_cases = [dict(c, g=j["g"]) for j in fcc for c in j["cases"]]
    b_hi = np.mean([esr_hi_seed(bc, f"B_{s}_vs_teacher", lambda c: c["T"]["B"] + c["offset"]) for s in ("s0", "s1")])
    fc_hi = {s: esr_hi_seed(fc_cases, f"FC_{s}_vs_teacher", lambda c: c["T"] + c["offset"]) for s in ("s0", "s1")}
    d["C1"] = {"b_mean_esr_total_ge_8": float(b_hi), "fc": fc_hi, "pass": {f"FC_{s}": bool(fc_hi[s] <= 0.6 * b_hi) for s in fc_hi}}
    bins = [(-40, -16), (-16, -8), (-8, 0), (0, 8), (8, 14), (14, 20), (20, 30)]
    def by_bin(cases, keyf, tot): return {f"{b[0]}..{b[1]}": (float(np.mean([keyf(c) for c in cases if b[0] <= tot(c) < b[1]])) if any(b[0] <= tot(c) < b[1] for c in cases) else None) for b in bins}
    d["esr_by_bin"] = {"B": by_bin(bc, lambda c: np.mean([c["esr"]["B_s0_vs_teacher"], c["esr"]["B_s1_vs_teacher"]]), lambda c: c["T"]["B"] + c["offset"]), "FC": by_bin(fc_cases, lambda c: np.mean([c["esr"]["FC_s0_vs_teacher"], c["esr"]["FC_s1_vs_teacher"]]), lambda c: c["T"] + c["offset"])}
    THR = {"level": ("rms_db", 1.0), "HF": ("hf3k_db", 1.0), "crest": ("crest_db", 1.0), "dyn": ("dyn_range_db", 1.5)}
    def flagged(cases, getter, keys):
        res = {}
        for k in keys:
            n = 0; gs = sorted({c["g"] for c in cases})
            for g in gs:
                for lab, (f, thr) in THR.items():
                    se = {o: float(np.mean([getter(c, k)[f] - c["real"][f] for c in cases if c["g"] == g and c["offset"] == o])) for o in OFFS}
                    if max(abs(np.mean(list(se.values()))), abs(se[6.0] - se[-12.0])) >= thr: n += 1
            res[k] = n
        return res
    bfl = flagged(bc, lambda c, k: c["student"][k], ["B_s0", "B_s1", "v3_C3"]); ffl = flagged(fc_cases, lambda c, k: c["student"][k], ["FC_s0", "FC_s1"]) if fc_cases else {}
    d["C3"] = {"flagged": {**bfl, **ffl}, "pass": {k: (ffl[k] <= np.mean([bfl["B_s0"], bfl["B_s1"]]) + 2) for k in ffl}}
    d["n_combinations"] = len({c["g"] for c in bc}) * 4
    # ---- sweeps: C6
    real_T = {}
    sw = {k: json.loads((FC / amp / f"sweep_{k}.json").read_text()) for k in KEYS}
    ratio, opp = {}, {}
    for k in KEYS:
        s = sw[k]; T = np.array(s["T"]); m = (T >= -20) & (T <= 14); Tg = np.array(T_of(amp, k, gains)); r_ = {}
        o_ = 0
        for f in ("rms_db", "hf3k_db", "crest_db", "dyn_range_db"):
            y = np.mean([[r[f] for r in s["feats"][dd]] for dd in HELD], axis=0); st = np.abs(np.diff(y[m])); r_[f] = float(st.max() / max(np.median(st), 0.05))
            mv = np.interp(Tg, T, y); rv = np.array([np.mean([real["music"][f"{g:g}|{dd}|0"][f] for dd in HELD]) for g in gains]); o = np.argsort(Tg); dm, dr = np.diff(mv[o]), np.diff(rv[o]); sel = np.abs(dr) >= 0.5
            o_ += int(((np.sign(dm) != np.sign(dr)) & sel & (np.abs(dm) >= 0.3)).sum())
        ratio[k] = max(r_.values()); opp[k] = o_
    d["C6"] = {"max_step_ratio": ratio, "opposite_steps": opp, "pass": {k: (opp[k] == 0 and ratio[k] <= 1.5 * np.mean([ratio["B_s0"], ratio["B_s1"]])) for k in ("FC_s0", "FC_s1")}}
    # ---- C5
    ver = json.loads((FC / "verify_fc.json").read_text())["models"]; d["C5"] = {k: bool(ver[f"{amp}_{k}"]["structure_matches_stock_export"] and ver[f"{amp}_{k}"]["finite_over_-20_to_+20"] and not ver[f"{amp}_{k}"]["extra_metadata_keys"]) for k in ("FC_s0", "FC_s1")}
    # ---- selection rule
    allpass = {k: bool(d["C1"]["pass"][k] and d["C2"][k] and d["C3"]["pass"][k] and d["C5"][k] and d["C6"]["pass"][k]) for k in ("FC_s0", "FC_s1")}; allpass_both = bool(all(allpass.values()) and d["C4"])
    if allpass_both: chosen = min(("FC_s0", "FC_s1"), key=lambda k: c2[k]["esr"]); why = "C1-C6 pass for both FC seeds; the seed with the lower mean level-matched ESR is recommended"
    else: chosen = min(("B_s0", "B_s1"), key=lambda k: c2[k]["esr"]); why = "the frozen criteria are not all met by both FC seeds, so the B seed with the lower mean level-matched ESR is recommended"
    d["allpass"] = allpass; d["allpass_both_and_C4"] = allpass_both; d["chosen"] = chosen; d["why"] = why; R[amp] = d
    sel_all[amp] = {"chosen": chosen, "why": why, "criteria": {"C1": d["C1"], "C2": d["C2"], "C3": d["C3"], "C4": d["C4"], "C5": d["C5"], "C6": d["C6"]}}
    # ---- tables
    out += [f"## {nm}\n", "### Coverage of distinct sounds: error at every physical position, native musical level (each model at its own intended Input gain; mean of seeds for B and FC)\n",
            "| Position | v3 C3 level / EQ / HF / crest / dyn | B | FC |", "|---|---|---|---|"]
    avg = lambda ks, g, f: float(np.mean([tabs[k][g][f] for k in ks])); fm = lambda ks, g: " / ".join(f"{avg(ks, g, f):+.1f}" if f == "level_signed" else f"{avg(ks, g, f):.1f}" for f in ("level_signed", "eq", "hf", "crest", "dyn"))
    for g in gains: out.append(f"| G{g:g} | {fm(['v3_C3'], g)} | {fm(['B_s0', 'B_s1'], g)} | {fm(['FC_s0', 'FC_s1'], g)} |")
    out += ["", "Level is signed (model minus real, dB); EQ, HF, crest and dynamic range are mean absolute errors in dB.\n", "### Summary (mean over positions; worst position level)\n", "| Model | level | EQ | HF | crest | dyn range | worst-position level | mean lm-ESR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in KEYS: out.append(f"| {k} | {c2[k]['level']:.2f} | {c2[k]['EQ']:.2f} | {c2[k]['HF']:.2f} | {c2[k]['crest']:.2f} | {c2[k]['dyn']:.2f} | {c2[k]['worst_level']:.2f} | {c2[k]['esr']:.3f} |")
    out += ["", "### Trained model versus its own teacher, by total level reaching the NAM (level-matched ESR; the targeted weakness)\n", "| Total level (dB re native) | " + " | ".join(d["esr_by_bin"]["B"].keys()) + " |", "|---|" + "---:|" * len(bins)]
    for lab, key in (("B (mean of seeds)", "B"), ("FC (mean of seeds)", "FC")): out.append(f"| {lab} | " + " | ".join(("-" if v is None else f"{v:.3f}") for v in d["esr_by_bin"][key].values()) + " |")
    out += ["", "### Playing intensity at fixed settings and the frozen criteria\n", "| Criterion | result |", "|---|---|",
            f"| C1: student-vs-teacher ESR at total level >= +8 dB (B mean {d['C1']['b_mean_esr_total_ge_8']:.3f}; need FC <= 60% of it) | FC seed 0 {d['C1']['fc']['s0']:.3f} ({'pass' if d['C1']['pass']['FC_s0'] else 'FAIL'}), FC seed 1 {d['C1']['fc']['s1']:.3f} ({'pass' if d['C1']['pass']['FC_s1'] else 'FAIL'}) |",
            f"| C2: native-level errors not worse than B by more than 0.3 dB (dyn 0.5), worst level <= 1.5 dB | FC seed 0 {'pass' if d['C2']['FC_s0'] else 'FAIL'}, FC seed 1 {'pass' if d['C2']['FC_s1'] else 'FAIL'} |",
            f"| C3: flagged soft-to-hard combinations of {d['n_combinations']} (B seeds {bfl['B_s0']}/{bfl['B_s1']}, C3 {bfl['v3_C3']}; FC allowed B mean + 2) | FC seed 0 {ffl.get('FC_s0', '-')}, seed 1 {ffl.get('FC_s1', '-')} ({'pass' if all(d['C3']['pass'].values()) else 'FAIL'}) |",
            f"| C4: seed reproducibility of the FC seeds | {'pass' if d['C4'] else 'FAIL'} |", f"| C5: range and format (anchors in [-20, +14], stock structure, finite -20..+20) | {'pass' if all(d['C5'].values()) else 'FAIL'} |",
            f"| C6: sweep (opposite steps: FC {opp['FC_s0']}/{opp['FC_s1']}; step ratio FC {ratio['FC_s0']:.1f}/{ratio['FC_s1']:.1f} vs B {ratio['B_s0']:.1f}/{ratio['B_s1']:.1f}) | {'pass' if all(d['C6']['pass'].values()) else 'FAIL'} |", "",
            f"**Selection rule result: {chosen}** ({why}).\n"]
    # extremes
    ex = json.loads((FC / amp / "extremes.json").read_text())["rows"]
    out += ["### Extremes of the practical Input range (-20 and +20 dB; mean over soft/normal/hard and two DIs; error against the physically equivalent real reference)\n", "| Model | Input gain | level err | HF err | crest err | dyn err | max raw peak (dBFS) | finite |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for k in KEYS:
        for T_ in (-20.0, 20.0):
            rr = [r for r in ex if r["model"] == k and r["input_gain_db"] == T_]; out.append(f"| {k} | {T_:+.0f} | {np.mean([r['level_err'] for r in rr]):+.1f} | {np.mean([r['hf_err'] for r in rr]):+.1f} | {np.mean([r['crest_err'] for r in rr]):+.1f} | {np.mean([r['dyn_err'] for r in rr]):+.1f} | {max(r['raw_peak_dbfs'] for r in rr):.1f} | {all(r['finite'] for r in rr)} |")
    out.append("")
    # plots: sweeps
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2)); Tb = T_of(amp, "B_s0", gains); Tf = T_of(amp, "FC_s0", gains)
    for ax, (f, ttl) in zip(axs, (("rms_db", "level (dBFS)"), ("hf3k_db", "HF>3k (dB rel.)"), ("crest_db", "crest (dB)"), ("dyn_range_db", "dynamic range (dB)"))):
        for k, col in (("v3_C3", "tab:gray"), ("B_s0", "tab:blue"), ("B_s1", "tab:cyan"), ("FC_s0", "tab:red"), ("FC_s1", "tab:orange")):
            s = sw[k]; ax.plot(s["T"], np.mean([[r[f] for r in s["feats"][dd]] for dd in HELD], axis=0), color=col, lw=1.2, label=k)
        rv = [np.mean([real["music"][f"{g:g}|{dd}|0"][f] for dd in HELD]) for g in gains]; ax.plot(Tf, rv, "ks", ms=4, label="real captures at FC gains"); ax.plot(Tb, rv, "x", color="gray", ms=4, label="real captures at B gains")
        ax.set_xlim(-24, 20); ax.set_title(f"{nm.split(' (')[0]}: {ttl} vs Input gain", fontsize=9); ax.set_xlabel("player Input gain (dB)"); ax.grid(alpha=.25); ax.legend(fontsize=5)
    fig.tight_layout(); fig.savefig(IMG / f"final_sweep_{amp}.png", dpi=100); plt.close(fig); out.append(f"![sweep {amp}](final/final_sweep_{amp}.png)\n")
(FC / "selection.json").write_text(json.dumps(sel_all, indent=1, default=float)); (FC / "report_data.json").write_text(json.dumps(R, indent=1, default=float))
fp = IMG / "decision_and_packaging.md"; body = (IMG / "assessment.md").read_text() + "\n" + "\n".join(out) + "\n" + (fp.read_text() if fp.exists() else "")
(DOC / "CONTINUOUS_GAIN_FINAL_CANDIDATES.md").write_text("# Continuous Gain: the best practical single NAM for each amp (final candidates)\n\n" + body if not body.startswith("# ") else body); print("written; chosen:", {a: v["chosen"] for a, v in sel_all.items()})
