"""Expansion report: per-amp complete-amplifier evaluation of the first-pass FC NAM against the source captures and the single-middle-capture Input-gain baseline (BASE),
plus the cross-amp generalisation table. Triage criteria R1-R7 are DECLARED HERE, before any result existed (they are a triage aid, not a quality score, and are not tuned per amp).
Usage: p5_report.py [amp ...] -> docs/CONTINUOUS_GAIN_EXPANSION.md, docs/expand/*.png, work/p4e/final/expansion_report.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = "twin"
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fc_common import *
import p4e_metrics as M
DOC = REPO / "docs"; IMG = DOC / "expand"; IMG.mkdir(parents=True, exist_ok=True)
FROZEN = json.loads((IMG / "manifest_frozen.json").read_text()); AMPS = sys.argv[1:] or list(FROZEN["amps"])
THR = {"level": ("rms_db", 1.0), "HF": ("hf3k_db", 1.0), "crest": ("crest_db", 1.0), "dyn": ("dyn_range_db", 1.5)}
PICK = {"soft": -12.0, "normal": 0.0, "hard": 6.0}
REGION_NAMES = ["bottom of the response range", "lower-middle", "upper-middle", "top of the response range"]
def wall(amp):
    p = FCDIR / "logs" / f"wall_{amp}_FC_s0.json"; return json.loads(p.read_text())["wall_seconds"] if p.exists() else None
R = {}; md = []
for amp in AMPS:
    os.environ["SINGLE_NAM_AMP"] = amp; F = FROZEN["amps"][amp]; real = json.loads((P4E / amp / "real_ref.json").read_text()); gains = real["gains"]; sel = F["selected_positions"]
    ev = {k: json.loads((FCDIR / amp / f"eval_{k}.json").read_text()) for k in ("FC_s0", "BASE")}; T = {k: dict(zip(gains, ev[k]["mappings"]["intended"])) for k in ev}
    tab = {k: M.model_table(real, ev[k], "intended") for k in ev}; ag, arc = response_arc(amp); at = lambda x: float(np.interp(x, ag, arc)); el = sorted(float(k) for k, v in F["statuses"].items() if v in ("VALID", "CORRECTED"))
    lo, hi = at(el[0]), at(el[-1]); reg_of = lambda g: min(3, max(0, int(4 * (at(g) - lo) / (hi - lo + 1e-9)))) if lo <= at(g) <= hi else (0 if at(g) < lo else 3)
    regions = {i: [g for g in gains if reg_of(g) == i] for i in range(4)}; d = {"amp": amp, "selected": sel, "mid_capture": base_capture(amp), "statuses": F["statuses"], "T_fc": {f"{g:g}": T["FC_s0"][g] for g in gains}, "T_base": {f"{g:g}": T["BASE"][g] for g in gains}}
    # native accuracy per region
    reg = {}
    for i, gs in regions.items():
        if not gs: continue
        reg[REGION_NAMES[i]] = {"positions": gs}
        for k in tab:
            reg[REGION_NAMES[i]][k] = {m: float(np.mean([tab[k][g][m] for g in gs])) for m in ("level_abs", "eq", "hf", "crest", "dyn", "thd", "esr")} | {"level_signed": float(np.mean([tab[k][g]["level_signed"] for g in gs])), "worst_level": float(max(tab[k][g]["level_abs"] for g in gs))}
    d["regions"] = reg
    # per position table and worst-position
    d["per_position"] = {k: {f"{g:g}": tab[k][g] for g in gains} for k in tab}
    okv = lambda k, m, gs: float(np.mean([tab[k][g][m] for g in gs])); allg = gains
    d["means"] = {k: {m: okv(k, m, allg) for m in ("level_abs", "eq", "hf", "crest", "dyn", "thd", "esr")} | {"worst_level": float(max(tab[k][g]["level_abs"] for g in allg)), "worst_hf": float(max(tab[k][g]["hf"] for g in allg))} for k in tab}
    # soft/normal/hard by pick and flags over all position x DI x offset combinations
    def errs(k, g, dd, o):
        y = ev[k]["results"]["intended"]["music"][f"{g:g}|{dd}|{o:g}"]; r = real["music"][f"{g:g}|{dd}|{o:g}"]; return {n: y[f] - r[f] for n, (f, t) in THR.items()}
    pk = {k: {p: {n: float(np.mean([abs(errs(k, g, dd, o)[n]) for g in gains for dd in HELD])) for n in THR} for p, o in PICK.items()} for k in ev}; d["pick_abs_err"] = pk
    fl = {k: sum(1 for g in gains for dd in HELD for o in OFFS if any(abs(errs(k, g, dd, o)[n]) >= THR[n][1] for n in THR)) for k in ev}; ncomb = len(gains) * len(HELD) * len(OFFS); d["flagged"] = {"combos": ncomb, **fl}
    # sweep progression and accessibility
    sw = {k: json.loads((FCDIR / amp / f"sweep_{k}.json").read_text()) for k in ev}; Tg = np.array(sw["FC_s0"]["T"], float); m = (Tg >= -20) & (Tg <= 14); prog = {}
    def sw_feat(k, f): return np.mean([[r[f] for r in sw[k]["feats"][dd]] for dd in HELD], axis=0)
    for k in ev:
        opp = 0; ratios = []; T_ = np.array([T[k][g] for g in gains])
        for name, (f, thr) in THR.items():
            y = sw_feat(k, f); st = np.abs(np.diff(y[m])); ratios.append(float(st.max() / max(np.median(st), 0.05)))
            mv = np.interp(T_, Tg, y); rv = np.array([np.mean([real["music"][f"{g:g}|{dd}|0"][f] for dd in HELD]) for g in gains]); o_ = np.argsort(T_); dm, dr = np.diff(mv[o_]), np.diff(rv[o_]); selm = np.abs(dr) >= 0.5; opp += int(((np.sign(dm) != np.sign(dr)) & selm & (np.abs(dm) >= 0.3)).sum())
        lev = sw_feat(k, "rms_db")[m]; prog[k] = {"opposite_direction_steps": opp, "max_step_ratio": float(max(ratios)), "level_decreases_gt_0p3dB": int((np.diff(lev) < -0.3).sum()), "level_range_db": float(lev.max() - lev.min())}
    d["progression"] = prog; acc = {}
    for k in ev:
        acc[k] = {}
        for g in gains:
            rv = {n: np.mean([real["music"][f"{g:g}|{dd}|0"][f] for dd in HELD]) for n, (f, t) in THR.items()}; dist = np.max(np.abs(np.array([[sw_feat(k, THR[n][0]) - rv[n]] for n in THR])[:, 0, :]) / np.array([THR[n][1] for n in THR])[:, None], axis=0)
            Tj = np.where((Tg >= -20) & (Tg <= 14))[0]; jb = Tj[np.argmin(dist[Tj])]; acc[k][f"{g:g}"] = {"best_T": float(Tg[jb]), "min_dist": float(dist[jb]), "accessible": bool(dist[jb] <= 1.0)}
    d["accessibility"] = acc
    # student vs teacher (drift), from cases
    cs = [json.loads(p.read_text()) for p in sorted((FCDIR / amp).glob("cases_fc_*.json"))]; rows = [(c["T"] + x["offset"], x["esr"]["FC_s0_vs_teacher"]) for c in cs for x in c["cases"]]
    d["esr_vs_teacher"] = {"total_lt0": float(np.mean([e for t, e in rows if t < 0])) if any(t < 0 for t, e in rows) else None, "total_ge8": float(np.mean([e for t, e in rows if t >= 8])) if any(t >= 8 for t, e in rows) else None, "n": len(rows)}
    ex = json.loads((FCDIR / amp / "extremes.json").read_text())["rows"]; d["extremes"] = {k: {f"{T_:+.0f}": {"level_err": float(np.mean([r["level_err"] for r in ex if r["model"] == k and r["input_gain_db"] == T_])), "hf_err": float(np.mean([r["hf_err"] for r in ex if r["model"] == k and r["input_gain_db"] == T_])),
        "crest_err": float(np.mean([r["crest_err"] for r in ex if r["model"] == k and r["input_gain_db"] == T_])), "dyn_err": float(np.mean([r["dyn_err"] for r in ex if r["model"] == k and r["input_gain_db"] == T_])), "peak_dbfs": float(max(r["raw_peak_dbfs"] for r in ex if r["model"] == k and r["input_gain_db"] == T_)), "finite": all(r["finite"] for r in ex if r["model"] == k and r["input_gain_db"] == T_)} for T_ in (-20.0, 20.0)} for k in ("FC_s0", "BASE")}
    vf = json.loads((FCDIR / amp / "verify.json").read_text()); d["verify"] = vf; d["wall_seconds"] = wall(amp)
    # triage criteria (declared above)
    mn = d["means"]["FC_s0"]; e = d["esr_vs_teacher"]
    R1 = vf["structure_matches_stock_export"] and vf["finite_over_-20_to_+20"] and not vf["extra_metadata_keys"]; R2 = mn["worst_level"] <= 1.5; R3 = mn["hf"] <= 1.5 and mn["crest"] <= 1.5 and mn["dyn"] <= 2.0 and mn["eq"] <= 1.0
    R4 = prog["FC_s0"]["opposite_direction_steps"] == 0; R5 = np.mean([v["accessible"] for v in acc["FC_s0"].values()]) >= 0.8; R6 = fl["FC_s0"] <= 0.4 * ncomb
    R7 = (e["total_ge8"] is None or e["total_lt0"] is None) or e["total_ge8"] <= 2 * e["total_lt0"]
    d["triage"] = {"R1_export_valid": bool(R1), "R2_worst_level_le_1.5dB": bool(R2), "R3_mean_tone_within_limits": bool(R3), "R4_no_opposite_sweep_steps": bool(R4), "R5_ge80pct_positions_accessible": bool(R5), "R6_flagged_combos_le_40pct": bool(R6), "R7_no_high_level_drift": bool(R7)}
    core = R1 and R2 and R4 and R5; extra = sum([R3, R6, R7]); d["first_pass_verdict"] = "useful first pass" if core and extra >= 2 else ("usable with important weaknesses" if core else "technical problem")
    # plots
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, (n, (f, t)) in zip(axs, THR.items()):
        for k, col in (("BASE", "tab:gray"), ("FC_s0", "tab:red")): ax.plot(Tg, sw_feat(k, f), color=col, label=k + " sweep")
        ax.plot([T["FC_s0"][g] for g in gains], [np.mean([real["music"][f"{g:g}|{dd}|0"][f] for dd in HELD]) for g in gains], "ks", ms=4, label="real captures at FC gains")
        for g in sel: ax.axvline(T["FC_s0"][g], color="tab:red", alpha=.15)
        ax.set_xlim(-24, 20); ax.set_title(f"{amp}: {n} ({f}) vs Input gain", fontsize=9); ax.set_xlabel("player Input gain (dB)"); ax.grid(alpha=.25); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(IMG / f"sweep_{amp}.png", dpi=100); plt.close(fig); R[amp] = d
(FCDIR / "expansion_report.json").write_text(json.dumps(R, indent=1, default=float))
# ---------------- markdown
o = ["# Continuous Gain: expansion to the remaining amplifiers (first pass)\n", f"Branch `research/continuous-gain-model`. Configuration frozen before any build or training in `docs/expand/manifest_frozen.json` (commit `{FROZEN['code_commit_at_freeze'][:7]}` + the freeze commit that added it). Recipe unchanged from the JCM800/Vibrolux FC models (`docs/final/manifest_frozen.json`): official NAM input plus three guitar DIs at -32..+20 dB, A2 architecture, 60 epochs, ONE seed. The JCM800 and Vibrolux models were not touched.\n"]
o += ["## Cross-amp summary\n", "Measured technical results only. **Nothing here has been listened to.** Triage criteria (declared in `scripts/p5_report.py` before any result existed, not tuned per amp): R1 valid standard export and finite over -20..+20 dB; R2 worst-position native level error <= 1.5 dB; R3 mean HF/crest <= 1.5 dB, dyn <= 2.0, EQ <= 1.0; R4 no opposite-direction sweep steps; R5 at least 80% of real positions reachable within working tolerances at some Input gain in -20..+14; R6 flagged combinations <= 40%; R7 student-vs-teacher ESR at total level >= +8 dB no more than 2x the ESR below 0 dB. Verdict: useful first pass = R1, R2, R4, R5 and at least two of R3, R6, R7.\n",
      "| Amp | Selected captures | Anchors (dB) | Train time (min) | Output gain (dB) | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Verdict |", "|---|---|---|---:|---:|---|---|---|---|---|---|---|---|"]
tick = lambda b: "pass" if b else "FAIL"
for amp, d in R.items():
    F = FROZEN["amps"][amp]; t = d["triage"]; o.append(f"| {F['name']} | " + ", ".join(f"G{g:g}" for g in d["selected"]) + " | " + ", ".join(f"{a:+.1f}" for a in F["anchors_input_gain_db"]) + f" | {(d['wall_seconds'] or 0) / 60:.0f} | {d['verify']['output_gain_compensation_db']:+.1f} | " + " | ".join(tick(v) for v in t.values()) + f" | **{d['first_pass_verdict']}** |")
o.append("")
for amp, d in R.items():
    F = FROZEN["amps"][amp]; mn, mb = d["means"]["FC_s0"], d["means"]["BASE"]
    o += [f"## {F['name']}\n", f"Source: `{F['capture_dir']}`. Positions available {', '.join(f'G{g:g}' for g in F['positions_available'])}; audit statuses: " + ", ".join(f"G{k}={v}" for k, v in F["statuses"].items()) + f". Selected: {F['selection_basis']}. Middle-capture baseline: G{d['mid_capture']:g} (eligible capture at the middle of the response range), Input gain chosen per position on the fit DIs only (an upper bound for 'just change one capture's input').\n",
          "![sweep](expand/sweep_" + amp + ".png)\n", "### Native accuracy at each position's intended Input gain (held-out DIs; positions not selected for training are omitted positions, SUSPECT ones analysis-only)\n",
          "| Position | Role | Input gain FC (dB) | FC level (signed) | FC EQ | FC HF | FC crest | FC dyn | BASE Input gain | BASE level (signed) | BASE EQ | BASE HF | BASE crest | BASE dyn |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    real = json.loads((P4E / amp / "real_ref.json").read_text())
    for g in real["gains"]:
        a, b = d["per_position"]["FC_s0"][f"{g:g}"], d["per_position"]["BASE"][f"{g:g}"]; role = "trained" if g in d["selected"] else (f"omitted ({F['statuses'].get(f'{g:g}', '?')})")
        o.append(f"| G{g:g} | {role} | {d['T_fc'][f'{g:g}']:+.1f} | {a['level_signed']:+.2f} | {a['eq']:.2f} | {a['hf']:.2f} | {a['crest']:.2f} | {a['dyn']:.2f} | {d['T_base'][f'{g:g}']:+.0f} | {b['level_signed']:+.2f} | {b['eq']:.2f} | {b['hf']:.2f} | {b['crest']:.2f} | {b['dyn']:.2f} |")
    o += ["", "### By region of the amp's response range (mean and worst; FC | BASE)\n", "| Region | Positions | level |err| | EQ | HF | crest | dyn | THD | worst level |", "|---|---|---|---|---|---|---|---|---|"]
    for rn, rr in d["regions"].items(): o.append(f"| {rn} | " + ", ".join(f"G{g:g}" for g in rr["positions"]) + " | " + " | ".join(f"{rr['FC_s0'][m]:.2f} \\| {rr['BASE'][m]:.2f}" for m in ("level_abs", "eq", "hf", "crest", "dyn", "thd")) + f" | {rr['FC_s0']['worst_level']:.2f} \\| {rr['BASE']['worst_level']:.2f} |")
    o += ["", f"Whole amp mean (FC | BASE): level {mn['level_abs']:.2f} | {mb['level_abs']:.2f}, EQ {mn['eq']:.2f} | {mb['eq']:.2f}, HF {mn['hf']:.2f} | {mb['hf']:.2f}, crest {mn['crest']:.2f} | {mb['crest']:.2f}, dyn {mn['dyn']:.2f} | {mb['dyn']:.2f}, THD {mn['thd']:.2f} | {mb['thd']:.2f}; worst-position level {mn['worst_level']:.2f} | {mb['worst_level']:.2f}.\n",
          "### Soft / normal / hard playing at each position's intended setting (mean |error| over positions and held-out DIs; FC | BASE)\n", "| Playing | level | HF | crest | dyn |", "|---|---|---|---|---|"]
    for p in PICK: o.append(f"| {p} ({PICK[p]:+.0f} dB DI) | " + " | ".join(f"{d['pick_abs_err']['FC_s0'][p][n]:.2f} \\| {d['pick_abs_err']['BASE'][p][n]:.2f}" for n in THR) + " |")
    fl = d["flagged"]; pr = d["progression"]; e = d["esr_vs_teacher"]
    o += ["", f"Flagged position/DI/level combinations (level/HF/crest >= 1.0 dB or dyn >= 1.5 dB): FC {fl['FC_s0']} of {fl['combos']}, BASE {fl['BASE']} of {fl['combos']}.\n",
          f"Continuous sweep (-20..+14 dB, nominal playing): opposite-direction steps FC {pr['FC_s0']['opposite_direction_steps']}, BASE {pr['BASE']['opposite_direction_steps']}; largest step / median step FC {pr['FC_s0']['max_step_ratio']:.1f}, BASE {pr['BASE']['max_step_ratio']:.1f}; native output-level range FC {pr['FC_s0']['level_range_db']:.1f} dB, BASE {pr['BASE']['level_range_db']:.1f} dB (level decreases >0.3 dB over the sweep: FC {pr['FC_s0']['level_decreases_gt_0p3dB']}, BASE {pr['BASE']['level_decreases_gt_0p3dB']}).\n",
          "Reachability (is there an Input gain in -20..+14 at which the model matches this real position within working tolerances on level/HF/crest/dyn at nominal playing?):\n", "| Position | FC best Input gain | FC distance (<=1 reachable) | BASE best Input gain | BASE distance |", "|---|---:|---:|---:|---:|"]
    for g in real["gains"]: a, b = d["accessibility"]["FC_s0"][f"{g:g}"], d["accessibility"]["BASE"][f"{g:g}"]; o.append(f"| G{g:g} | {a['best_T']:+.0f} | {a['min_dist']:.2f} {'' if a['accessible'] else '(NOT reachable)'} | {b['best_T']:+.0f} | {b['min_dist']:.2f} {'' if b['accessible'] else '(not reachable)'} |")
    o += ["", f"Student vs its own teacher (level-matched ESR, mean): total level < 0 dB {e['total_lt0']:.4f}, total level >= +8 dB {e['total_ge8'] if e['total_ge8'] is None else round(e['total_ge8'], 4)} ({e['n']} cases).\n", "Extremes of the player range (mean over soft/normal/hard, two DIs; error against the real reference driven equivalently; FC | BASE):\n", "| Input gain | level err | HF err | crest err | dyn err | max raw peak (dBFS) | finite |", "|---|---|---|---|---|---|---|"]
    for T_ in ("-20", "+20"): a, b = d["extremes"]["FC_s0"][T_], d["extremes"]["BASE"][T_]; o.append(f"| {T_} dB | {a['level_err']:+.1f} \\| {b['level_err']:+.1f} | {a['hf_err']:+.1f} \\| {b['hf_err']:+.1f} | {a['crest_err']:+.1f} \\| {b['crest_err']:+.1f} | {a['dyn_err']:+.1f} \\| {b['dyn_err']:+.1f} | {a['peak_dbfs']:.1f} \\| {b['peak_dbfs']:.1f} | {a['finite']} |")
    o += ["", f"Export: `{d['verify']['file']}` ({d['verify']['bytes']} bytes), standard structure identical to a stock export: {d['verify']['structure_matches_stock_export']}; finite over -20..+20 dB: {d['verify']['finite_over_-20_to_+20']}; output scale c = {d['verify']['output_scale_c']:.4f} (set the player Output gain to {d['verify']['output_gain_compensation_db']:+.2f} dB to restore the captured level).\n",
          f"Triage: {', '.join(k.split('_')[0] + ' ' + tick(v) for k, v in d['triage'].items())}. **{d['first_pass_verdict']}.**\n"]
(DOC / "CONTINUOUS_GAIN_EXPANSION.md").write_text("\n".join(o) + "\n"); print("written")
