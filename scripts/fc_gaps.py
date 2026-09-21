"""Final-candidate task, part 4: evidence-based gap assessment per amp from EXISTING data only (no new renders).
A/B coverage (student errors per position at native level), C continuous-sweep transitions, D soft-to-hard problems, E teacher-vs-student attribution, F omitted-capture association.
Usage: python scripts/fc_gaps.py -> work/p4e/final/gaps.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4e_metrics as M
import pl_load as L
from scipy.interpolate import PchipInterpolator
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "work" / "p4e" / "final"; OUT.mkdir(parents=True, exist_ok=True)
ANCH = {"jcm800": [1.0, 2.0, 4.0, 10.0], "vibrolux": [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]}
FLAG = {"level_abs": 1.0, "eq": 1.0, "hf": 1.0, "crest": 1.0, "dyn": 1.5}
res = {}
for amp in ("jcm800", "vibrolux"):
    real, ev = M.load(amp); gains = real["gains"]; R = {"amp": amp}
    # A/B: coverage at native musical level, intended mapping (B) / fixed (C3)
    tabs = {"B_s0": M.model_table(real, ev["B_s0"], "intended"), "B_s1": M.model_table(real, ev["B_s1"], "intended"), "C3": M.model_table(real, ev["v3_C3"], "fixed")}
    R["per_position"] = {k: {f"{g:g}": {m: t[g][m] for m in ("level_signed", "level_abs", "eq", "hf", "crest", "dyn", "thd", "io_slope", "esr")} for g in gains} for k, t in tabs.items()}
    R["flagged_positions"] = {k: {f"{g:g}": [m for m, thr in FLAG.items() if t[g][m] >= thr] for g in gains if any(t[g][m] >= thr for m, thr in FLAG.items())} for k, t in tabs.items()}
    # C: continuous sweep vs the real progression (real interpolated along the model's own input-gain axis)
    Tb = np.array(M.MAN and [None] or [None])
    import p4e_common as C
    os_T = C.intended_T(amp, "B", gains) if amp else None
    seg = {}
    for k in ("B_s0", "B_s1"):
        sw = json.loads((ROOT / "work" / "p4e" / amp / f"sweep_{k}.json").read_text()); T = np.array(sw["T"]); m_ = (T >= -24) & (T <= 16)
        for lab, f, thr in (("level", "rms_db", 1.5), ("HF", "hf3k_db", 1.5), ("crest", "crest_db", 1.5), ("dyn", "dyn_range_db", 2.0)):
            mod = np.mean([[r[f] for r in sw["music"][d]] for d in M.HELD], axis=0)
            rv = np.array([np.mean([real["music"][f"{g:g}|{d}|0"][f] for d in M.HELD]) for g in gains]); order = np.argsort(os_T); pc = PchipInterpolator(np.array(os_T)[order], rv[order])
            grid = T[m_ & (T >= min(os_T)) & (T <= max(os_T))]; dev = mod[m_ & (T >= min(os_T)) & (T <= max(os_T))] - pc(grid)
            bad = [float(t) for t, d_ in zip(grid, dev) if abs(d_) > thr]
            seg.setdefault(lab, {})[k] = {"threshold": thr, "max_abs_dev": float(np.max(np.abs(dev))), "input_gains_over_threshold": bad}
    R["sweep_vs_real"] = seg
    # D/E: soft-to-hard problems and attribution (playability cases)
    cs = L.load_cases(amp); pg = sorted({c["g"] for c in cs}); rows = []
    for g in pg:
        for lab, f, thr in (("level", "rms_db", 1.0), ("HF", "hf3k_db", 1.0), ("crest", "crest_db", 1.0), ("dyn range", "dyn_range_db", 1.5)):
            se = {o: float(np.mean([np.mean([c["student"][k][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o]) for k in ("B_s0", "B_s1")])) for o in L.OFFS}
            te = {o: float(np.mean([c["teacher"]["B"][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o])) for o in L.OFFS}
            fx, sw_ = np.mean(list(se.values())), se[6.0] - se[-12.0]; ft, st_ = np.mean(list(te.values())), te[6.0] - te[-12.0]
            if max(abs(fx), abs(sw_)) >= thr:
                dyn_type = abs(sw_) >= max(thr, abs(fx)); a_, b_ = (sw_, st_) if dyn_type else (fx, ft)
                who = "teacher" if abs(b_) >= 0.6 * abs(a_) else ("student" if abs(a_ - b_) >= 0.6 * abs(a_) else "both")
                rows.append({"g": g, "measure": lab, "type": "swing" if dyn_type else "fixed", "student_fixed": fx, "student_swing": sw_, "teacher_fixed": ft, "teacher_swing": st_, "where": who})
    R["softhard_flagged"] = rows; R["attribution"] = {w: sum(r["where"] == w for r in rows) for w in ("teacher", "student", "both")}
    # F: omitted vs anchor positions, TEACHER errors and swings
    om = [g for g in pg if g not in ANCH[amp]]; an = [g for g in pg if g in ANCH[amp]]
    def terr(g, o, f): return float(np.mean([c["teacher"]["B"][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o]))
    grp = lambda gs, f: {"nominal_mean_abs": float(np.mean([abs(terr(g, 0.0, f)) for g in gs])), "swing_mean_abs": float(np.mean([abs(terr(g, 6.0, f) - terr(g, -12.0, f)) for g in gs]))}
    R["omitted_vs_anchor_teacher"] = {lab: {"anchors": grp(an, f), "omitted": grp(om, f), "omitted_positions": om} for lab, f in (("level", "rms_db"), ("HF", "hf3k_db"), ("crest", "crest_db"), ("dyn range", "dyn_range_db"))}
    R["omitted_per_position_swing"] = {lab: {f"{g:g}": float(terr(g, 6.0, f) - terr(g, -12.0, f)) for g in pg} for lab, f in (("level", "rms_db"), ("HF", "hf3k_db"), ("crest", "crest_db"), ("dyn range", "dyn_range_db"))}
    res[amp] = R
(OUT / "gaps.json").write_text(json.dumps(res, indent=1, default=float))
for amp, R in res.items():
    print("=====", amp)
    print("flagged positions (native level, working flags level/EQ/HF/crest >=1 dB, dyn >=1.5):")
    for k, v in R["flagged_positions"].items(): print(f"   {k}: {v}")
    print("sweep vs real progression, input gains where |model - real| > threshold (B_s0 / B_s1):")
    for lab, d in R["sweep_vs_real"].items(): print(f"   {lab}: max dev {d['B_s0']['max_abs_dev']:.1f}/{d['B_s1']['max_abs_dev']:.1f}; over threshold at {d['B_s0']['input_gains_over_threshold']} | {d['B_s1']['input_gains_over_threshold']}")
    print("soft-to-hard flagged:", len(R["softhard_flagged"]), R["attribution"])
    for r in sorted(R["softhard_flagged"], key=lambda r: -max(abs(r['student_swing']), abs(r['student_fixed'])))[:8]: print(f"   G{r['g']:g} {r['measure']:9s} {r['type']:5s} student fixed {r['student_fixed']:+.1f} swing {r['student_swing']:+.1f} | teacher {r['teacher_fixed']:+.1f}/{r['teacher_swing']:+.1f} | {r['where']}")
    print("teacher error at anchors vs omitted positions (nominal |err| / swing |err|):")
    for lab, d in R["omitted_vs_anchor_teacher"].items(): print(f"   {lab:9s} anchors {d['anchors']['nominal_mean_abs']:.2f}/{d['anchors']['swing_mean_abs']:.2f}   omitted {d['omitted']['nominal_mean_abs']:.2f}/{d['omitted']['swing_mean_abs']:.2f}  ({d['omitted_positions'] if lab=='level' else ''})")
