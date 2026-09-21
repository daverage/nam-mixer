"""Package one deliverable directory per amp: deliverables/expansion/<amp>/ (model, source manifest, anchors, Input range, output gain, player guide, evaluation, limitations, listening status).
Usage: p5_package.py [amp ...]  (after p5_eval.sh and p5_report.py)"""
import json, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent; FROZEN = json.loads((REPO / "docs/expand/manifest_frozen.json").read_text()); RPT = json.loads((REPO / "work/p4e/final/expansion_report.json").read_text())
REG = ["bottom of the response range", "lower-middle", "upper-middle", "top of the response range"]
for amp in sys.argv[1:] or list(RPT):
    d = RPT[amp]; F = FROZEN["amps"][amp]; out = REPO / "deliverables" / "expansion" / amp; out.mkdir(parents=True, exist_ok=True); v = d["verify"]
    nam = REPO / v["file"]; shutil.copy(nam, out / f"{amp}_continuous_gain_FC_s0.nam")
    (out / "manifest.json").write_text(json.dumps({"amp": amp, "name": F["name"], "model_file": f"{amp}_continuous_gain_FC_s0.nam", "model_sha256": v["sha256"], "source_captures": F["captures"], "capture_dir": F["capture_dir"], "selection_basis": F["selection_basis"],
        "anchors_input_gain_db": dict(zip([f"G{g:g}" for g in F["selected_positions"]], F["anchors_input_gain_db"])), "player_input_gain_range_db": [-20, 14], "output_gain_compensation_db": v["output_gain_compensation_db"], "training": {"recipe": "docs/expand/manifest_frozen.json", "seed": 0, "epochs": 60, "wall_seconds": d["wall_seconds"], "concurrent_models": 6},
        "alignment_shifts_samples": {k: c["alignment_shift_samples"] for k, c in F["captures"].items()}, "listening_status": "NOT LISTENED TO"}, indent=1))
    (out / "evaluation.json").write_text(json.dumps({k: d[k] for k in ("means", "regions", "pick_abs_err", "flagged", "progression", "accessibility", "esr_vs_teacher", "extremes", "triage", "first_pass_verdict")}, indent=1))
    rows = []
    for rn, rr in d["regions"].items():
        Ts = [d["T_fc"][f"{g:g}"] for g in rr["positions"]]; rows.append(f"| {rn} | " + ", ".join(f"G{g:g}" for g in rr["positions"]) + f" | {min(Ts):+.0f} to {max(Ts):+.0f} dB |")
    unreach = [f"G{k}" for k, a in d["accessibility"]["FC_s0"].items() if not a["accessible"]]
    (out / "README.md").write_text(f"""# {F['name']}: continuous-gain NAM (first pass)

**Status: automatically generated, first pass, seed 0. Not listened to, not audibly validated.** Technical verdict: **{d['first_pass_verdict']}** (triage in `docs/CONTINUOUS_GAIN_EXPANSION.md`).

## Player setup (any NAM player, no NAM Mixer needed)
- Load `{amp}_continuous_gain_FC_s0.nam`.
- **Input gain: -20 to +14 dB** (the trained range; +20 dB is finite but beyond training).
- **Output gain: {v['output_gain_compensation_db']:+.2f} dB** to restore the captured level (the training target was scaled by c = {v['output_scale_c']:.4f} to keep peaks below -0.2 dBFS).
- Native output level rises with Input gain across the range ({d['progression']['FC_s0']['level_range_db']:.1f} dB from -20 to +14 dB).

## Player guide (Input gain per region of this amp's measured response range; physical positions in brackets)
| Region | Physical positions covered | Input gain (dB) |
|---|---|---|
""" + "\n".join(rows) + f"""

Selected training captures and anchors (Input gain at which each capture was trained): {', '.join(f"G{g:g} at {a:+.1f} dB" for g, a in zip(F['selected_positions'], F['anchors_input_gain_db']))}. Sounds between anchors are interpolations the model learned, not captures.

## Evaluation (held-out DIs, details in evaluation.json)
Mean level/EQ/HF/crest/dyn error vs the real captures: {d['means']['FC_s0']['level_abs']:.2f} / {d['means']['FC_s0']['eq']:.2f} / {d['means']['FC_s0']['hf']:.2f} / {d['means']['FC_s0']['crest']:.2f} / {d['means']['FC_s0']['dyn']:.2f} dB (single-middle-capture baseline {d['means']['BASE']['level_abs']:.2f} / {d['means']['BASE']['eq']:.2f} / {d['means']['BASE']['hf']:.2f} / {d['means']['BASE']['crest']:.2f} / {d['means']['BASE']['dyn']:.2f}); worst-position level {d['means']['FC_s0']['worst_level']:.2f} dB. Real positions not reachable within working tolerances at any Input gain: {', '.join(unreach) if unreach else 'none'}.

## Known limitations
See the amp section of `docs/CONTINUOUS_GAIN_EXPANSION.md`. Positions excluded as SUSPECT in the capture audit: {', '.join(f'G{k}' for k, s in F['statuses'].items() if s == 'SUSPECT') or 'none'}. One seed only. No tonal refinement applied. A different (unaudited) IR/cabinet than the one used to judge the tone may change how it sounds.
""")
    print("packaged", amp)
