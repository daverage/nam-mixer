"""Write the FROZEN Phase 4E experiment manifest (machine readable). Run once, commit, THEN train."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "work"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
AMPDIR = {"jcm800": "Marshall JCM800 2203 - updated", "vibrolux": "[500 Epochs] Fender Super-Sonic 60W Head mk.1 - Flat EQ - Complete Pack"}
import os, sys
os.environ["SINGLE_NAM_AMP"] = "jcm800"; sys.path.insert(0, str(ROOT / "scripts"))
import single_nam_common as sc  # noqa: E402
T_LO, T_HI = -22.0, 14.0
CFG = {
    "jcm800": {"channel": "High channel, Gain 1-10 (+ genuine half-steps 1.5-9.5 for independent validation)", "gains": [1, 2, 4, 10],
               "A": [-22.0, -18.0, -10.0, 14.0], "B": [-22.0, -10.5, -2.6, 14.0]},
    "vibrolux": {"channel": "Vibrolux channel, T5 B5, Volume 1-10 (integers only)", "gains": [1, 2, 3, 4, 7, 10],
                 "A": [-22.0, -18.0, -14.0, -10.0, 2.0, 14.0], "B": [-22.0, -17.4, -8.4, -3.1, 5.3, 14.0]}}
m = {"experiment": "Continuous Gain Phase 4E pilot", "frozen_on": "before the first Phase 4E training run", "brief": "docs/CONTINUOUS_GAIN_PHASE4E_REVISED.md (user brief)",
     "code_commit_at_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
     "note_on_commit": "The commit that adds this file is the freeze commit; code_commit_at_freeze is its parent, containing the scripts used.",
     "decisions_from_user": ["JCM800 primary set is G1,G2,G4,G10 as selected by the frozen rule; G1,G2,G4,G9 is an optional follow-up only", "no optional adaptive-C3 models", "primary budget: 4 configurations (A/B x 2 amps) x 2 seeds = 8 models"],
     "selection_rule": "top-5 profile-ranked sets at the amp's k*, lowest combined (profile + audio-domain) rank, ties by J (docs/CONTINUOUS_GAIN_PHASE4E_PLAN.md)",
     "amps": {}, "architecture": {}, "training": {}, "data": {}, "evaluation": {}, "comparison_rules": {}, "outputs": {}}
for a, c in CFG.items():
    AUD = json.loads((R / "p4" / a / "audit.json").read_text())["captures"]
    os.environ["SINGLE_NAM_AMP"] = a
    import importlib; importlib.reload(sc)
    caps = {}
    for g in sorted({1, 2, 3, 4, 5, 7, 10} | set(c["gains"])):
        p = sc.capture_path(float(g))
        caps[f"G{g}"] = {"file": p.name, "sha256": sha(p), "qa_status": AUD[f"{float(g):g}"]["status"],
                         "alignment_correction_samples": (AUD[f"{float(g):g}"]["correction"] or {}).get("samples", 0), "used_for_training": g in c["gains"]}
    half = {}
    if a == "jcm800":
        for g in [x + 0.5 for x in range(1, 10)]:
            p = sc.capture_path(g); au = AUD[f"{g:g}"]
            half[f"G{g:g}"] = {"file": p.name, "sha256": sha(p), "qa_status": au["status"], "alignment_correction_samples": (au["correction"] or {}).get("samples", 0), "use": "independent validation only"}
    SEL = json.loads((R / "p4" / a / "selection.json").read_text()); PR = json.loads((R / "p4" / a / "profile.json").read_text())
    last_fast = max(b for r in PR["regions"].values() for _, b in r["fast"])
    m["amps"][a] = {"channel": c["channel"], "captures": caps, "half_step_captures": half, "selected_positions": c["gains"], "eligible_anchor_positions": SEL["eligible"],
                    "anchors_designated_input_gain_db": {"A_fixed_physical_position_spacing": dict(zip([f"G{g}" for g in c["gains"]], c["A"])), "B_response_distance_spacing": dict(zip([f"G{g}" for g in c["gains"]], c["B"]))},
                    "reference_level_dbfs": -30.0, "training_level_of_anchor_dbfs": "anchor dB + reference (-30 dBFS)",
                    "regions_for_reporting": {"plateau_from_position": last_fast, "rule": "plateau = positions at/after the last position where any measured dimension still changes quickly (4B profile); positions before it are split into equal-count thirds by knob order: low, transition, high"},
                    "response_coordinate": SEL["response_coordinate"], "profile_source": f"work/p4/{a}/profile.json (sha256 {sha(R / 'p4' / a / 'profile.json')[:16]})", "audit_source": f"work/p4/{a}/audit.json"}
m["architecture"] = {"type": "standard NAM A2 (PackedWaveNet)", "parameters": 22783, "trainer": "official nam.train.core.train via scripts/single_nam_train.py (data-split/input-detection hooks patched only)", "trainer_script_sha256": sha(ROOT / "scripts" / "single_nam_train.py"),
                     "exported_format": "standard .nam; no conditioning input, no custom metadata, no post-processing"}
m["training"] = {"epochs": 60, "batch_size": 16, "ny": 8192, "optimiser": "Adam (official trainer default), lr 0.004, exponential decay gamma 0.994 per epoch", "checkpoint_selection": "stock best-validation checkpoint (pooled validation ESR); identical for every configuration",
                 "seeds": [0, 1], "expected_optimiser_updates": "about 175 per epoch x 60 = about 10,500 (set by the length of the training audio, identical for every configuration; recorded from the final checkpoints after training)",
                 "exposure_note": "training audio length is identical (467 s) for every configuration, so a 4-capture or 6-capture model sees the same number of updates as v3 C3, C5 and C10 but each capture is exposed differently (fewer captures = more exposure per capture). This is a disclosed confound for comparisons with the v3 models.",
                 "concurrency": "all 8 models train concurrently on one machine (MPS); wall-clock per model is recorded with the concurrency level; v3 solo/concurrent timings (about 17-24 min) are the reference",
                 "v3_baselines_preserved": {f"C{n}": {"model": f"work/cg/{a}/cg_{n}/", "trained_gains": json.loads((R / "cg" / a / f"cg_{n}" / "manifest.json").read_text())["gains"]} for a in CFG for n in (3, 5, 10)}}
m["data"] = {"builder": "scripts/p4e_build.py (new; v3 cg_build.py untouched and validated bit-identical to v3 for JCM800 G1,G5,G10 with fixed anchors)", "builder_sha256": sha(ROOT / "scripts" / "p4e_build.py"),
             "recipe": "official NAM input (190 s) + clean_smooth/moderate_hotrod/high_thrash (25 s each) at -16,-8,0,+8 dB; validation high_metalcore 12 s; teacher = level-driven blend of selected real captures (hybrid.multi_blend), each rendered at the -30 dBFS reference level; one fixed -0.2 dBFS peak ceiling; no per-capture level matching",
             "alignment": "verified Phase 4A music-derived shifts only (none needed for either pilot amp's selected captures); no click-based shifts (v3 used click-based shifts for the Vibrolux: -2,-2,-1 samples on G1-G3, disclosed)",
             "bundles": {}}
for a in CFG:
    for cfg in "AB":
        bm = json.loads((R / "p4e" / a / f"{cfg}_bundle" / "manifest.json").read_text())
        m["data"]["bundles"][f"{a}_{cfg}"] = {"path": f"work/p4e/{a}/{cfg}_bundle", "train_seconds": bm["train_seconds"], "val_seconds": bm["val_seconds"], "levels_db": bm["levels_db"], "output_scale_c": bm["output_scale_c"],
                                              "target_audio_sha256": bm["target_audio_sha256"], "input_audio_sha256": bm["input_audio_sha256"]}
m["evaluation"] = {
    "held_out_dis": ["moderate_brit", "clean_mayer", "bass_rollin"], "fit_dis": ["clean_smooth", "moderate_hotrod", "high_thrash", "high_metalcore"], "clip_seconds": 15,
    "di_offsets_db": [-12, -6, 0, 6], "positions": "all real integer captures, plus the nine genuine half-steps on the JCM800 (independent check); omitted-from-training positions reported separately from trained positions",
    "independence": "JCM800: half-steps and held-out DIs are fully independent. Vibrolux: omitted integer positions are neural-training holdouts but were used in 4D subset selection, so they are NOT fully independent research validation.",
    "metrics": ["native output RMS, peak and signed level error", "EQ band errors (six bands), HF>3 kHz, tilt, signed coloration direction", "sine THD at input levels -42,-30,-18,-6 dBFS, H2/H3 distribution",
                "input/output slopes (-54->-30, -30->0), response to changing DI intensity (output change from -12 to +6 dB DI)", "crest, dynamic range, transient rise rate (p95 of positive 5 ms envelope steps)", "level-matched ESR (supporting only)"],
    "reporting": "mean, worst-position and per-region (low / transition / high / plateau) errors, per DI level, no composite score; real-vs-model curves; signed errors kept",
    "playback_mappings": {"intended": "A: fixed rule -22+4(N-1) dB at every position; B: linear in the response coordinate between its declared anchors (reproduces them exactly); v3 C3/C5/C10: fixed rule",
                          "fixed": "the v3 fixed rule for every model (for B this is a deliberate mismatch and is labelled so)",
                          "fitted": "one ordered (non-decreasing) mapping per model, fitted ONLY on the fit DIs by the Phase 4C objective (tone + saturation + compression, level excluded, 1 dB grid, T <= 30 dB, monotone DP over integer positions; half-steps by PCHIP), then frozen for all held-out DIs and levels"},
    "flags_only": "Phase 4C working thresholds (tone 0.15/1.0 dB, HF 0.2/1.0, crest 0.2/1.0, THD 1.5/5, IO 0.5/2, dyn 0.3/1.5, level 0.4/1.5) flag differences for examination; they are not validated perceptual thresholds"}
m["comparison_rules"] = {"A_vs_v3_C3": "compares a new capture set (count AND placement change) under the original anchor rule; NOT claimed to isolate placement alone",
                         "B_vs_A": "isolates the training-anchor rule within each selected set (all other settings identical)",
                         "seed_rule": "a difference is conclusive only if it is larger than the seed-to-seed difference for that metric and has the same sign in both seeds; otherwise inconclusive",
                         "no_composite": "no composite score and no single-metric victory; trade-offs (e.g. compression better but level/tone/transients worse) are reported as trade-offs",
                         "criteria_not_revised_after_results": True}
m["outputs"] = {"root": "work/p4e/", "models": "work/p4e/<amp>/<A|B>_bundle/<amp>_P4E_<A|B>_s<seed>/*.nam", "results_doc": "docs/CONTINUOUS_GAIN_PHASE4E_RESULTS.md", "listening": "work/p4e/listening/ (+ docs/phase4e/)"}
out = json.dumps(m, indent=1)
(R / "p4e").mkdir(exist_ok=True); (R / "p4e" / "manifest_frozen.json").write_text(out)
(ROOT / "docs" / "phase4e").mkdir(exist_ok=True); (ROOT / "docs" / "phase4e" / "manifest_frozen.json").write_text(out)
print("frozen; code commit", m["code_commit_at_freeze"], "sha256", hashlib.sha256(out.encode()).hexdigest()[:16])
