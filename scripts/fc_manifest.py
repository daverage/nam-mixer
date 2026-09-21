"""Write the FROZEN final-candidate manifest (machine readable). Run once, commit, THEN train."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
os.environ["SINGLE_NAM_AMP"] = "jcm800"; import importlib, single_nam_common as sc
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
P4E = json.loads((ROOT / "docs" / "phase4e" / "manifest_frozen.json").read_text())
m = {"task": "Continuous Gain: best practical single NAM per amp (final candidates)", "frozen_before_training": True,
     "code_commit_at_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "note": "The commit adding this file is the freeze commit; code_commit_at_freeze is its parent.",
     "basis": "Phase 4E configuration B capture sets (unchanged), plus two targeted changes: (1) plugin-compatible anchor mapping (lowest anchor -20 dB), (2) guitar training material covering the whole playback level range",
     "evidence_for_changes": {"mapping": "docs/CONTINUOUS_GAIN_FINAL_CANDIDATES.md section 5 (teacher-only: -20..+14 dB is at least as good as -22..+14, and better on the Vibrolux; +20 top is worse)",
                              "coverage": "student departure from its own teacher rises steeply once total level exceeds about +8 dB on both amps (ESR 0.006 at -8..+8, 0.013 at +8..+14, 0.022-0.026 at +14..+20, 0.049-0.056 above +20); guitar training audio has 3% of active audio at -16..-10 dBFS and none above -10, while playback at G10 sits at -14.8 (normal) and -8.8 dBFS (hard)"},
     "no_capture_set_change": "no evidence that G5, G6, G7 or G8 adds a missing sound (teacher-only comparison, docs section 5); capture sets stay as B",
     "amps": {}, "training": {}, "data": {}, "evaluation": {}, "criteria": {}, "selection_rule": {}, "outputs": {}}
CAPS = {"jcm800": [1, 2, 4, 10], "vibrolux": [1, 2, 3, 4, 7, 10]}; ANCH = {"jcm800": [-20.0, -9.2, -1.7, 14.0], "vibrolux": [-20.0, -15.6, -7.1, -2.2, 5.7, 14.0]}
for a in CAPS:
    os.environ["SINGLE_NAM_AMP"] = a; importlib.reload(sc); AUD = json.loads((ROOT / "work" / "p4" / a / "audit.json").read_text())["captures"]
    m["amps"][a] = {"captures": {f"G{g}": {"file": sc.capture_path(float(g)).name, "sha256": sha(sc.capture_path(float(g))), "qa_status": AUD[f"{float(g):g}"]["status"], "alignment_correction_samples": (AUD[f"{float(g):g}"]["correction"] or {}).get("samples", 0)} for g in CAPS[a]},
                    "anchors_designated_input_gain_db": dict(zip([f"G{g}" for g in CAPS[a]], ANCH[a])), "mapping_rule": "response-distance anchors on [-20, +14] dB (arc length of the measured profile, >= 4 dB apart); other positions linear in the response coordinate between anchors",
                    "b_baseline_anchors_db": P4E["amps"][a]["anchors_designated_input_gain_db"]["B_response_distance_spacing"]}
b = {a: json.loads((ROOT / "work" / "p4e" / "final" / a / "FC_bundle" / "manifest.json").read_text()) for a in CAPS}
m["training"] = {"architecture": "standard NAM A2 (PackedWaveNet), 22,783 parameters, official trainer via scripts/single_nam_train.py (unchanged from Phase 4E)", "trainer_script_sha256": sha(ROOT / "scripts" / "single_nam_train.py"),
                 "epochs": 60, "batch_size": 16, "ny": 8192, "optimiser": "Adam lr 0.004, exponential decay 0.994 per epoch (official default, unchanged)", "seeds": [0, 1], "checkpoint_selection": "stock best-validation checkpoint",
                 "expected_updates": "about 14,600 (data is 1.42x longer than Phase 4E's 467 s; disclosed confound: more updates than B)", "concurrency": "4 models concurrently on one machine (MPS)"}
m["data"] = {"builder": "scripts/fc_build.py (reproduces the frozen Phase 4E B bundle exactly when given B's arguments: both audio hashes identical, checked for the Vibrolux)", "builder_sha256": sha(ROOT / "scripts" / "fc_build.py"),
             "recipe": "official NAM input (190 s) + clean_smooth/moderate_hotrod/high_thrash 20 s each at offsets -32,-24,-16,-8,0,+8,+14,+20 dB; validation high_metalcore 8 s at -24,0,+8,+20; teacher = envelope-driven blend of the selected real captures (hybrid.multi_blend), one -0.2 dBFS peak ceiling, no per-capture level matching, no alignment corrections needed",
             "bundles": {a: {"path": f"work/p4e/final/{a}/FC_bundle", "train_seconds": b[a]["train_seconds"], "val_seconds": b[a]["val_seconds"], "levels_db": b[a]["levels_db"], "output_scale_c": b[a]["output_scale_c"], "target_audio_sha256": b[a]["target_audio_sha256"], "input_audio_sha256": b[a]["input_audio_sha256"]} for a in CAPS}}
m["evaluation"] = {"held_out_dis": ["moderate_brit", "clean_mayer", "bass_rollin"], "di_offsets_db": [-12, -6, 0, 6], "positions": "all real integer captures (JCM800 also the nine genuine half-steps as independent check)",
                   "models": "v3 C3 (its fixed-rule mapping), Phase 4E B seeds 0/1 (frozen B mapping), FC seeds 0/1 (FC mapping); each at its OWN intended Input gain; one global output constant per model (the training scale), no per-gain or per-level correction",
                   "measures": "level, EQ bands, HF, crest, dynamic range, transient, sine THD/IO slope as before; student-vs-teacher ESR by total level; flagged-combination counts with the working flags (level 1.0, HF 1.0, crest 1.0, dyn 1.5 dB); continuous Input-gain sweep -20..+20; extremes of the practical range",
                   "listening": "pending; a listening set for the new candidate will be generated; no perceptual claim is made without it"}
m["criteria"] = {"C1_targeted_weakness": "student-vs-teacher level-matched ESR over cases whose total level (Input gain + playing offset) is at or above +8 dB is at least 40% lower than the mean of the two B seeds, for EACH FC seed",
                 "C2_no_native_level_regression": "over all positions at DI offset 0, mean |error| vs the real capture for level, HF, crest, EQ is not worse than the B seed mean by more than 0.3 dB (dynamic range 0.5 dB), and worst-position level error <= 1.5 dB, for EACH FC seed",
                 "C3_fixed_setting_playability": "flagged amp/position/measure combinations in the soft-to-hard test (working flags) not more than the B seed mean + 2, for EACH FC seed",
                 "C4_reproducibility": "difference between the two FC seeds in each C2 measure is not larger than the larger of 0.3 dB and the two B seeds' difference",
                 "C5_range_and_format": "all anchors within [-20, +14] dB; ordinary .nam structure identical to a stock export; finite output over -20..+20 dB Input gain; raw peaks reported",
                 "C6_sweep": "no step of the continuous Input-gain sweep turns the opposite way to the real amp's progression; largest step ratio (max/median) not more than 1.5x B's",
                 "interpretation": "criteria are working thresholds, not perceptual measures; they are not revised after results. A pass means 'not demonstrably worse and specifically better where targeted'; audible quality still needs listening."}
m["selection_rule"] = {"recommended": "per amp, if C1-C6 all pass for both FC seeds: the FC seed with the lower mean level-matched ESR to the real captures over all evaluation cases (a supporting measure used only to break the tie between seeds); otherwise the B seed with the lower value of the same measure; v3 C3 is reported as a documented alternative (its listening result on the Vibrolux is noted) and is not selected by this rule unless both FC and B fail C5 (range/format) in a way C3 does not",
                       "not_allowed": "averaging models, runtime blending, different models for different gain regions"}
m["outputs"] = {"models": "work/p4e/final/<amp>/FC_bundle/<amp>_FC_s<seed>/*.nam", "deliverables": "deliverables/<amp>/ (chosen .nam, guide, mapping)", "report": "docs/CONTINUOUS_GAIN_FINAL_CANDIDATES.md"}
(ROOT / "docs" / "final" / "manifest_frozen.json").write_text(json.dumps(m, indent=1)); print("frozen; parent commit", m["code_commit_at_freeze"][:10])
