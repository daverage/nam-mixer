"""Expansion milestone step 1: FREEZE the per-amp FC configuration before any build/training. Sets come from the existing Phase 4D rule (best set at k*); anchors from the
existing FC response-distance rule (fc_common.anchor_levels, -20..+14 dB). -> docs/expand/manifest_frozen.json"""
import hashlib, json, os, subprocess, sys
import numpy as np
os.environ["SINGLE_NAM_AMP"] = "twin"
sys.path.insert(0, os.path.dirname(__file__))
from fc_common import anchor_levels, response_arc
import single_nam_common as S
REPO = S.REPO
SETS = {"twin": ([1, 2, 3, 4, 9], "4D k*=5 best set"), "supersonic": ([1, 2, 3, 9], "4D k*=4 best set (Bassman channel; G4, G7, G10 SUSPECT and excluded)"),
        "peavey": ([1, 3, 6], "4D k*=3 best set (G4, G10 SUSPECT and excluded)"), "peavey6505": ([1, 3, 5, 9], "4D k*=4 best set"), "mesa": ([1, 3, 9], "4D k*=3 best set"),
        "orange": ([1, 4, 6, 8, 9], "EXCEPTION: 4D found no k* (no set within tolerance at any count); the 4D best 5-set is used as the same-size rule used for the Phase 4E reference set; G2, G7 SUSPECT and excluded")}
NAMES = {"twin": "Fender 57 Custom Twin", "supersonic": "Fender Super-Sonic Bassman channel", "peavey": "Peavey 5150", "peavey6505": "Peavey 6505+", "mesa": "Mesa Dual Rectifier Rev G (Red Modern)", "orange": "Orange Dual Terror (Fat)"}
out = {"task": "Continuous Gain expansion: first-pass FC NAM per remaining amp", "frozen_before_training": True, "code_commit_at_freeze": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
       "recipe_reference": "docs/final/manifest_frozen.json (FC recipe unchanged: scripts/fc_build.py + scripts/single_nam_train.py, A2, 60 epochs, seed 0 only, official input + 3 guitar DIs at offsets -32..+20 dB, validation high_metalcore)",
       "training": {"seeds": [0], "epochs": 60, "second_seed_rule": "only if the first pass shows a reproducibility concern or a candidate is otherwise ready for use"},
       "build_args": {"offsets": [-32, -24, -16, -8, 0, 8, 14, 20], "val_offsets": [-24, 0, 8, 20], "di_seconds": 20, "val_seconds": 8}, "amps": {}}
for amp, (gains, why) in SETS.items():
    os.environ["SINGLE_NAM_AMP"] = amp; import importlib; importlib.reload(S)
    au = json.loads((REPO / "work" / "p4" / amp / "audit.json").read_text())["captures"]
    g_, T = anchor_levels(amp, [float(g) for g in gains]); allg = sorted(float(k) for k in au)
    out["amps"][amp] = {"name": NAMES[amp], "capture_dir": str(S.capture_path(1.0).parent), "positions_available": allg, "statuses": {k: v["status"] for k, v in au.items()},
        "selected_positions": gains, "selection_basis": why, "anchors_input_gain_db": [round(t, 1) for t in T],
        "captures": {f"{g:g}": {"file": S.capture_path(float(g)).name, "sha256": hashlib.sha256(S.capture_path(float(g)).read_bytes()).hexdigest(), "audit_status": au[f"{g:g}"]["status"], "alignment_shift_samples": (au[f"{g:g}"]["correction"] or {}).get("samples", 0)} for g in gains},
        "response_arc": dict(zip([f"{g:g}" for g in response_arc(amp)[0]], [round(float(a), 3) for a in response_arc(amp)[1]]))}
    print(amp, gains, [round(t, 1) for t in T])
(REPO / "docs" / "expand" / "manifest_frozen.json").write_text(json.dumps(out, indent=1))
