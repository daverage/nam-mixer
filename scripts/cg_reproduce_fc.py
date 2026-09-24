"""Reproduce the FROZEN JCM800 / Vibrolux FC configurations through the hybrid/continuous_gain/ backend and compare them
field-by-field (capture sets, anchors, target/input audio hashes, output scale) with the frozen record.
Reads only the archived Phase 4 measurements under work/p4 (extract research_work_dirs_*.tar.gz first) and the
user's own capture files; writes nothing outside the given --out directory. Usage: cg_reproduce_fc.py <amp> [--out DIR] [--skip-audio]"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import os
ap = argparse.ArgumentParser(); ap.add_argument("amp"); ap.add_argument("--out", type=Path, default=None); ap.add_argument("--skip-audio", action="store_true"); a = ap.parse_args()
os.environ["SINGLE_NAM_AMP"] = a.amp
from single_nam_common import capture_path, official_input
from hybrid.continuous_gain.probe import load_reference_di, SR
from hybrid.continuous_gain.selection import select_captures, resolve_selection
from hybrid.continuous_gain.anchors import response_anchors
from hybrid.continuous_gain.audit import alignment_shift
from hybrid.continuous_gain.bundle import make_chain, build_training_audio, bundle_manifest_core, FC_RECIPE
from hybrid.core.nam_loader import load_nam
from hybrid.core.render import render

P = json.loads((REPO / "work/p4" / a.amp / "profile.json").read_text()); A = json.loads((REPO / "work/p4" / a.amp / "audit.json").read_text())
frozen = json.loads((REPO / "work/p4e/final" / a.amp / "FC_bundle/manifest.json").read_text())
F = json.loads((REPO / "docs/history/Continuous Gain/final/manifest_frozen.json").read_text())
res = {"amp": a.amp, "checks": {}}
an = select_captures(P, A); sel = resolve_selection(an, P, A, "automatic")
res["selected"] = sel["selected"]; res["checks"]["capture_set"] = sel["selected"] == [float(g) for g in frozen["gains"]]
S = json.loads((REPO / "work/p4" / a.amp / "selection.json").read_text())
pos, anchors = response_anchors(an["response_coordinate"], sel["selected"])
res["anchors"] = anchors; res["checks"]["anchors"] = anchors == [float(t) for t in frozen["anchors_designated_gain_db"]]
res["checks"]["anchors_vs_frozen_manifest"] = anchors == [float(t) for t in F["amps"][a.amp]["anchors_designated_input_gain_db"].values()]
res["checks"]["capture_set_vs_frozen_manifest"] = [f"G{g:g}" for g in sel["selected"]] == list(F["amps"][a.amp]["captures"].keys())
shifts = {g: alignment_shift(A["captures"][f"{g:g}"]) for g in pos}
chain = make_chain(pos, anchors)
res["checks"]["levels"] = list(chain.levels_db) == frozen["levels_db"]
if not a.skip_audio:
    models = {g: load_nam(capture_path(g)) for g in pos}
    built = build_training_audio(chain, lambda g, x: render(models[g], x, SR), shifts, official_input(), load_reference_di, FC_RECIPE, progress=lambda m: print(m, flush=True))
    core = bundle_manifest_core(built, chain, anchors, shifts)
    for k in ("target_audio_sha256", "input_audio_sha256", "output_scale_c", "peak_ceiling_gain_reduction_db", "train_stop", "total", "train_seconds", "val_seconds", "train_offsets_db", "val_offsets_db", "alignment_shifts_samples"):
        res["checks"][k] = core[k] == frozen[k]
        if core[k] != frozen[k]: res["checks"][k + "_got_vs_frozen"] = [core[k], frozen[k]]
    if a.out: a.out.mkdir(parents=True, exist_ok=True)
print(json.dumps(res, indent=1))
if a.out: (a.out / f"reproduce_{a.amp}.json").write_text(json.dumps(res, indent=1))
sys.exit(0 if all(v is True for k, v in res["checks"].items() if isinstance(v, bool)) else 1)
