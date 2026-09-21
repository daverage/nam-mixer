"""Evidence for the training-coverage gap (existing data only): student-vs-teacher departure by total level reaching the NAM, and the level distribution of the training input.
Usage: python scripts/fc_coverage.py -> work/p4e/final/coverage.json"""
import json, os, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent)); os.environ["SINGLE_NAM_AMP"] = "vibrolux"
import pl_load as L
import p4e_common as C
from hybrid.envelope import bounded_causal_envelope_db
ROOT = Path(__file__).resolve().parent.parent; out = {"esr_by_total_level": {}, "training_level_hist": {}}
bins = [(-40, -16), (-16, -8), (-8, 0), (0, 8), (8, 14), (14, 20), (20, 30)]
for amp in ("jcm800", "vibrolux"):
    cs = L.load_cases(amp); d = {}
    for cfg in ("B", "A"):
        for b in bins:
            v = [np.mean([c["esr"][f"{cfg}_s0_vs_teacher"], c["esr"][f"{cfg}_s1_vs_teacher"]]) for c in cs if b[0] <= c["T"][cfg] + c["offset"] < b[1]]
            d.setdefault(cfg, {})[f"{b[0]}..{b[1]}"] = {"esr": float(np.mean(v)) if v else None, "n": len(v)}
    out["esr_by_total_level"][amp] = d
edges = np.arange(-70, 5, 6)
for amp in ("jcm800", "vibrolux"):
    root = ROOT / "work" / "p4e" / amp / "B_bundle"; m = json.loads((root / "manifest.json").read_text()); X, _ = sf.read(root / "input.wav", dtype="float32"); seg = [s for s in m["segments"] if s["split"] == "train"]
    def hist(ss):
        e = np.concatenate([bounded_causal_envelope_db(X[s["start"]:s["stop"]], 48000) for s in ss]); e = e[e > -70]; return (np.histogram(e, bins=edges)[0] / max(len(e), 1) * 100).tolist(), len(e) / 48000
    g, gs = hist([s for s in seg if s["source"] != "official"]); o, os_ = hist([s for s in seg if s["source"] == "official"])
    out["training_level_hist"][amp] = {"edges": edges.tolist(), "guitar_pct": g, "guitar_seconds": gs, "official_pct": o, "official_seconds": os_}
di = C.clip("clean_mayer"); ed = bounded_causal_envelope_db(di, 48000); out["native_median_active_dbfs"] = float(np.median(ed[ed > ed.max() - 40]))
(ROOT / "work" / "p4e" / "final" / "coverage.json").write_text(json.dumps(out, indent=1)); print("saved coverage.json; native median", round(out["native_median_active_dbfs"], 1))
