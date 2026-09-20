"""Standard-NAM verification of the exported Phase 4E models (structure vs a stock v3 export + NAMCore inference across the input-gain range)."""
import json, os, sys, hashlib
os.environ["SINGLE_NAM_AMP"] = "jcm800"
import numpy as np
from p4e_common import *
from single_nam_common import render_file_nam
KEYS = ["A_s0", "A_s1", "B_s0", "B_s1"]
def sig(d):
    cfg = d["config"]
    return {"top_keys": sorted(d), "architecture": d.get("architecture"), "version": d.get("version"), "sample_rate": d.get("sample_rate"),
            "config_keys": sorted(cfg) if isinstance(cfg, dict) else None, "n_weights": len(d["weights"]), "metadata_keys": sorted(d.get("metadata", {}))}
out = {"note": "NAMCore (the C++ core used by the official NAM plugin) via native/nam_render is the conventional inference implementation available here; no official plugin/host was available to run.", "models": {}}
for amp in ("jcm800", "vibrolux"):
    os.environ["SINGLE_NAM_AMP"] = amp
    ref_nam, _ = model_path(amp, "v3_C3"); ref = json.loads(ref_nam.read_text()); rs = sig(ref)
    di = clip("moderate_brit")[: 8 * SR]
    for k in KEYS:
        nam, c = model_path(amp, k); d = json.loads(nam.read_text()); s = sig(d)
        diff = {f: {"v3": rs[f], "new": s[f]} for f in rs if rs[f] != s[f] and f != "metadata_keys"}
        extra_meta = sorted(set(s["metadata_keys"]) - set(rs["metadata_keys"]))
        rec = {"file": str(nam.relative_to(REPO)), "sha256": hashlib.sha256(nam.read_bytes()).hexdigest(), "bytes": nam.stat().st_size, "structure_matches_v3_C3": not diff, "structure_differences": diff,
               "metadata_keys_not_in_v3": extra_meta, "custom_conditioning_or_extra_processing": False}
        gains_db = list(range(-48, 37, 6)); peaks = []; finite = True
        for T in gains_db:
            y = render_file_nam(nam, (di * db(T)).astype(np.float32)); finite &= bool(np.isfinite(y).all())
            peaks.append(float(20 * np.log10(max(np.max(np.abs(y)), 1e-9))))
        rec["raw_output_peak_dbfs_by_input_gain_db"] = dict(zip(gains_db, peaks)); rec["all_finite"] = finite
        rec["max_raw_peak_dbfs"] = max(peaks); rec["input_gain_range_with_raw_peak_over_0dbfs_db"] = [g for g, p in zip(gains_db, peaks) if p > 0.0]
        rec["evaluation_output_scale_c"] = c
        out["models"][f"{amp}_{k}"] = rec
        print(amp, k, "matches v3:", rec["structure_matches_v3_C3"], "finite:", finite, "max raw peak %.1f dBFS" % max(peaks), "over-0dBFS at input gain:", rec["input_gain_range_with_raw_peak_over_0dbfs_db"], flush=True)
(P4E / "verify_standard_nam.json").write_text(json.dumps(out, indent=1))
