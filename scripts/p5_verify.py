"""Standard-NAM verification of the expansion exports (structure vs a stock export, finite over -20..+20 dB Input gain, raw peaks). -> work/p4e/final/<amp>/verify.json"""
import hashlib, json, os, sys
os.environ["SINGLE_NAM_AMP"] = "jcm800"
import numpy as np
from fc_common import *
from single_nam_common import render_file_nam
def sig(d): cfg = d["config"]; return {"top_keys": sorted(d), "architecture": d.get("architecture"), "version": d.get("version"), "sample_rate": d.get("sample_rate"), "config_keys": sorted(cfg) if isinstance(cfg, dict) else None, "n_weights": len(d["weights"])}
ref = json.loads(model_path("jcm800", "v3_C3")[0].read_text()); rs = sig(ref)
for amp in sys.argv[1:]:
    os.environ["SINGLE_NAM_AMP"] = amp; nam, c = fc_model_path(amp, "FC_s0"); d = json.loads(nam.read_text()); s = sig(d); diff = {f: {"stock": rs[f], "new": s[f]} for f in rs if rs[f] != s[f]}; di = clip("moderate_brit")[: 6 * SR]; gains = list(range(-20, 21, 4)); peaks = []; fin = True
    for T in gains: y = render_file_nam(nam, (di * db(T)).astype(np.float32)); fin &= bool(np.isfinite(y).all()); peaks.append(float(20 * np.log10(max(np.max(np.abs(y)), 1e-9))))
    out = {"file": str(nam.relative_to(REPO)), "sha256": hashlib.sha256(nam.read_bytes()).hexdigest(), "bytes": nam.stat().st_size, "structure_matches_stock_export": not diff, "differences": diff, "extra_metadata_keys": sorted(set(d.get("metadata", {})) - set(ref.get("metadata", {}))),
           "finite_over_-20_to_+20": fin, "raw_peak_dbfs_by_input_gain": dict(zip(gains, peaks)), "output_scale_c": c, "output_gain_compensation_db": float(-20 * np.log10(c))}
    (FCDIR / amp / "verify.json").write_text(json.dumps(out, indent=1)); print(amp, "structure ok:", not diff, "finite:", fin, "max raw peak %.1f dBFS" % max(peaks), "output comp %+.2f dB" % out["output_gain_compensation_db"])
