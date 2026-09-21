"""Extremes of the practical Input range (-20/+20 dB) with soft/normal/hard playing for FC_s0 and BASE. Reference: lowest selected capture at -20 (extra offset relative to its own intended gain),
highest selected capture driven by the same extra offset at +20. Usage: p5_extremes.py <amp> -> work/p4e/final/<amp>/extremes.json"""
import json, os, sys
os.environ["SINGLE_NAM_AMP"] = sys.argv[1]
import numpy as np
from fc_common import *
from single_nam_common import render_file_nam
amp = sys.argv[1]; res = {"amp": amp, "rows": []}; sel = FC_CFG[amp][0]; g_lo, g_hi = min(sel), max(sel)
for key in ("FC_s0", "BASE"):
    nam, c = fc_model_path(amp, key); Tlo, Thi = fc_intended_T(amp, key, [g_lo, g_hi])
    for Tx, g_ref, ref_extra in ((-20.0, g_lo, -20.0 - Tlo), (20.0, g_hi, 20.0 - Thi)):
        for d in ("clean_mayer", "moderate_brit"):
            for pk, o in (("soft", -12.0), ("normal", 0.0), ("hard", 6.0)):
                x = clip(d); y = render_file_nam(nam, (x * db(o + Tx)).astype(np.float32)) / c; ry = render_aligned(amp, g_ref, (x * db(o + ref_extra)).astype(np.float32)); f = feats2(y.astype(np.float32)); r = feats2(ry)
                res["rows"].append({"model": key, "input_gain_db": Tx, "di": d, "pick": pk, "reference_position": g_ref, "reference_extra_offset_db": ref_extra, "finite": bool(np.isfinite(y).all()), "raw_peak_dbfs": float(20 * np.log10(max(np.max(np.abs(y)), 1e-9))),
                                    "level_err": f["rms_db"] - r["rms_db"], "hf_err": f["hf3k_db"] - r["hf3k_db"], "crest_err": f["crest_db"] - r["crest_db"], "dyn_err": f["dyn_range_db"] - r["dyn_range_db"]})
(FCDIR / amp / "extremes.json").write_text(json.dumps(res, default=float)); print("done", amp)
