"""Apparent physical position of an output: nearest REAL capture (same DI, same musical-input offset) in tone + saturation + dynamics feature space. Level excluded."""
import json, numpy as np
from pl_load import ROOT, load_cases, OFFS
FEATS = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db", "hf3k_db", "tilt_db", "crest_db", "dyn_range_db"]
def apparent(amp):
    real = json.loads((ROOT / "work" / "p4e" / amp / "real_ref.json").read_text()); gains = real["gains"]; cs = load_cases(amp); out = {}
    for c in cs:
        d, o = c["di"], c["offset"]
        R = np.array([[real["music"][f"{g:g}|{d}|{o:g}"][f] for f in FEATS] for g in gains])
        scale = np.maximum(np.ptp(np.array([[real["music"][f"{g:g}|{d}|0"][f] for f in FEATS] for g in gains]), axis=0), 0.5)
        def where(feats):
            v = np.array([feats[f] for f in FEATS]); dist = np.sqrt(np.mean(((R - v) / scale) ** 2, axis=1)); i = int(np.argmin(dist))
            order = np.argsort(dist); return gains[i], float(dist[i]), float(dist[list(gains).index(c["g"])])
        for who, feats in (("teacher_B", c["teacher"]["B"]), ("teacher_A", c["teacher"]["A"]), ("teacher_C3", c["teacher"]["C3"]), *[(k, v) for k, v in c["student"].items()]):
            out.setdefault((c["g"], o, who), []).append(where(feats))
    return out, gains
if __name__ == "__main__":
    for amp in ("vibrolux", "jcm800"):
        out, gains = apparent(amp); print("=====", amp, "apparent physical position (median of 3 DIs) at FIXED virtual gain; rows: intended position; cols: musical input -12/-6/0/+6")
        for g in sorted({k[0] for k in out}):
            print(f"  G{g:<4g}", " | ".join(f"{who}: " + "/".join(f"{np.median([x[0] for x in out[(g, o, who)]]):4.1f}" for o in OFFS) for who in ("teacher_B", "B_s0", "B_s1", "A_s0", "v3_C3")))
