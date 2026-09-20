"""Score the weight-band variants against the predeclared criteria (docs/phase4e/weight_band_criteria.md). Pure functions over the band_* JSON files."""
import glob, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; PL = ROOT / "work" / "p4e" / "playability"
PHIS = [0.0, 0.5, 0.75, 0.9, 1.0]; OFFS = [-12.0, -6.0, 0.0, 6.0]
MEAS = {"level": "rms_db", "HF": "hf3k_db", "crest": "crest_db", "dyn range": "dyn_range_db", "tilt": "tilt_db"}
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]
ANCH = {"jcm800": [1.0, 2.0, 4.0, 10.0], "vibrolux": [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]}
def load(amp):
    cs = []
    for f in sorted(glob.glob(str(PL / f"band_cases_{amp}_*.json"))):
        j = json.loads(Path(f).read_text()); cs += [dict(c, g=j["g"]) for c in j["cases"]]
    return cs
def err(cs, cfg, phi, g, o, f):
    return float(np.mean([c["cfg"][cfg]["phi"][str(phi)]["feats"][f] - c["real"][f] for c in cs if c["g"] == g and c["offset"] == o]))
def eq_err(cs, cfg, phi, g, o):
    return float(np.mean([np.mean([abs(c["cfg"][cfg]["phi"][str(phi)]["feats"][b] - c["real"][b]) for b in EQ]) for c in cs if c["g"] == g and c["offset"] == o]))
def metrics(amp, cfg, phi, cs=None):
    cs = cs or load(amp); gains = sorted({c["g"] for c in cs}); out = {"swing": {}, "fixed": {}}
    for lab, f in MEAS.items():
        sw = [err(cs, cfg, phi, g, 6.0, f) - err(cs, cfg, phi, g, -12.0, f) for g in gains]
        fx = [np.mean([err(cs, cfg, phi, g, o, f) for o in OFFS]) for g in gains]
        out["swing"][lab] = float(np.mean(np.abs(sw))); out["fixed"][lab] = float(np.mean(np.abs(fx))); out.setdefault("swing_by_pos", {})[lab] = dict(zip(gains, sw))
    out["eq_swing"] = float(np.mean([abs(eq_err(cs, cfg, phi, g, 6.0) - eq_err(cs, cfg, phi, g, -12.0)) for g in gains]))
    an = cs[0]["cfg"][cfg]["gains"]
    cm = lambda g, o: float(np.dot(np.mean([c["cfg"][cfg]["phi"][str(phi)]["mean_weight_active"] for c in cs if c["g"] == g and c["offset"] == o], axis=0), an))
    out["com_swing"] = float(np.mean([cm(g, 6.0) - cm(g, -12.0) for g in gains]))
    out["nearest_weight_nominal"] = float(np.mean([c["cfg"][cfg]["phi"][str(phi)]["nearest_anchor_weight"] for c in cs if c["offset"] == 0.0]))
    om = [g for g in gains if g not in ANCH[amp]]
    out["omitted_err"] = {lab: {"mean": float(np.mean([abs(err(cs, cfg, phi, g, 0.0, f)) for g in om])), "by_pos": {g: abs(err(cs, cfg, phi, g, 0.0, f)) for g in om}} for lab, f in MEAS.items() if lab != "tilt"}
    out["all_err_nominal"] = {lab: float(np.mean([abs(err(cs, cfg, phi, g, 0.0, f)) for g in gains])) for lab, f in MEAS.items() if lab != "tilt"}
    out["median_crossfade_ms"] = float(np.median([c["cfg"][cfg]["phi"][str(phi)]["cross"]["median_ms"] for c in cs if c["cfg"][cfg]["phi"][str(phi)]["cross"]["median_ms"] is not None] or [np.nan]))
    out["switches_per_s"] = float(np.mean([c["cfg"][cfg]["phi"][str(phi)]["cross"]["switches_per_s"] for c in cs]))
    out["hf10k_excess"] = float(np.mean([c["cfg"][cfg]["phi"][str(phi)]["hf10k"] - c["real_hf10k"] for c in cs]))
    out["esr_mean"] = float(np.mean([c["cfg"][cfg]["phi"][str(phi)]["esr"] for c in cs]))
    out["esr_by_offset"] = {o: float(np.mean([c["cfg"][cfg]["phi"][str(phi)]["esr"] for c in cs if c["offset"] == o])) for o in OFFS}
    return out
def criteria(amp, cfg="B"):
    cs = load(amp); base = metrics(amp, cfg, 0.0, cs); res = {}
    for phi in PHIS[1:]:
        m = metrics(amp, cfg, phi, cs); c1 = {k: (1 - m["swing"][k] / base["swing"][k]) for k in ("level", "HF", "crest", "dyn range")}
        c2m = {k: m["omitted_err"][k]["mean"] - base["omitted_err"][k]["mean"] for k in ("level", "HF", "crest", "dyn range")}
        c2s = {k: max(m["omitted_err"][k]["by_pos"][g] - base["omitted_err"][k]["by_pos"][g] for g in m["omitted_err"][k]["by_pos"]) for k in ("level", "HF", "crest", "dyn range")}
        c3 = {"median_crossfade_ms": m["median_crossfade_ms"], "hf10k_excess_vs_phi0": m["hf10k_excess"] - base["hf10k_excess"]}
        p1 = all(v >= 0.40 for v in c1.values()); p2 = all(v <= 0.5 for v in c2m.values()) and all(v <= 1.5 for v in c2s.values()); p3 = (m["median_crossfade_ms"] >= 5.0) and (c3["hf10k_excess_vs_phi0"] <= 1.5)
        res[phi] = {"c1_swing_reduction": c1, "c2_omitted_mean_increase": c2m, "c2_worst_position_increase": c2s, "c3": c3, "pass1": p1, "pass2": p2, "pass3": p3, "promising": p1 and p2 and p3}
    return res, base
if __name__ == "__main__":
    for amp in ("jcm800", "vibrolux"):
        res, base = criteria(amp); print("=====", amp, "cfg B; phi=0 mean|swing|:", {k: round(v, 2) for k, v in base["swing"].items()})
        for phi, r in res.items():
            print(f" phi={phi}: swing reduction", {k: f"{v*100:.0f}%" for k, v in r["c1_swing_reduction"].items()}, "| omitted-pos mean err increase", {k: round(v, 2) for k, v in r["c2_omitted_mean_increase"].items()}, "worst pos +", {k: round(v, 1) for k, v in r["c2_worst_position_increase"].items()}, "| xfade ms", None if r["c3"]["median_crossfade_ms"] is None else round(r["c3"]["median_crossfade_ms"], 1), "hf10k excess", round(r["c3"]["hf10k_excess_vs_phi0"], 2), "| PASS 1/2/3:", r["pass1"], r["pass2"], r["pass3"], "-> PROMISING" if r["promising"] else "")
