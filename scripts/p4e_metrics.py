"""Phase 4E metric aggregation (pure functions over work/p4e/<amp>/{real_ref,eval_*}.json)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P4E = ROOT / "work" / "p4e"
MAN = json.loads((ROOT / "docs" / "phase4e" / "manifest_frozen.json").read_text())
HELD = MAN["evaluation"]["held_out_dis"]; OFFS = [float(o) for o in MAN["evaluation"]["di_offsets_db"]]
EQ = ["eq_sub_db", "eq_low_db", "eq_lowmid_db", "eq_mid_db", "eq_presence_db", "eq_air_db"]
THD_LV = (-42, -30, -18, -6)
FLOOR = -80.0
MODELS = ["v3_C3", "v3_C5", "v3_C10", "A_s0", "A_s1", "B_s0", "B_s1"]
GROUPS = ["level_signed", "level_abs", "eq", "hf", "tilt", "thd", "h23", "io_slope", "di_intensity", "crest", "dyn", "transient", "esr"]
LABEL = {"level_signed": "level signed (dB)", "level_abs": "level |err| (dB)", "eq": "EQ bands (dB)", "hf": "HF>3k (dB)", "tilt": "tilt (dB)", "thd": "THD 4 levels (dB)",
         "h23": "H2/H3 (dB)", "io_slope": "IO slopes (dB)", "di_intensity": "DI-intensity resp. (dB)", "crest": "crest (dB)", "dyn": "dyn range (dB)", "transient": "transient rise (dB)", "esr": "lm-ESR"}

def load(amp):
    real = json.loads((P4E / amp / "real_ref.json").read_text()); ev = {}
    for k in MODELS:
        p = P4E / amp / f"eval_{k}.json"
        if p.exists(): ev[k] = json.loads(p.read_text())
    return real, ev

def variants(model, e):
    return ["intended", "fixed", "fitted"] if model.startswith("B") else ["fixed", "fitted"]

def _get(e, variant, kind):
    r = e["results"]["fixed" if (variant == "intended" and e["results"]["intended"] == "same as fixed") else variant]
    return r[kind]

def position_metrics(real, e, variant, g, o=0.0):
    """errors of one model/mapping at one position and DI offset (mean over held-out DIs); tone-based ones are DI-independent."""
    M = _get(e, variant, "music"); T = _get(e, variant, "tones"); RM, RT = real["music"], real["tones"]
    k = lambda gg, d, oo: f"{gg:g}|{d}|{oo:g}"
    d_ = lambda f, oo=o: np.array([M[k(g, d, oo)][f] - RM[k(g, d, oo)][f] for d in HELD])
    out = {}
    out["level_signed"] = float(d_("rms_db").mean()); out["level_abs"] = float(np.abs(d_("rms_db")).mean())
    out["eq"] = float(np.mean([np.abs(d_(b)).mean() for b in EQ])); out["hf"] = float(np.abs(d_("hf3k_db")).mean()); out["tilt"] = float(np.abs(d_("tilt_db")).mean())
    out["crest"] = float(np.abs(d_("crest_db")).mean()); out["dyn"] = float(np.abs(d_("dyn_range_db")).mean()); out["transient"] = float(np.abs(d_("rise_p95_db")).mean())
    out["hf_signed"] = float(d_("hf3k_db").mean()); out["tilt_signed"] = float(d_("tilt_db").mean()); out["crest_signed"] = float(d_("crest_db").mean())
    out["eq_signed"] = {b: float(d_(b).mean()) for b in EQ}
    tk = lambda t, f0, lv: t[f"{g:g}|{f0}|{lv:g}"]
    out["thd"] = float(np.mean([abs(max(tk(T, 440, lv)["thd_db"], FLOOR) - max(tk(RT, 440, lv)["thd_db"], FLOOR)) for lv in THD_LV]))
    out["thd_signed"] = float(np.mean([max(tk(T, 440, lv)["thd_db"], FLOOR) - max(tk(RT, 440, lv)["thd_db"], FLOOR) for lv in THD_LV]))
    out["h23"] = float(np.mean([abs(max(tk(T, 440, -18)[h], FLOOR) - max(tk(RT, 440, -18)[h], FLOOR)) for h in ("h2_db", "h3_db")]))
    io = lambda t: (tk(t, 440, -30)["out_rms_db"] - tk(t, 440, -54)["out_rms_db"] - 24, tk(t, 440, 0)["out_rms_db"] - tk(t, 440, -30)["out_rms_db"] - 30)
    a, b = io(T), io(RT); out["io_slope"] = float(np.mean([abs(a[0] - b[0]), abs(a[1] - b[1])])); out["io_signed"] = [float(a[0] - b[0]), float(a[1] - b[1])]
    di = lambda MM: np.mean([MM[k(g, d, 6)]["rms_db"] - MM[k(g, d, -12)]["rms_db"] for d in HELD])
    out["di_intensity"] = float(abs(di(M) - di(RM))); out["di_intensity_model"] = float(di(M)); out["di_intensity_real"] = float(di(RM))
    if "lm_esr" in M[k(g, HELD[0], 0)]:
        out["esr"] = float(np.mean([M[k(g, d, 0)]["lm_esr"] for d in HELD]))
    return out

def model_table(real, e, variant, offset=0.0):
    return {g: position_metrics(real, e, variant, g, offset) for g in real["gains"]}

def summarise(tab, gains, regions=None, trained=None):
    """mean / worst-position per group, per region, trained vs omitted."""
    def agg(gs, grp):
        v = [tab[g][grp] for g in gs if grp in tab[g]]; a = [abs(x) for x in v]
        return {"mean": float(np.mean(v)) if v else float("nan"), "worst": float(np.max(a)) if a else float("nan")}
    out = {"all": {grp: agg(gains, grp) for grp in GROUPS}}
    if regions:
        for r in ("low", "transition", "high", "plateau"):
            gs = [g for g in gains if regions.get(f"{g:g}") == r]
            if gs: out[r] = {grp: agg(gs, grp) for grp in GROUPS}
    if trained is not None:
        for nm, gs in (("trained", [g for g in gains if g in trained]), ("omitted", [g for g in gains if g not in trained and g == int(g)]), ("half_steps", [g for g in gains if g != int(g)])):
            if gs: out[nm] = {grp: agg(gs, grp) for grp in GROUPS}
    return out
