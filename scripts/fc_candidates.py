"""Score the teacher-only candidate evaluations (fc_teacher_eval.py). Usage: python scripts/fc_candidates.py -> prints tables, writes work/p4e/final/candidates.json"""
import glob, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; FC = ROOT / "work" / "p4e" / "final"
MEAS = {"level": "rms_db", "HF": "hf3k_db", "crest": "crest_db", "dyn": "dyn_range_db"}; FLAG = {"level": 1.0, "HF": 1.0, "crest": 1.0, "dyn": 1.5, "EQ": 1.0}
HELD = ["moderate_brit", "clean_mayer", "bass_rollin"]; OFFS = [-12.0, -6.0, 0.0, 6.0]
def score(j):
    P = j["positions"]; C = j["cases"]; g_ = lambda g, o, f: float(np.mean([C[f"{g:g}|{d}|{o:g}"]["err"][f] for d in HELD]))
    eq = lambda g, o: float(np.mean([C[f"{g:g}|{d}|{o:g}"]["eq_abs"] for d in HELD]))
    out = {"nominal": {}, "swing": {}, "flagged_positions": {}, "worst_position": {}}
    for lab, f in MEAS.items():
        nom = {g: abs(g_(g, 0.0, f)) for g in P}; out["nominal"][lab] = float(np.mean(list(nom.values()))); out["worst_position"][lab] = max(nom, key=nom.get)
        out["swing"][lab] = float(np.mean([abs(g_(g, 6.0, f) - g_(g, -12.0, f)) for g in P]))
        out["flagged_positions"][lab] = [g for g in P if nom[g] >= FLAG[lab]]
    out["nominal"]["EQ"] = float(np.mean([eq(g, 0.0) for g in P])); out["flagged_positions"]["EQ"] = [g for g in P if eq(g, 0.0) >= FLAG["EQ"]]
    out["nominal_by_position"] = {lab: {f"{g:g}": g_(g, 0.0, f) for g in P} for lab, f in MEAS.items()}
    sw = j["sweep"]; T = np.array(sw["T"]); st = {}
    for lab, f in MEAS.items():
        y = np.mean([[r[f] for r in sw["feats"][d]] for d in HELD], axis=0); d_ = np.abs(np.diff(y)); st[lab] = {"max_step": float(d_.max()), "at_T": float(T[int(np.argmax(d_))]), "ratio": float(d_.max() / max(np.median(d_), 0.05))}
    out["sweep"] = st
    return out
res = {}
for amp in ("vibrolux", "jcm800"):
    print("=====", amp)
    cands = {Path(f).stem.split("_", 2)[2]: json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(FC / f"teacher_{amp}_*.json")))}
    res[amp] = {n: score(j) for n, j in cands.items()}
    print(f"{'cand':5s} anchors                                 | nominal |err| mean: level HF crest dyn EQ | flagged positions (level/HF/crest/dyn/EQ) | mean |swing| level HF crest dyn | sweep max step level HF crest dyn (ratio worst)")
    for n, s in res[amp].items():
        j = cands[n]; print(f"{n:5s} {[f'G{g:g}' for g in j['gains']]!s:36s}| {s['nominal']['level']:.2f} {s['nominal']['HF']:.2f} {s['nominal']['crest']:.2f} {s['nominal']['dyn']:.2f} {s['nominal']['EQ']:.2f} | "
                            f"{len(s['flagged_positions']['level'])}/{len(s['flagged_positions']['HF'])}/{len(s['flagged_positions']['crest'])}/{len(s['flagged_positions']['dyn'])}/{len(s['flagged_positions']['EQ'])} | {s['swing']['level']:.2f} {s['swing']['HF']:.2f} {s['swing']['crest']:.2f} {s['swing']['dyn']:.2f} | "
                            f"{s['sweep']['level']['max_step']:.2f} {s['sweep']['HF']['max_step']:.2f} {s['sweep']['crest']['max_step']:.2f} {s['sweep']['dyn']['max_step']:.2f} ({max(v['ratio'] for v in s['sweep'].values()):.1f})")
    b = res[amp]["B"]; print("  worst positions for B (nominal): ", {k: f"G{v:g}" for k, v in b["worst_position"].items()}, " flagged:", {k: [f"G{x:g}" for x in v] for k, v in b["flagged_positions"].items() if v})
(FC / "candidates.json").write_text(json.dumps(res, indent=1, default=float))
