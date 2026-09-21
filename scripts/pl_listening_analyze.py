"""Analyse the fixed-virtual-gain listening results. Usage: python scripts/pl_listening_analyze.py results.csv [more.csv ...] [--key KEY.json] [--out summary.md]
Reads the CSV exported by listening_tool.html, resolves the blind labels with the key, and reports (per candidate, amp, playing intensity, virtual gain):
mean closeness, per-item wins, paired B-vs-C3 comparison, dependence on playing intensity, issue tags, and whether the hidden reference was recognised."""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KEYS = {"fixed": ROOT / "work" / "p4e" / "listening_fixed_gain" / "listening_KEY_do_not_open_before_listening.json", "fc": ROOT / "work" / "p4e" / "final" / "listening" / "listening_KEY_do_not_open_before_listening.json"}
DEFAULT_KEY = KEYS["fixed"]
MODEL_NAMES = {"v3_C3": "v3 C3 (old)", "B_s0": "Phase 4E B seed 0", "B_s1": "Phase 4E B seed 1", "FC_s0": "FC seed 0 (new)", "FC_s1": "FC seed 1 (new)", "HIDDEN_REFERENCE": "hidden real amp"}

def load(paths, key, impute=None):
    """impute: score assumed for a candidate of an item that appears in the file but has no row (an untouched slider stays at 50). None = leave missing."""
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p, newline="")):
            k = key.get(r["stimulus_id"]);
            if not k or r["label"] not in k["labels"]: continue
            rows.append({"who": r.get("listener", ""), "sid": r["stimulus_id"], "model": k["labels"][r["label"]], "score": float(r["closeness"]), "amp": k["amp"], "gain": k["virtual_gain"],
                         "pick": k["pick"] if k["kind"] == "single" else "sequence", "kind": k["kind"], "issues": [t.strip() for t in (r.get("issues") or "").split(";") if t.strip()], "mode": r.get("mode_at_rating", "")})
    if impute is not None:
        seen = {(r["who"], r["sid"]) for r in rows}; have = {(r["who"], r["sid"], r["model"]) for r in rows}
        for who, sid in sorted(seen):
            k = key[sid]
            for lab, m in k["labels"].items():
                if (who, sid, m) not in have:
                    rows.append({"who": who, "sid": sid, "model": m, "score": float(impute), "amp": k["amp"], "gain": k["virtual_gain"], "pick": k["pick"] if k["kind"] == "single" else "sequence",
                                 "kind": k["kind"], "issues": [], "mode": "", "imputed": True})
    return rows

def analyse(rows):
    out = {}
    items = defaultdict(dict)
    for r in rows: items[(r["who"], r["sid"])][r["model"]] = r
    def mean(xs): return float(np.mean(xs)) if xs else float("nan")
    out["n_ratings"] = len(rows); out["n_items"] = len(items)
    present = [m for m in MODEL_NAMES if any(r["model"] == m for r in rows)]; cand = [m for m in present if m != "HIDDEN_REFERENCE"]; out["models"] = present; out["candidates"] = cand
    out["mean_by_model"] = {m: mean([r["score"] for r in rows if r["model"] == m]) for m in present}
    for dim in ("amp", "pick", "gain"):
        out[f"mean_by_{dim}"] = {str(v): {m: mean([r["score"] for r in rows if r[dim] == v and r["model"] == m]) for m in present} for v in sorted({r[dim] for r in rows}, key=str)}
    wins = defaultdict(float); n = 0; ref_top = 0.0; ref_n = 0; paired = []; pairs = defaultdict(list)
    def grp(it, ms):
        v = [it[m]["score"] for m in ms if m in it]; return mean(v) if v else None
    PAIRS = {"B_minus_C3": (["B_s0", "B_s1"], ["v3_C3"]), "FC_minus_B": (["FC_s0", "FC_s1"], ["B_s0", "B_s1"]), "FC_minus_C3": (["FC_s0", "FC_s1"], ["v3_C3"]), "FC_s0_minus_FC_s1": (["FC_s0"], ["FC_s1"]), "B_s0_minus_B_s1": (["B_s0"], ["B_s1"])}
    for it in items.values():
        cands = {m: v["score"] for m, v in it.items() if m != "HIDDEN_REFERENCE"}
        if cands:
            top = max(cands.values()); w = [m for m, s in cands.items() if s == top]; n += 1
            for m in w: wins[m] += 1 / len(w)
        if "HIDDEN_REFERENCE" in it and cands:
            ref_n += 1; ref_top += 1.0 if it["HIDDEN_REFERENCE"]["score"] > max(cands.values()) else (0.5 if it["HIDDEN_REFERENCE"]["score"] == max(cands.values()) else 0.0)
        for nm, (x, y) in PAIRS.items():
            a_, b_ = grp(it, x), grp(it, y)
            if a_ is not None and b_ is not None: pairs[nm].append(a_ - b_)
    out["wins"] = {m: wins[m] for m in cand}; out["n_items_with_candidates"] = n
    out["hidden_reference_rated_top_fraction"] = ref_top / ref_n if ref_n else float("nan")
    out["paired"] = {nm: {"n": len(v), "mean": mean(v), "first_better": int(sum(p > 0 for p in v)), "second_better": int(sum(p < 0 for p in v)), "ties": int(sum(p == 0 for p in v))} for nm, v in pairs.items() if v}
    pdep = {}
    for m in cand:
        g = lambda pk: mean([r["score"] for r in rows if r["model"] == m and r["pick"] == pk])
        pdep[m] = {"soft": g("soft"), "normal": g("normal"), "hard": g("hard"), "sequence": g("sequence")}
    out["pick_dependence"] = pdep
    tags = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for t in r["issues"]: tags[r["model"]][t] += 1
    out["issue_tags"] = {m: dict(v) for m, v in tags.items()}
    return out

def markdown(a):
    f = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"
    o = ["# Fixed-virtual-gain listening results", "", f"{a['n_ratings']} ratings over {a['n_items']} listener-items. Scores are closeness to the reference (0-100; for sequence items, how well the candidate keeps the reference's character from soft to hard).", "",
         "## Overall mean score by model", "", "| Model | mean |", "|---|---:|"] + [f"| {MODEL_NAMES[m]} | {f(v)} |" for m, v in a["mean_by_model"].items()]
    hr = a["hidden_reference_rated_top_fraction"]
    o += ["", f"**Attention check:** the hidden real amp was rated best in {hr*100:.0f}% of items (chance is 20-25%, one in the number of options). If this is low the listening session is not trustworthy.", "",
          "## Per-item wins (best candidate, excluding the hidden reference; ties split)", "", "| " + " | ".join(MODEL_NAMES[m] for m in a["candidates"]) + " | items |", "|" + "---:|" * (len(a["candidates"]) + 1), "| " + " | ".join(f"{a['wins'][m]:.1f}" for m in a["candidates"]) + f" | {a['n_items_with_candidates']} |", ""] + [f"**Paired {nm.replace('_minus_', ' minus ')} per item:** mean {f(v['mean'])} points over {v['n']} items; first better in {v['first_better']}, second better in {v['second_better']}, ties {v['ties']}." for nm, v in a["paired"].items()] + ["",
          "## By playing intensity (does it still sound like that amp when played softer or harder?)", "", "| Model | soft | normal | hard | soft-to-hard sequence |", "|---|---:|---:|---:|---:|"] + [f"| {MODEL_NAMES[m]} | {f(v['soft'])} | {f(v['normal'])} | {f(v['hard'])} | {f(v['sequence'])} |" for m, v in a["pick_dependence"].items()]
    for dim, title in (("amp", "By amp"), ("gain", "By virtual gain")):
        o += ["", f"## {title}", "", "| " + dim + " | " + " | ".join(MODEL_NAMES[m] for m in a["models"]) + " |", "|---|" + "---:|" * len(a["models"])] + [f"| {k} | " + " | ".join(f(v[m]) for m in a["models"]) + " |" for k, v in a[f"mean_by_{dim}"].items()]
    o += ["", "## Issues ticked (count per model)", ""] + [f"- **{MODEL_NAMES[m]}**: " + (", ".join(f"{t} ({c})" for t, c in sorted(v.items(), key=lambda x: -x[1])) or "none") for m, v in a["issue_tags"].items()]
    return "\n".join(o) + "\n"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+"); ap.add_argument("--key", default=None); ap.add_argument("--set", choices=list(KEYS), default="fixed", help="which listening set: fixed (v3 C3 vs B) or fc (v3 C3, B s0, FC s0, FC s1)"); ap.add_argument("--out"); ap.add_argument("--impute", type=float, default=50.0, help="score for candidates of visited items with no row (untouched slider = 50); use --impute -1 to leave missing"); a = ap.parse_args()
    key = json.loads(Path(a.key or KEYS[a.set]).read_text()); imp = None if a.impute < 0 else a.impute; md = markdown(analyse(load(a.csv, key, imp))); print(md)
    if a.out: Path(a.out).write_text(md)
