"""Automated PREVIEW of the fixed-virtual-gain listening set: objective distances between each candidate and the real reference, computed from the SAME audio files a
listener hears. Not a perceptual prediction. Reports by amp / virtual gain / playing intensity only (no item numbers, no A-D letters) so the blind test stays blind.
Usage: python scripts/pl_listening_auto.py -> docs/phase4e/listening_fixed_gain/AUTOMATED_PREVIEW.md (+ work/p4e/listening_fixed_gain/auto_metrics.json)"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["SINGLE_NAM_AMP"] = "jcm800"
import numpy as np, soundfile as sf
import p4e_common as C
from hybrid.audio_metrics import multi_resolution_log_spectral_distance as mrlsd
from hybrid.validation import compute_esr_metrics

ROOT = Path(__file__).resolve().parent.parent; LG = C.P4E / "listening_fixed_gain"; OUT = ROOT / "docs" / "phase4e" / "listening_fixed_gain" / "AUTOMATED_PREVIEW.md"
key = json.loads((LG / "listening_KEY_do_not_open_before_listening.json").read_text()); pub = {p["id"]: p for p in json.loads((LG / "stimuli_public.json").read_text())}
MODELS = ["v3_C3", "B_s0", "B_s1"]; SR = 48000
TRAINED = {"v3_C3": {1.0, 5.0, 10.0}, "B": {"jcm800": {1.0, 2.0, 4.0, 10.0}, "vibrolux": {1.0, 2.0, 3.0, 4.0, 7.0, 10.0}}}
NAME = {"v3_C3": "v3 C3", "B_s0": "B seed 0", "B_s1": "B seed 1"}; AMP = {"jcm800": "JCM800", "vibrolux": "Vibrolux"}
def rd(path): y, sr = sf.read(LG / path, dtype="float32"); assert sr == SR; return y
def lm_esr(a, b):
    n = min(len(a), len(b)); p, q = a[SR // 2:n].astype(np.float64), b[SR // 2:n].astype(np.float64); return float(compute_esr_metrics(p, q)["raw_esr"])
def feat_diff(a, b):
    fa, fb = C.feats2(a), C.feats2(b)
    return {"level": fa["rms_db"] - fb["rms_db"], "HF": fa["hf3k_db"] - fb["hf3k_db"], "crest": fa["crest_db"] - fb["crest_db"], "dyn": fa["dyn_range_db"] - fb["dyn_range_db"],
            "EQ": float(np.mean([abs(fa[b_] - fb[b_]) for b_ in C.EQ]))}
SEG = [(0, 4.0), (4.5, 8.5), (9.0, 13.0)]
rows = []
for sid, k in key.items():
    p = pub[sid]; ref_n = rd(p["files"]["REFERENCE"]["native"]); ref_l = rd(p["files"]["REFERENCE"]["levelmatched"])
    for L, m in k["labels"].items():
        if m == "HIDDEN_REFERENCE": continue
        cn = rd(p["files"][L]["native"]); cl = rd(p["files"][L]["levelmatched"]); n = min(len(cl), len(ref_l))
        r = {"amp": k["amp"], "gain": k["virtual_gain"], "pick": k["pick"] if k["kind"] == "single" else "sequence", "model": m,
             "mrlsd": float(mrlsd(cl[:n], ref_l[:n])), "esr_lm": lm_esr(cl, ref_l), **{f"d_{a}": v for a, v in feat_diff(cn, ref_n).items()}}
        if k["kind"] == "sequence":
            f = lambda y: [C.feats2(y[int(a * SR):int(b * SR)]) for a, b in SEG]
            fc, fr = f(cn), f(ref_n)
            r["seq"] = {"level_rise": (fc[2]["rms_db"] - fc[0]["rms_db"]) - (fr[2]["rms_db"] - fr[0]["rms_db"]), "HF_change": (fc[2]["hf3k_db"] - fc[0]["hf3k_db"]) - (fr[2]["hf3k_db"] - fr[0]["hf3k_db"]),
                        "crest_change": (fc[2]["crest_db"] - fc[0]["crest_db"]) - (fr[2]["crest_db"] - fr[0]["crest_db"]), "dyn_change": (fc[2]["dyn_range_db"] - fc[0]["dyn_range_db"]) - (fr[2]["dyn_range_db"] - fr[0]["dyn_range_db"]),
                        "real_level_rise_db": fr[2]["rms_db"] - fr[0]["rms_db"]}
        rows.append(r)
(LG / "auto_metrics.json").write_text(json.dumps(rows, indent=1))
def mean(x): return float(np.mean(x)) if len(x) else float("nan")
by = defaultdict(dict)
for r in rows: by[(r["amp"], r["gain"], r["pick"])][r["model"]] = r
o = ["# Automated preview of the fixed-gain listening set (NOT a perceptual result)\n",
"**Read this AFTER you listen if you want your ratings unbiased; read it BEFORE if you only want to know where to spend your time.** Human listening is still pending; nothing here is an audible claim.\n",
"For every clip these are objective distances between each candidate and the REAL reference, computed from the same audio files you will hear (`scripts/pl_listening_auto.py`). Lower = closer to the real amp. They are proxies: **MR-LSD** is a multi-resolution log-spectral distance on the level-matched versions (tone, saturation and dynamics without loudness); **EQ**, **HF>3k**, **crest** and **dyn range** are level-independent feature differences (candidate minus real, dB); **level** is the native-level difference; **lm-ESR** is a waveform error after level matching (a weak guide for saturated material). To keep the blind test blind this page reports by amp, virtual gain and playing intensity only, with no item numbers or A-D letters. The hidden real amp in each item is not scored (it would be zero).\n",
"## 1. Overall (mean over all single clips, MR-LSD dB, lower is closer)\n", "| Model | MR-LSD | EQ err | |HF| | |crest| | |dyn range| | |level| | lm-ESR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
single = [r for r in rows if r["pick"] != "sequence"]
for m in MODELS:
    s = [r for r in single if r["model"] == m]
    o.append(f"| {NAME[m]} | {mean([r['mrlsd'] for r in s]):.2f} | {mean([r['d_EQ'] for r in s]):.2f} | {mean([abs(r['d_HF']) for r in s]):.2f} | {mean([abs(r['d_crest']) for r in s]):.2f} | {mean([abs(r['d_dyn']) for r in s]):.2f} | {mean([abs(r['d_level']) for r in s]):.2f} | {mean([r['esr_lm'] for r in s]):.3f} |")
o += ["", "## 2. By amp and playing intensity (mean MR-LSD; the closest candidate is in bold)\n", "| Amp | intensity | v3 C3 | B seed 0 | B seed 1 | B (mean of seeds) vs v3 C3 |", "|---|---|---:|---:|---:|---|"]
for amp in ("vibrolux", "jcm800"):
    for pk in ("soft", "normal", "hard"):
        v = {m: mean([r["mrlsd"] for r in single if r["amp"] == amp and r["pick"] == pk and r["model"] == m]) for m in MODELS}; b = (v["B_s0"] + v["B_s1"]) / 2; best = min(v, key=v.get)
        o.append(f"| {AMP[amp]} | {pk} | " + " | ".join((f"**{v[m]:.2f}**" if m == best else f"{v[m]:.2f}") for m in MODELS) + f" | B is {abs(1 - b / v['v3_C3']) * 100:.0f}% {'closer' if b < v['v3_C3'] else 'FARTHER'} |")
o += ["", "## 3. Per setting (single clips; MR-LSD, closest bold; B is the mean of its two seeds)\n", "| Amp | virtual gain | intensity | v3 C3 | B | predicted closest | B seeds agree? | trained on this setting |", "|---|---:|---|---:|---:|---|---|---|"]
wins = defaultdict(float); rank = []
for (amp, g, pk), d in sorted(by.items(), key=lambda x: (x[0][0] != "vibrolux", x[0][1], ["soft", "normal", "hard", "sequence"].index(x[0][2]))):
    if pk == "sequence": continue
    c3, b0, b1 = d["v3_C3"]["mrlsd"], d["B_s0"]["mrlsd"], d["B_s1"]["mrlsd"]; b = (b0 + b1) / 2; best = "B" if b < c3 else "v3 C3"; wins[(amp, best)] += 1
    agree = "yes" if (b0 < c3) == (b1 < c3) else "NO (seeds split)"; rank.append((abs(b - c3) / max(c3, 1e-9), amp, g, pk, b, c3, best, agree))
    tr = ", ".join(n for n, on in (("v3 C3", g in TRAINED["v3_C3"]), ("B", g in TRAINED["B"][amp])) if on) or "neither"
    o.append(f"| {AMP[amp]} | {g:g} | {pk} | {c3:.2f} | {b:.2f} | {best} | {agree} | {tr} |")
o += ["", "*Trained on this setting* marks which model had this virtual gain as a training anchor. A model is naturally favoured where it was trained (v3 C3 at G5 and G10; B at G2, G4, G10 on the JCM800 and G3, G7, G10 on the Vibrolux), so the most informative rows are the ones where the model was NOT trained on the setting.\n", "Predicted closest (single clips): " + "; ".join(f"{AMP[a]}: B {int(wins[(a, 'B')])}, v3 C3 {int(wins[(a, 'v3 C3')])}" for a in ("vibrolux", "jcm800")) + ".\n",
      "## 4. Soft-to-hard sequences: does the candidate change character the way the real amp does?\n",
      "Candidate minus real for the CHANGE between the soft and the hard part of each sequence (0 = behaves like the real amp). Real level rise is the real amp's own soft-to-hard loudness increase.\n", "| Amp | virtual gain | real level rise (dB) | model | level rise error | HF change error | crest change error | dyn-range change error |", "|---|---:|---:|---|---:|---:|---:|---:|"]
for (amp, g, pk), d in sorted(by.items(), key=lambda x: (x[0][0] != "vibrolux", x[0][1])):
    if pk != "sequence": continue
    for m in MODELS:
        s = d[m]["seq"]; o.append(f"| {AMP[amp]} | {g:g} | {s['real_level_rise_db']:.1f} | {NAME[m]} | {s['level_rise']:+.1f} | {s['HF_change']:+.1f} | {s['crest_change']:+.1f} | {s['dyn_change']:+.1f} |")
tot = {m: mean([abs(d[m]["seq"][k]) for (a, g, pk), d in by.items() if pk == "sequence" for k in ("level_rise", "HF_change", "crest_change", "dyn_change")]) for m in MODELS}
o += ["", "Mean absolute soft-to-hard behaviour error over the eight sequences and four measures: " + ", ".join(f"{NAME[m]} {tot[m]:.2f} dB" for m in MODELS) + ".\n"]
rank.sort(reverse=True)
def only_winner_trained(r):
    amp, g, best = r[1], r[2], r[6]; c3t, bt = g in TRAINED["v3_C3"], g in TRAINED["B"][amp]
    return (best == "v3 C3" and c3t and not bt) or (best == "B" and bt and not c3t)
inform = [x for x in rank if not only_winner_trained(x)]
o += ["## 5. Where to listen first\n", "Single clips with the largest predicted B-versus-v3-C3 difference, EXCLUDING cases where the predicted winner is the only model trained on that setting (those are expected), plus any case where the two B seeds disagree:\n",
      "| Amp | virtual gain | intensity | B MR-LSD | v3 C3 MR-LSD | predicted closest | B seeds agree? | trained on this setting |", "|---|---:|---|---:|---:|---|---|---|"]
shown = set()
for r in inform[:8] + [x for x in rank if x[7].startswith("NO")][:4]:
    if (r[1], r[2], r[3]) in shown: continue
    shown.add((r[1], r[2], r[3])); tr = ", ".join(n for n, on in (("v3 C3", r[2] in TRAINED["v3_C3"]), ("B", r[2] in TRAINED["B"][r[1]])) if on) or "neither"
    o.append(f"| {AMP[r[1]]} | {r[2]:g} | {r[3]} | {r[4]:.2f} | {r[5]:.2f} | {r[6]} | {r[7]} | {tr} |")
seqs = sorted(((mean([abs(d["v3_C3"]["seq"][k]) for k in ("level_rise", "HF_change", "crest_change", "dyn_change")]) - mean([abs(d[m]["seq"][k]) for m in ("B_s0", "B_s1") for k in ("level_rise", "HF_change", "crest_change", "dyn_change")]), amp, g) for (amp, g, pk), d in by.items() if pk == "sequence"), reverse=True)
o += ["", "Soft-to-hard sequences where v3 C3 and B differ most in behaviour (positive = B keeps the real amp's soft-to-hard behaviour better): " + "; ".join(f"{AMP[a]} G{g:g} ({d:+.1f} dB)" for d, a, g in seqs[:3]) + "; smallest or reversed: " + "; ".join(f"{AMP[a]} G{g:g} ({d:+.1f} dB)" for d, a, g in seqs[-2:]) + ".", ""]
o += ["", "## 6. Reading this honestly\n",
"- These numbers can agree or disagree with what you hear. A log-spectral distance treats a 2 dB tilt and a 2 dB level-independent HF error alike, ignores playing feel, and does not weight what guitarists notice.",
"- A large predicted difference is a reason to listen to that item first, not a verdict. Items where B and v3 C3 are predicted to be similar are the ones where your ears add the most information.",
"- The earlier objective diagnostics (`docs/CONTINUOUS_GAIN_FIXED_GAIN_PLAYABILITY.md`) measured the same clips' behaviour differently (feature errors against the real capture across musical-input levels); this preview is the per-clip view of the same effects."]
OUT.write_text("\n".join(o)); print("\n".join(o))
