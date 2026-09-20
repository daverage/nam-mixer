"""Phase 4E listening package (reproducible, seeded). Reference set is labelled; the three candidates (v3 C3, A, B) are blind and randomised.
Native-level and level-matched versions. Usage: p4e_listening.py  (after training + p4e_eval)"""
import csv, json, os, random, sys
os.environ["SINGLE_NAM_AMP"] = "jcm800"
import numpy as np, soundfile as sf
from pathlib import Path
import p4e_common as C
from single_nam_common import render_file_nam
SR = C.SR; SECS = 6
SETTINGS = {"cleaner": 2.0, "transition": 4.0, "saturated": 7.0, "top": 10.0}
PICK = {"normal": 0.0, "soft": -12.0, "hard": 6.0}
DIS = ["clean_mayer", "moderate_brit"]
SEED_LIST = {"reference": "real capture (4A-aligned)", "cand": ["v3_C3", "A_s0", "B_s0"]}
OUT = C.P4E / "listening"; rng = random.Random(20260920)
key_rows, ws_rows = [], []
def wr(path, y): path.parent.mkdir(parents=True, exist_ok=True); sf.write(path, np.clip(y, -1, 1), SR, subtype="PCM_16")
for amp in ("jcm800", "vibrolux"):
    os.environ["SINGLE_NAM_AMP"] = amp
    import importlib; importlib.reload(sys.modules["single_nam_common"]); from single_nam_common import capture as cap
    C.capture = cap
    AUD = json.loads((C.REPO / "work" / "p4" / amp / "audit.json").read_text())["captures"]
    ev = {k: json.loads((C.P4E / amp / f"eval_{k}.json").read_text()) for k in SEED_LIST["cand"]}
    def T_of(k, g): return ev[k]["mappings"]["intended"][ev[k]["gains"].index(g)]
    nam = {k: C.model_path(amp, k) for k in SEED_LIST["cand"]}
    n = 0
    for sname, g in SETTINGS.items():
        for di in DIS:
            for pk, off in PICK.items():
                if pk != "normal" and sname not in ("transition", "saturated"): continue
                x = C.clip(di)[: SECS * SR]
                ref = C.render(cap(g), (x * C.db(off)).astype(np.float32), SR)
                cs = C.AUD if False else None
                cands = {}
                for k in SEED_LIST["cand"]:
                    path, c = nam[k]; cands[k] = render_file_nam(path, (x * C.db(T_of(k, g) + off)).astype(np.float32)) / c
                sid = f"{amp}_{sname}_G{g:g}_{di}_{pk}"
                labels = ["X1", "X2", "X3"]; rng.shuffle(labels); lab = dict(zip(SEED_LIST["cand"], labels))
                for mode in ("native", "levelmatched"):
                    ys = {"REFERENCE": ref, **{lab[k]: (cands[k] if mode == "native" else cands[k] * np.sqrt(np.mean(ref[SR//2:] ** 2) / max(np.mean(cands[k][SR//2:] ** 2), 1e-20))) for k in cands}}
                    peak = max(np.max(np.abs(v)) for v in ys.values()); gain = min(1.0, 0.95 / peak)   # one common gain per stimulus: relative levels preserved
                    for name, y in ys.items():
                        sub = "reference" if name == "REFERENCE" else f"blind_{mode}"
                        wr(OUT / amp / sub / f"{sid}__{'REF' if name=='REFERENCE' else name}__{mode}.wav", y * gain)
                key_rows.append({"stimulus": sid, **{lab[k]: k for k in cands}})
                ws_rows.append({"stimulus": sid, "amp": amp, "setting": f"{sname} (G{g:g})", "DI": di, "picking": pk, "closest_to_reference_1st": "", "2nd": "", "3rd": "", "undesirable (harsh/fizzy/dull/thin/boomy/squashed/too clean/noisy/artefacts)": "", "loudness progression comment": "", "notes": ""})
                n += 1
    print(amp, n, "stimuli")
(OUT).mkdir(exist_ok=True, parents=True)
with open(OUT / "listening_KEY_do_not_open_before_listening.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["stimulus", "X1", "X2", "X3"]); w.writeheader(); w.writerows(key_rows)
with open(OUT / "listening_worksheet.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ws_rows[0])); w.writeheader(); w.writerows(ws_rows)
(OUT / "README.md").write_text("""# Phase 4E listening package (STATUS: PERCEPTUAL EVALUATION PENDING - no human listening has happened)

- `<amp>/reference/`: the real capture (labelled REF) for every stimulus. `<amp>/blind_native/` and `<amp>/blind_levelmatched/`: three candidates per stimulus with randomised labels X1-X3 (old v3 C3, new A seed 0, new B seed 0; key in `listening_KEY_do_not_open_before_listening.csv`).
- Native-level files preserve relative loudness between reference and candidates (one common gain per stimulus to avoid clipping); level-matched files match each candidate's RMS to the reference so tone, saturation and dynamics can be judged without loudness dominating.
- Settings: cleaner (G2), transition (G4), saturated (G7), top (G10); soft (-12 dB) and hard (+6 dB) picking at transition and saturated; DIs clean_mayer and moderate_brit (held-out, never in training). Each candidate is played at the intended playback mapping of its own configuration (v3 C3 and A: fixed rule; B: response-distance rule).
- Fill in `listening_worksheet.csv` (which candidate is closest to the reference, what is undesirable, loudness progression). Open the key only afterwards.
""")
print("written", OUT)
