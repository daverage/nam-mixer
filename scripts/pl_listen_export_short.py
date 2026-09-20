"""Export the shortlist listening audio with descriptive (unblinded) names plus a document describing every file.
Usage: python scripts/pl_listen_export_short.py [output_dir]   (default work/p4e/listening_fixed_gain_short)"""
import csv, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import pl_listen_short as S
import pl_listening_analyze as A
import numpy as np, soundfile as sf
from hybrid.audio_metrics import multi_resolution_log_spectral_distance as _mrlsd
LG = S.LG; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "p4e" / "listening_fixed_gain_short"
key, pub = S.key, S.pub
stage = {}
for amp in ("jcm800", "vibrolux"):
    j = json.loads((LG / "_stage" / f"stage_{amp}.json").read_text()); stage[amp] = {s["sid"]: s for s in j["stimuli"]}; stage[amp]["_meta"] = j
res = ROOT / "docs" / "phase4e" / "listening_fixed_gain" / "results" / "fixed_gain_listening_results_2026-09-20.csv"
rows = A.load([res], key, 50.0) if res.exists() else []
rating = {(r["sid"], r["model"]): (r["score"], r.get("imputed", False), r["issues"]) for r in rows}
auto = {}
am = LG / "auto_metrics.json"
if am.exists():
    for r in json.loads(am.read_text()): auto[(r["amp"], r["gain"], r["pick"], r["model"])] = r
NAME = {"jcm800": "Marshall JCM800", "vibrolux": "Fender Super-Sonic Vibrolux"}; DI = {"jcm800": "moderate_brit", "vibrolux": "clean_mayer"}
MODEL_FILE = {"v3_C3": "v3_C3", "B_s0": "B_seed0", "B_s1": "B_seed1"}
MODEL_DESC = {"REAL": "the real amp capture at this setting (the reference)", "v3_C3": "old single NAM trained on physical captures G1, G5, G10 (v3 C3), played at its own fixed-rule Input gain",
              "B_s0": "Phase 4E B, seed 0 (trained on {b}), played at its response-distance Input gain", "B_s1": "Phase 4E B, seed 1 (same captures and settings as seed 0, different training seed)"}
BSET = {"jcm800": "G1, G2, G4, G10", "vibrolux": "G1, G2, G3, G4, G7, G10"}
TINFO = {}
for amp in ("jcm800", "vibrolux"):
    tp = LG / "_stage" / f"teacher_info_{amp}.json"
    if tp.exists(): TINFO.update(json.loads(tp.read_text()))
def _rd(pth): return sf.read(pth, dtype="float32")[0]
def _rms_db(y): return 20 * np.log10(np.sqrt(np.mean(y[24000:].astype(np.float64) ** 2)) + 1e-12)
OUT.mkdir(parents=True, exist_ok=True); md = []
md += ["# Fixed-gain listening: shortlist audio (10 items)\n",
"Every file here is one of the clips used in the blind listening test, renamed by what it actually is. **These names are unblinded** (the test itself used random letters A-D); the hidden 'real amp' copy in each blind item was a bit-identical copy of the REAL file, so it is not duplicated here.\n",
"## How to read the folders\n",
"One folder per item, in the order of the listening page (`01_` to `10_`). Each folder holds four amp sounds (REAL and three models), each in two versions, plus the dry DI and the two teacher renders:\n",
"- **`REAL`**: the real amp capture at this virtual gain, with the same guitar performance.\n- **`v3_C3`**: the old single NAM (trained on G1, G5, G10).\n- **`B_seed0`**, **`B_seed1`**: the two Phase 4E B models.\n- **`_native`**: relative loudness preserved (the real level differences between REAL and each model are kept); **`_levelmatched`**: each model is scaled to the REAL file's loudness, so you compare tone, saturation and dynamics without loudness. Use both.\n",
"## What the extra files are (dry DI and teacher)\n",
"- **`DI_dry_original.wav`**: the guitar recording exactly as recorded (the same window, no level change, no amp). For the sequence item it is the 4 s phrase once. **`DI_musical_input.wav`**: the same recording after the picking-intensity level change (soft -12 dB, normal 0, hard +6 dB), i.e. what the guitarist plays into the player; for the sequence item it is the soft, normal and hard versions in a row. Both carry the same +4.5 dB makeup gain as the amp clips so a fixed listening volume gives comparable loudness (the real recording is 4.5 dB quieter). The waveform that actually reaches the NAM is this file multiplied by the Input gain (up to +14 dB), which would exceed digital full scale, so it is not stored.\n",
"- **`TEACHER_B`** and **`TEACHER_v3C3`**: the *training target* each model was trained to imitate. It is built from the real amp captures only (no neural network): at every moment it blends the two nearest real captures of that model's training set, chosen by the level of the signal reaching the NAM (Input gain applied), each real capture driven at its reference level. B's two seeds share one teacher; v3 C3 has its own. Listening order that answers whether a problem is in the target or in the trained model: **REAL, then TEACHER, then the model.** If the teacher already differs from REAL, the target construction is responsible; if the teacher matches REAL and the model does not, the training is. The tables give mean blend weights on the training captures for each clip (over the active part of the clip).\n",
"## How the clips were made\n",
"The guitar is a held-out DI recording (`clean_mayer` for the Vibrolux, `moderate_brit` for the JCM800), never used in training. Picking intensity is a global level change of that recording: **soft = -12 dB, normal = 0 dB, hard = +6 dB**. For each item the player's **Input gain is fixed** at the value shown, and only the playing changes. REAL is the original capture of that amp setting driven by the DI at the chosen level (no Input gain). Each model output is scaled by one fixed constant so its level is comparable to the real amp (the JCM800 models' training constant is undone; the Vibrolux's is 1.0), and every file has the same +4.5 dB makeup gain so quiet clips are audible. Format: WAV, 16-bit, 48 kHz, mono. Single clips are 5 s; the sequence clip is 13 s: **0-4 s soft, 4.5-8.5 s normal, 9-13 s hard** (0.5 s gaps), the same phrase each time.\n",
"'Your rating' is the closeness score you gave in the blind test (0-100; a `*` means the slider was untouched and counted as 50). 'MR-LSD' is an objective spectral distance to REAL on the level-matched versions (lower = closer; not a perceptual score); 'level vs real' is the native level difference in dB.\n"]
for n, (amp, g, pk) in enumerate(S.SHORT, 1):
    sid = [s for s, k in key.items() if k["amp"] == amp and k["virtual_gain"] == g and k["pick"] == pk][0]; k = key[sid]; p = pub[sid]
    st = stage[amp][k["stage_id"]]; seq = k["kind"] == "sequence"; pkn = "soft-normal-hard sequence" if seq else pk
    folder = OUT / f"{n:02d}_{'Vibrolux' if amp == 'vibrolux' else 'JCM800'}_G{g:g}_{pkn.replace(' ', '_').replace('-', '_')}"; folder.mkdir(exist_ok=True)
    ws = st["window_start_s"]; ig = st["input_gain_db"]
    md += [f"## {n:02d}. {NAME[amp]}, virtual gain {g:g}, {pkn}\n",
           f"Guitar: `{DI[amp]}`, clip starting {ws:.1f} s into the recording, {st['duration_s']:g} s. " + ("Segments: soft (-12 dB), then normal, then hard (+6 dB) at a fixed Input gain. " if seq else f"Picking: {pk} ({st['pick_offset_db']:+g} dB). ") +
           f"Input gain used: v3 C3 **{ig['v3_C3']:+.1f} dB**, B seeds **{ig['B_s0']:+.1f} dB**; REAL: none (the original capture at this setting).\n",
           "| File | What it is | Your rating | Issues you ticked | MR-LSD to REAL | level vs REAL (dB) |", "|---|---|---:|---|---:|---:|"]
    for m, lab in (("REAL", None), ("v3_C3", None), ("B_s0", None), ("B_s1", None)):
        L = None if m == "REAL" else [x for x, mm in k["labels"].items() if mm == m][0]
        for mode in ("native", "levelmatched"):
            src = LG / (p["files"]["REFERENCE"][mode] if m == "REAL" else p["files"][L][mode])
            base = "REAL" if m == "REAL" else MODEL_FILE[m]; dst = folder / f"{base}_{mode}.wav"; shutil.copyfile(src, dst)
        r = rating.get((sid, m)); a = auto.get((amp, g, pk if not seq else "sequence", m))
        desc = MODEL_DESC[m].format(b=BSET[amp]) if m == "B_s0" else MODEL_DESC[m]
        rt = "-" if m == "REAL" else (f"{r[0]:.0f}{'*' if r[1] else ''}" if r else "-"); isu = "; ".join(r[2]) if r and r[2] else "-"
        md.append(f"| `{folder.name}/{'REAL' if m == 'REAL' else MODEL_FILE[m]}_native.wav` and `_levelmatched.wav` | {desc} | {rt} | {isu} | " + ("-" if m == "REAL" or not a else f"{a['mrlsd']:.2f}") + " | " + ("-" if m == "REAL" or not a else f"{a['d_level']:+.1f}") + " |")
    sd = LG / "_stage" / amp / k["stage_id"]; ti = TINFO.get(k["stage_id"])
    if ti and (sd / "TEACHER_B__native.wav").exists():
        for nm in ("DI_dry_original", "DI_musical_input"): shutil.copyfile(sd / f"{nm}__native.wav", folder / f"{nm}.wav")
        refl = _rd(sd / "REFERENCE__levelmatched.wav"); refn = _rd(sd / "REFERENCE__native.wav")
        for t, lab in (("B", "TEACHER_B"), ("v3C3", "TEACHER_v3C3")):
            for mode in ("native", "levelmatched"): shutil.copyfile(sd / f"{lab}__{mode}.wav", folder / f"{lab}_{mode}.wav")
            tl = _rd(sd / f"{lab}__levelmatched.wav"); tn = _rd(sd / f"{lab}__native.wav"); n_ = min(len(tl), len(refl)); an = [f"G{g_:g} {w:.2f}" for g_, w in zip(ti["anchors"][t], ti["weights_mean_active"][t])]
            md.append(f"| `{folder.name}/{lab}_native.wav` and `_levelmatched.wav` | teacher for {'B (both seeds)' if t == 'B' else 'v3 C3'}: real-capture blend at Input gain {ti['input_gain_db'][t]:+.1f} dB; mean weights: " + ", ".join(an) + f" | - | - | {float(_mrlsd(tl[:n_], refl[:n_])):.2f} | {_rms_db(tn) - _rms_db(refn):+.1f} |")
        md.append("| `" + folder.name + "/DI_dry_original.wav`, `DI_musical_input.wav` | the guitar recording as recorded, and after the picking-intensity level change (no amp) | - | - | - | - |")
        st_t = lambda m, lab: float(_mrlsd(_rd(sd / f"{m}__levelmatched.wav")[:len(_rd(sd / f"{lab}__levelmatched.wav"))], _rd(sd / f"{lab}__levelmatched.wav")[:len(_rd(sd / f"{m}__levelmatched.wav"))]))
        md.append("")
        md.append(f"Trained model versus its own teacher (MR-LSD, level-matched, lower = closer): B seed 0 {st_t('B_s0', 'TEACHER_B'):.2f}, B seed 1 {st_t('B_s1', 'TEACHER_B'):.2f}, v3 C3 {st_t('v3_C3', 'TEACHER_v3C3'):.2f}. The teacher rows' MR-LSD and level columns compare the teacher with REAL.\n")
    hid = rating.get((sid, "HIDDEN_REFERENCE"))
    md += ["", f"In the blind test a hidden copy of REAL was also in this item; you rated it {'%.0f%s' % (hid[0], '*' if hid[1] else '')}.\n" if hid else ""]
md += ["## Related documents\n", "- `docs/phase4e/listening_fixed_gain/RESULTS_first_listening.md`: the analysis of your ratings.\n- `docs/phase4e/listening_fixed_gain/AUTOMATED_PREVIEW.md`: the objective distances for all clips.\n- `docs/phase4e/listening_fixed_gain/LIVE_GUITAR_TEST_QUICK.md`: the matching live-guitar test (Input gains per setting).\n- Models: `docs/phase4e/models/` (B) and `docs/phase4e/models/v3/` (v3 C3).\n"]
(OUT / "README.md").write_text("\n".join(md)); print("exported", len(list(OUT.rglob("*.wav"))), "audio files to", OUT)
