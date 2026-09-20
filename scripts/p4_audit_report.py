"""Render docs/CONTINUOUS_GAIN_PHASE4_QA.md from work/p4/<amp>/audit.json."""
import collections
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "work" / "p4"
NAMES = {"jcm800": "Marshall JCM800 2203 (High)", "twin": "Fender 57 Custom Twin", "supersonic": "Fender Super-Sonic, Bassman channel",
         "vibrolux": "Fender Super-Sonic, Vibrolux channel", "peavey": "Peavey 5150", "peavey6505": "Peavey 6505+ Scooped",
         "mesa": "Mesa Dual Rectifier", "orange": "Orange Dual Terror"}
out = ["# Phase 4A: capture-quality audit (all eight amp configurations)\n",
       "Reproducible: `scripts/p4_probe_captures.py <amp>` (raw, uncorrected probe bank) then `scripts/p4_audit.py <amp>` (checks, statuses); data in `work/p4/<amp>/` (`captures.json`, `audit.json`, `corrected/`). Source `.nam` files are never modified: a correction is a documented render-time shift, with original and corrected renders saved side by side. No missing capture is ever replaced by an interpolated one.\n",
       "Statuses: **VALID** no identified problem; **CORRECTED** a verified technical fault was corrected and the correction passed validation; **SUSPECT** unexplained anomaly, needs investigation (quarantine, never silently accepted or deleted); **INVALID** confirmed unrecoverable. A capture is never marked down merely for departing from a smooth or monotonic gain curve.\n",
       "Checks: timing (click onset vs set median, then confirmed on real music by cross-correlation against a neighbouring capture and reproduced on a second DI); clipping, dropouts, NaN, DC; noise floor and hum against neighbours; output-level trend, cross-checked against sine outputs and the file's own `loudness` metadata; metadata consistency; tone/dynamics deviation from neighbouring captures (raises SUSPECT only); and a model-determinism proxy. **Repeatability:** no repeat physical captures exist, so real repeatability is not measurable; the determinism proxy only shows the trained model is stable, not that the capture is.\n",
       "## Summary\n", "| Amp | VALID | CORRECTED | SUSPECT | INVALID | Not VALID |", "|---|---:|---:|---:|---:|---|"]
A = {a: json.loads((R / a / "audit.json").read_text()) for a in NAMES}
for a, d in A.items():
    c = collections.Counter(v["status"] for v in d["captures"].values())
    nv = ", ".join(f"G{k} {v['status'].lower()}" for k, v in d["captures"].items() if v["status"] != "VALID") or "-"
    out.append(f"| {NAMES[a]} | {c['VALID']} | {c['CORRECTED']} | {c['SUSPECT']} | {c['INVALID']} | {nv} |")
out += ["", "## Cases named in the brief\n"]
def ev(a, g): return A[a]["captures"][g]
j = ev("jcm800", "8.5")
out.append(f"- **JCM800 G8.5 timing:** CORRECTED. Click onset is {j['original']['onset']:+.0f} samples vs the set median (about +6); music cross-correlation gives {j['correction']['samples']:+d}-sample shift ({j['correction']['verified_by']}). After the shift the residual music lag against G8 is {j['correction']['after_correction_music_lag_vs_reference']}. Only the half-step G8.5 is affected; no integer JCM800 capture was, so the v3 training set (integers only) was not.")
p4 = ev("peavey", "4")
out.append(f"- **Peavey 5150 G4 output level:** SUSPECT, and a level problem, not a timing one. G4 is {p4['correction']['suggested_level_correction_db_estimate']:.1f} dB quieter than the interpolation of G3 and G5 at every sustained input level (sine and music), and the file's own `loudness` metadata (-16.4 vs about -12 for the neighbours) shows the same drop, while G4 has more low-level gain than G3. A constant fixed drop at saturation with both neighbours equal is not typical amp behaviour and is consistent with a capture-chain level error, but the physical amp is not available to confirm. **Suggested level correction +{p4['correction']['suggested_level_correction_db_estimate']:.1f} dB: an ESTIMATE, not applied.** Level-progression analysis should quarantine G4; its tone data are probably usable.")
pe = A["peavey"]["captures"]
shifts = {k: v["correction"]["samples"] for k, v in pe.items() if v["correction"] and "samples" in v["correction"]}
out.append(f"- **Peavey timing offsets:** confirmed real in the music, not only in the click. Corrections relative to the set median: {', '.join(f'G{k} {v:+d}' for k, v in shifts.items())} samples (G6, G9, G10 are 400-520 samples; the rest 10-50). Click-onset and music offsets differ by up to 25 samples, so music cross-correlation is the better basis. All corrections reproduce on a second DI within 3 samples.")
for a, g in (("supersonic", "10"), ("orange", "7")):
    e = ev(a, g)
    out.append(f"- **{NAMES[a]} G{g} noise:** {e['status']}. " + " ".join(x for x in e["evidence"] if "noise" in x) + " The hiss may be genuine amp noise at high gain, so this is a flag to investigate, not a fault.")
out.append("- **Super-Sonic Bassman G4, G7:** SUSPECT for the same noise-spike test (-51 and -52 dBFS on silence against about -66 for neighbours).\n")
out.append("## Consequence for the v3 results\n")
out.append("The v3 training used click-onset offsets (via `capture_lag`) for Super-Sonic, Vibrolux, Peavey 5150, 6505+, Mesa and Orange, and none for the JCM800 and Twin. The music-derived offsets here differ from the click offsets by up to about 25 samples (Peavey G3 -52 vs -50, G5 +2 vs -10, G6 +406 vs +416; Orange G9 +17 vs +9, G10 +18 vs +6), so some v3 training targets were misaligned by that much. That is small against the multi-hundred-sample offsets that were corrected, but it is the same order as A2 timing detail, so it should not be ignored when reading v3 numbers for the Peavey and Orange. Peavey G4's level error was also included in the v3 Peavey training data unmodified.\n")
out.append("## Per-capture detail\n")
for a, d in A.items():
    out.append(f"### {NAMES[a]}\n")
    out.append("| Gain | Status | Evidence | Correction |\n|---:|---|---|---|")
    for k, v in d["captures"].items():
        c = v["correction"]
        cs = "-" if not c else (f"shift {c['samples']:+d} samples ({c['basis']})" if "samples" in c else "") + (f"; suggested level {c['suggested_level_correction_db_estimate']:+.1f} dB (estimate, not applied)" if c.get("suggested_level_correction_db_estimate") else "")
        e = "; ".join(x for x in v["evidence"] if not x.startswith("info:"))[:400] or "-"
        out.append(f"| {k} | {v['status']} | {e} | {cs} |")
    out.append("")
Path(__file__).resolve().parent.parent.joinpath("docs/CONTINUOUS_GAIN_PHASE4_QA.md").write_text("\n".join(out))
print(len(out), "lines")
