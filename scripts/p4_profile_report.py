"""Render docs/CONTINUOUS_GAIN_PHASE4_PROFILES.md and copy plots to docs/phase4/."""
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "work" / "p4"
(ROOT / "docs" / "phase4").mkdir(exist_ok=True)
NAMES = {"jcm800": "Marshall JCM800 2203 (High)", "twin": "Fender 57 Custom Twin", "supersonic": "Fender Super-Sonic, Bassman channel",
         "vibrolux": "Fender Super-Sonic, Vibrolux channel", "peavey": "Peavey 5150", "peavey6505": "Peavey 6505+ Scooped",
         "mesa": "Mesa Dual Rectifier", "orange": "Orange Dual Terror"}
fmt = lambda iv: ", ".join(f"G{a:g}-G{b:g}" for a, b in iv) or "none"
def merge(iv):
    out = []
    for a, b in iv:
        if out and out[-1][1] == a: out[-1][1] = b
        else: out.append([a, b])
    return out
out = ["# Phase 4B: measured response profiles (all eight amps)\n",
"Built from the raw probe bank (`scripts/p4_probe_captures.py`) by `scripts/p4_profile.py`; data `work/p4/<amp>/profile.json`, plots `docs/phase4/profile_<amp>.png`. Level, tone/EQ, saturation, compression, dynamics and noise are kept as **separate dimensions**; nothing is collapsed into one gain curve. In the plots, markers are measured, dashed lines are shape-preserving (PCHIP) estimates between measured positions with a band showing the median leave-one-out error, and red rings mark a capture quarantined for that dimension only (level for Peavey G4 and Orange G2; noise for the noise-flagged captures). Plateaus and non-monotonic behaviour are kept as measured. Source material: the four fit DIs at native level and offsets -12/-6/+6 dB, plus sine sweeps (110/440 Hz, -54 to 0 dBFS RMS).\n",
"**How to read 'fast' and 'stable':** each dimension's change between adjacent positions is normalised by that dimension's total range; an interval is *fast* if it changes at more than 1.5x the amp's median rate and *stable* below 0.5x. These are relative to each amp's own behaviour.\n"]
for amp, nm in NAMES.items():
    P = json.loads((R / amp / "profile.json").read_text())
    shutil.copy(R / amp / "profile.png", ROOT / "docs" / "phase4" / f"profile_{amp}.png")
    out.append(f"## {nm}\n")
    out.append(f"![{amp} profile](phase4/profile_{amp}.png)\n")
    out.append("| Dimension | Changes quickly | Stable |\n|---|---|---|")
    for dim, r in P["regions"].items():
        out.append(f"| {dim} | {fmt(merge(r['fast']))} | {fmt(merge(r['stable']))} |")
    out.append("")
    pl = P["level_plateau_but_character_changing"]
    out.append(f"- **Output-level plateau while character still changes:** {fmt(merge(pl)) if pl else 'none detected'}.")
    q = {k: v for k, v in P["quarantine"].items() if v}
    out.append(f"- **Quarantined from estimates:** " + (", ".join(f"{k}: " + ", ".join(f"G{g:g}" for g in v) for k, v in q.items()) if q else "none") + ".")
    lv = P["series"]["music RMS @0 dB DI"]["values"]; gs = P["gains"]
    ints = [g for g in gs if g == int(g)]
    # character-stable point: last interval with any dimension 'fast'
    last_fast = max((b for r in P["regions"].values() for a, b in r["fast"]), default=None)
    out.append(f"- **Last position where any dimension is still changing quickly:** {'G%g' % last_fast if last_fast else 'n/a'}; above it the response is comparatively stable.")
    hs = [(n, e["half_step_true_err_median"] / e["range"]) for n, e in P["series"].items() if "half_step_true_err_median" in e]
    if hs:
        out.append(f"- **Independent check of the estimate (genuine half-step captures, never used to fit):** the PCHIP curve through the integer positions predicts the measured half-steps to a median error of {np.median([h for _, h in hs])*100:.1f}% of each series' range (worst series: {max(hs, key=lambda x: x[1])[0]}, {max(hs, key=lambda x: x[1])[1]*100:.1f}%). This is the real, measured uncertainty of interpolating between captures on this amp.")
    else:
        loo = [e["loo_median_abs_err"] / e["range"] for e in P["series"].values() if e["loo_median_abs_err"]]
        out.append(f"- **Interpolation uncertainty (leave-one-out, no half-steps available):** median {np.median(loo)*100:.1f}% of each series' range.")
    out.append("")
(ROOT / "docs" / "CONTINUOUS_GAIN_PHASE4_PROFILES.md").write_text("\n".join(out))
shutil.copy(R / "mapping_curves.png", ROOT / "docs" / "phase4" / "mapping_curves.png")
print(len(out), "lines")
