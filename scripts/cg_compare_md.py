"""Rebuild docs/CONTINUOUS_GAIN_v3_COMPARISON.md from every amp that has a finished work/cg/<amp>/summary.md."""
from __future__ import annotations

from pathlib import Path

import cg_screen
import cg_summary

R = Path(__file__).resolve().parent.parent / "work" / "cg"
NAMES = {"jcm800": "Marshall JCM800 2203 (High, Gain 1-10)", "twin": "Fender 57 Custom Twin (Ch1, Volume 1-10)",
         "supersonic": "Fender Super-Sonic, Bassman channel (Volume 1-10)", "vibrolux": "Fender Super-Sonic, Vibrolux channel (Volume 1-10)",
         "peavey": "Peavey 5150 (Gain 1-10)", "peavey6505": "Peavey 6505+ Scooped (Gain 1-10)",
         "mesa": "Mesa Dual Rectifier Modern Red (Gain 1-MAX)", "orange": "Orange Dual Terror Fat (G1-G10)"}
ORDER = ["jcm800", "twin", "supersonic", "vibrolux", "peavey", "peavey6505", "mesa", "orange"]


def main():
    amps = [a for a in ORDER if (R / a / "summary.md").exists() and (R / a / "summary.md").stat().st_size]
    missing = [NAMES[a] for a in ORDER if a not in amps]
    out = ["# Continuous Gain v3: real captures vs single trained NAMs\n",
           "One standard `.nam` per configuration (10, 5 or 3 captures, A2, 60 epochs, seed 0) driven only by input gain at frozen, designed gain positions, compared with the real per-gain captures on held-out DIs. `g5` is the real G5 capture pushed by the same input gains (baseline, not a goal). Deltas are candidate minus real capture (0 = matches). lmESR = level-matched error-to-signal ratio (captures are normalised, so level is compared separately). Single seed; no human listening has happened. Raw data: `work/cg/<amp>/report_<amp>.csv` (gitignored)."
           + (f" Not yet included (still training): {', '.join(missing)}." if missing else "") + "\n",
           "## Overview (integer gains, DI level 0 dB, mean of 3 held-out DIs)\n",
           "| Amp | new10 lmESR | new5 | new3 | g5 | new10 tone err dB | new3 tone err dB | g5 tone err dB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in amps:
        rows = cg_summary.load(a)
        m = lambda s, k: cg_summary.mean([r for r in rows if r["kind"] == "music" and r["source"] == s and r["offset_db"] == 0.0 and r["gain"] == int(r["gain"])], k)
        out.append(f"| {NAMES[a]} | {m('new10','lm_esr'):.3f} | {m('new5','lm_esr'):.3f} | {m('new3','lm_esr'):.3f} | {m('g5','lm_esr'):.3f} | {m('new10','band_err_db'):.2f} | {m('new3','band_err_db'):.2f} | {m('g5','band_err_db'):.2f} |")
    out += ["", "Read with care: Peavey and Super-Sonic captures have timing offsets (see each amp's QA); heavily saturated material makes ESR a weak guide, so use the tone, EQ, saturation and dynamics columns as well. The Fender models compress and distort more than the real amp at high input levels (see each amp's saturation probe: out-level change for +24 dB input).\n",
            "## Capture pre-screen (prototype, `scripts/cg_screen.py`)\n", f"Flags captures that break the amp's smooth gain curve, timing offsets, and noisy captures. Nothing was excluded on any of the {len(amps)} amps below.\n"]
    for a in amps:
        out.append(f"- **{a}**: " + ("; ".join(f"G{g:g} {st}: {why}" for g, st, r, why in cg_screen.screen(a) if st != "OK") or "no issues"))
    out.append("")
    out += [(R / a / "summary.md").read_text() for a in amps]
    p = Path(__file__).resolve().parent.parent / "docs" / "CONTINUOUS_GAIN_v3_COMPARISON.md"
    p.write_text("\n".join(out))
    print(p, "amps:", amps)


if __name__ == "__main__":
    main()
