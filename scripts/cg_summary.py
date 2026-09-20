"""Markdown comparison tables from cg_report CSVs: per amp, real capture vs new NAMs vs G5-pushed baseline.
Deltas are (candidate - real capture) so 0 = matches. Usage: cg_summary.py <amp>... > out.md"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "work" / "cg"
EQ = ["sub", "low", "lowmid", "mid", "presence", "air"]


def load(amp):
    with open(ROOT / amp / f"report_{amp}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("amp", "kind", "source", "di"):
                r[k] = float(v) if v not in ("", None) else float("nan")
    return rows


def mean(rows, key):
    v = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(v)) if v else float("nan")


def main(amps):
    for amp in amps:
        rows = load(amp)
        sources = [s for s in ("new10", "new5", "new3", "g5") if any(r["source"] == s for r in rows)]
        gains = sorted({r["gain"] for r in rows})
        ints = [g for g in gains if g == int(g)]
        qa = json.loads((ROOT / amp / f"capture_qa_{amp}.json").read_text())
        print(f"\n## {amp}\n")
        lat = qa["click_latency_samples"]
        med = float(np.median(list(lat.values())))
        odd = {g: v for g, v in lat.items() if abs(v - med) > 20}
        print(f"**Capture QA.** Click latency median {med:g} samples; anomalies: {odd or 'none'}. "
              f"Reversals in real-capture progression (integer gains; 0 = monotone): RMS {qa['real_rms_db_reversals']}, "
              f"HF>3k {qa['real_hf3k_db_reversals']}, crest {qa['real_crest_db_reversals']}. "
              f"Silence noise floor {min(qa['silence_noise_db'].values()):.0f} to {max(qa['silence_noise_db'].values()):.0f} dBFS.\n")
        real_rms = qa["real_rms_db_by_gain"]; real_hf = qa["real_hf3k_db_by_gain"]; real_cr = qa["real_crest_db_by_gain"]
        print("Real capture progression (music at reference level, mean of DIs): RMS dBFS / HF>3k dB / crest dB\n")
        print("| Gain | " + " | ".join(f"G{int(g)}" for g in ints) + " |\n|---|" + "---:|" * len(ints))
        for name, v in (("RMS", real_rms), ("HF>3k", real_hf), ("crest", real_cr)):
            print(f"| {name} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
        print()
        for s in sources:
            print(f"\n### {amp}: {s} vs real capture (music, DI level 0 dB; mean of 3 held-out DIs; deltas = candidate - real)\n")
            print("| Gain | lmESR | tone err dB | max EQ band Δ dB | tilt Δ | HF>3k Δ | crest Δ | dyn-range Δ | level Δ dB | THD@440 Δ dB |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for g in gains:
                m = [r for r in rows if r["kind"] == "music" and r["gain"] == g and r["offset_db"] == 0.0]
                rl = {(r["di"]): r for r in m if r["source"] == "real"}
                cd = [r for r in m if r["source"] == s]
                if not cd:
                    continue
                d = lambda k: float(np.mean([r[k] - rl[r["di"]][k] for r in cd]))
                eqmax = float(np.mean([max(abs(r[f"eq_{b}_db"] - rl[r["di"]][f"eq_{b}_db"]) for b in EQ) for r in cd]))
                tn = [r for r in rows if r["kind"] == "tone" and r["gain"] == g and r["offset_db"] == 0.0 and r["di"] == "sine440"]
                tr = [r for r in tn if r["source"] == "real"]; tc = [r for r in tn if r["source"] == s]
                thd = tc[0]["thd_db"] - tr[0]["thd_db"] if tr and tc else float("nan")
                mark = "*" if g not in (5.0,) and False else ""
                print(f"| {g:g} | {mean(cd,'lm_esr'):.3f} | {mean(cd,'band_err_db'):.2f} | {eqmax:.2f} | {d('tilt_db'):+.2f} | {d('hf3k_db'):+.2f} | "
                      f"{d('crest_db'):+.2f} | {d('dyn_range_db'):+.2f} | {d('rms_db'):+.2f} | {thd:+.1f} |")
        print(f"\n### {amp}: DI level sweep (integer gains; mean lmESR / tone err dB)\n")
        offs = sorted({r["offset_db"] for r in rows if r["kind"] == "music"})
        print("| DI offset dB | " + " | ".join(sources) + " |\n|---:|" + "---:|" * len(sources))
        for o in offs:
            cells = []
            for s in sources:
                c = [r for r in rows if r["kind"] == "music" and r["source"] == s and r["offset_db"] == o and r["gain"] == int(r["gain"])]
                cells.append(f"{mean(c,'lm_esr'):.3f} / {mean(c,'band_err_db'):.2f}")
            print(f"| {o:+g} | " + " | ".join(cells) + " |")
        print(f"\n### {amp}: saturation probe (sine THD dB @ 0 dB / compression: out level change for +24 dB in, at 440 Hz)\n")
        print("| Gain | real THD | " + " | ".join(f"{s} THD" for s in sources) + " | real dOut | " + " | ".join(f"{s} dOut" for s in sources) + " |")
        print("|---:|---:|" + "---:|" * len(sources) + "---:|" + "---:|" * len(sources))
        for g in ints:
            def t(src, off): 
                x = [r for r in rows if r["kind"] == "tone" and r["source"] == src and r["gain"] == g and r["di"] == "sine440" and r["offset_db"] == off]
                return x[0] if x else None
            dout = lambda src: (t(src, 12.0)["out_rms_db"] - t(src, -12.0)["out_rms_db"]) if t(src, 12.0) and t(src, -12.0) else float("nan")
            print(f"| {g:g} | {t('real',0.0)['thd_db']:.1f} | " + " | ".join(f"{t(s,0.0)['thd_db']:.1f}" for s in sources) + f" | {dout('real'):.1f} | " + " | ".join(f"{dout(s):.1f}" for s in sources) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
