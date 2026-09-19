"""Render the markdown tables/aggregates for
docs/CONTINUOUS_GAIN_MID_GAIN_VS_PROFILE.md from the JSON produced by
`scripts/continuous_gain_mid_vs_profile_benchmark.py`.

Pure formatting -- no rendering, no metrics, no fitting happens here. Kept
separate so the (long) benchmark never has to be re-run to change how a
table is presented.

Usage:
    python scripts/continuous_gain_mid_vs_profile_report.py \
        [--in work/continuous_gain_mid_vs_profile/results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS = ["A_direct", "A_calibrated", "B_profile", "C_dense"]
METHOD_TITLES = {"A_direct": "A-direct (G5 + linear dB law)",
                 "A_calibrated": "A-calibrated (G5 + fitted mapping)",
                 "B_profile": "B (multi-anchor profile)",
                 "C_dense": "C (dense interpolation)"}


def usable(row: dict) -> bool:
    """A row counts toward an aggregate only if it is a genuine interpolation
    test: not the anchor's own position, not a clamped/extrapolated mapping,
    and not a capture with an isolated timing defect."""
    return not (row.get("anchor_self_match") or row.get("extrapolated_mapping") or row.get("timing_defect"))


def agg(rows: list[dict], di: str, method: str, key: str = "raw_esr") -> tuple[float, float]:
    vals = [r["metrics"][di][method][key] for r in rows if usable(r) and di in r["metrics"]]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.max(vals))


def fmt(v: float, nd: int = 4) -> str:
    if v != v:
        return "n/a"
    return f"{v:.{nd}f}"


def per_amp_table(amp: dict, di: str) -> str:
    lines = [f"| {amp['control_label']} | Flags | A-direct gain | A-cal gain | B gain | "
             "A-direct ESR | A-cal ESR | B ESR | C ESR | A-cal lvl Δ dB | B lvl Δ dB | A-cal spec | B spec |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in amp["withheld_rows"]:
        if di not in r["metrics"]:
            continue
        m = r["metrics"][di]
        flags = []
        if r.get("anchor_self_match"):
            flags.append("anchor")
        if r.get("extrapolated_mapping"):
            flags.append("**extrap**")
        if r.get("timing_defect"):
            flags.append("**defect**")
        lines.append(
            f"| {r['position']:g} | {' '.join(flags) or '--'} | "
            f"{r['gains']['A_direct_db']:+.1f} | {r['gains']['A_calibrated_db']:+.1f} | "
            f"{r['gains']['B_profile_db']:+.1f} | "
            f"{fmt(m['A_direct']['raw_esr'])} | {fmt(m['A_calibrated']['raw_esr'])} | "
            f"{fmt(m['B_profile']['raw_esr'])} | {fmt(m['C_dense']['raw_esr'])} | "
            f"{m['A_calibrated']['level_delta_db']:+.2f} | {m['B_profile']['level_delta_db']:+.2f} | "
            f"{m['A_calibrated']['spectral_correlation']:.3f} | {m['B_profile']['spectral_correlation']:.3f} |")
    means = {mth: agg(amp["withheld_rows"], di, mth) for mth in METHODS}
    lines.append(f"| **mean (usable)** | | | | | " + " | ".join(fmt(means[m][0]) for m in METHODS)
                 + " | | | | |")
    lines.append(f"| **worst (usable)** | | | | | " + " | ".join(fmt(means[m][1]) for m in METHODS)
                 + " | | | | |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path,
                    default=_REPO_ROOT / "work/continuous_gain_mid_vs_profile/results.json")
    args = ap.parse_args()
    data = json.loads(args.inp.read_text())

    print(f"# generated {data['generated_utc']}  total {data.get('total_wall_clock_s', 0)/60:.1f} min\n")

    print("## Cross-amp summary (primary DI, usable withheld positions only)\n")
    print("| Amp | Split | n usable | " + " | ".join(f"{METHOD_TITLES[m]} mean/worst" for m in METHODS) + " |")
    print("|---|---|---:|" + "---:|" * len(METHODS))
    for amp in data["amps"]:
        if amp.get("skipped"):
            continue
        n = sum(1 for r in amp["withheld_rows"] if usable(r))
        cells = []
        for m in METHODS:
            mean, worst = agg(amp["withheld_rows"], data["primary_di"], m)
            cells.append(f"{fmt(mean)} / {fmt(worst)}")
        print(f"| {amp['name']} | {amp['split']} | {n} | " + " | ".join(cells) + " |")

    for amp in data["amps"]:
        if amp.get("skipped"):
            print(f"\n## {amp['amp']}: SKIPPED ({amp.get('missing')})")
            continue
        print(f"\n\n## {amp['name']}\n")
        print(f"split={amp['split']}  timing_defects={amp['timing_defects']}  "
              f"wall={amp.get('wall_clock_s', 0)/60:.1f} min")
        print(f"\nnotes: {amp['notes']}\n")
        print("### Measured G5-anchor mapping (fit on training positions only)\n")
        print("| position | optimal input gain (dB) | search raw ESR | boundary-pinned |")
        print("|---:|---:|---:|---|")
        for p, v in sorted(amp["measured_anchor_mapping"].items(), key=lambda kv: float(kv[0])):
            pin = "lower" if v["hit_lower_bound"] else ("upper" if v["hit_upper_bound"] else "--")
            print(f"| {float(p):g} | {v['input_gain_db']:+.1f} | {v['raw_esr']:.4f} | {pin} |")
        slope = amp["withheld_rows"][0]["direct_slope_db_per_step"] if amp["withheld_rows"] else float("nan")
        print(f"\nConventional linear law fitted slope: **{slope:+.2f} dB per knob step**\n")

        print(f"### Withheld positions -- primary DI ({data['primary_di']})\n")
        print(per_amp_table(amp, data["primary_di"]))

        print("\n### Anchor sets the PRODUCTION builder actually selected, per fold\n")
        print("| withheld | anchors selected | build quality | worst build-time ESR | region used |")
        print("|---:|---|---|---:|---|")
        for k, v in amp["fold_profiles"].items():
            if k.startswith("_"):
                continue
            print(f"| {float(k):g} | {v['anchor_positions']} | {v['quality']} | "
                  f"{v['worst_raw_esr_build_time']:.4f} | {v['region_used']} |")

        if amp["train_rows"]:
            print("\n### In-sample (training) positions, primary DI -- for train-vs-withheld contrast\n")
            print("| position | A-direct ESR | A-cal ESR | B ESR |")
            print("|---:|---:|---:|---:|")
            for r in amp["train_rows"]:
                m = r["metrics"][data["primary_di"]]
                print(f"| {r['position']:g} | {fmt(m['A_direct']['raw_esr'])} | "
                      f"{fmt(m['A_calibrated']['raw_esr'])} | {fmt(m['B_profile']['raw_esr'])} |")
            for label, sel in (("all", lambda r: True), ("excl. anchor", lambda r: not r["anchor_self_match"])):
                vals = {m: [r["metrics"][data["primary_di"]][m]["raw_esr"]
                            for r in amp["train_rows"] if sel(r)] for m in METHODS[:3]}
                print(f"\nin-sample mean ({label}): " +
                      ", ".join(f"{m} {np.mean(v):.4f}" for m, v in vals.items()))

        print("\n### Multi-DI generalization (mappings/anchors frozen on the primary DI)\n")
        print("| DI | " + " | ".join(f"{METHOD_TITLES[m]} mean/worst" for m in METHODS) + " |")
        print("|---|" + "---:|" * len(METHODS))
        for di in ("standard", "clean", "metalcore"):
            cells = []
            any_data = False
            for m in METHODS:
                mean, worst = agg(amp["withheld_rows"], di, m)
                if mean == mean:
                    any_data = True
                cells.append(f"{fmt(mean)} / {fmt(worst)}")
            if any_data:
                print(f"| {di} | " + " | ".join(cells) + " |")

        print("\n### Timing diagnostics (isolated defects)\n")
        for p, v in sorted(amp["timing_diagnostics"].items(), key=lambda kv: float(kv[0])):
            if v["defective"]:
                print(f"- position {float(p):g}: lags "
                      f"{[(e['neighbour'], e['lag_samples']) for e in v['neighbour_lags']]}, "
                      f"neighbours-skipping-it {v['neighbours_skipping_this_capture']}")

    cont = data.get("continuity")
    if cont:
        print(f"\n\n## Runtime continuity / cost ({cont['amp']}, anchors {cont['profile_anchors']})\n")
        print("| method | models resident | resident NAM bytes | mean wall s/position | "
              "max wall s/position | total child CPU s | peak child RSS MB |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for m, c in cont["runtime_cost"].items():
            print(f"| {m} | {c['nam_models_resident']} | {c['nam_model_bytes_resident']} | "
                  f"{c['mean_wall_s_per_position']:.3f} | {c['max_wall_s_per_position']:.3f} | "
                  f"{c['total_child_cpu_s']:.1f} | {c['peak_child_rss_bytes']/1e6:.1f} |")
        print("\n| sweep | method | max frame level jump dB | mean frame jump dB | frames >3 dB | "
              "max |Δsample| | samples >0.25 Δ | peak | rms dBFS |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for mode, entry in cont["sweeps"].items():
            for m in ("G5_only", "multi_anchor"):
                s = entry[m]
                print(f"| {mode} | {m} | {s['max_frame_level_jump_db']:.2f} | "
                      f"{s['mean_frame_level_jump_db']:.3f} | {s['frames_over_3db_jump']} | "
                      f"{s['max_sample_delta']:.4f} | {s['samples_over_0p25_delta']} | "
                      f"{s['peak']:.3f} | {s['rms_dbfs']:.2f} |")
        st = cont["static_level_continuity"]
        print(f"\nanchor regions: {st['anchor_regions']}")
        for m in ("G5_only", "multi_anchor"):
            print(f"- {m}: max static level step {st[m]['max_step_db']:.2f} dB, "
                  f"monotonic={st[m]['monotonic_increasing']}, negative steps={st[m]['negative_steps']}")
        dips = [e for e in cont["phase_cancellation_probe"] if e["dips_below_all_contributors"]]
        print(f"- phase-cancellation probe: {len(dips)} of "
              f"{len(cont['phase_cancellation_probe'])} probed positions dip below all contributors"
              f"{': ' + str([e['position'] for e in dips]) if dips else ''}")


if __name__ == "__main__":
    main()
