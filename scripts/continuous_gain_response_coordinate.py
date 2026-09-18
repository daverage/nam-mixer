"""Runs the Continuous Gain Model response-coordinate experiment described
in docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md, Parts 1-4 and 6, against a
single real gain/volume-sweep dataset.

For each interior (leave-one-out-able) capture, reports:
  - Part 1: adjacent-step measurements across the whole sweep
  - Part 2: RMS-only vs level+spectral cumulative response axis
  - Part 3 Test A: oracle response-fraction mismatch vs knob-linear, and its
    correlation with knob-linear raw ESR (diagnostic only)
  - Part 3 Test B: non-oracle response-coordinate interpolation vs
    knob-linear interpolation, leave-one-out
  - Part 4/6: whether the response-aware capture-placement score picks a
    different (and more informative) step than RMS-only

Requires the native nam_render tool (real inference).

Usage:
    python scripts/continuous_gain_response_coordinate.py <amp_dir> --pattern "JCM800 Hi P6 B8 M4 T7 G{gain}.nam" --gains 2,4,6,8,10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.continuous_gain import (
    GainCapture,
    GainCaptureSet,
    interpolate_output,
    leave_one_out_validation,
    run_ground_truth_harness,
)
from hybrid.nam_loader import load_nam
from hybrid.response_coordinate import (
    RMS_ONLY_WEIGHTS,
    build_response_axis,
    combined_step_scores,
    compute_adjacent_metrics,
    interpolate_output_response_coordinate,
    oracle_response_mismatch,
)
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
DI_PATH = _REPO_ROOT / "assets/di/moderate_brit.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("amp_dir", type=Path)
    parser.add_argument("--pattern", required=True, help='Filename pattern with a "{gain}" placeholder')
    parser.add_argument("--gains", required=True, help="Comma-separated knob positions, e.g. 2,4,6,8,10")
    parser.add_argument("--seconds", type=int, default=6)
    args = parser.parse_args()

    gains = [float(g) for g in args.gains.split(",")]

    def load_cached(gain: float):
        candidates = [f"{gain:g}", f"{gain:.1f}", f"a{gain:.0f}", f"a{gain:.1f}"]
        for candidate in candidates:
            path = args.amp_dir / args.pattern.format(gain=candidate)
            if path.exists():
                return load_nam(path)
        raise FileNotFoundError(f"No capture for gain={gain} in {args.amp_dir} (tried {candidates})")

    dry, sr = sf.read(DI_PATH)
    dry = dry.astype(np.float32)[: sr * args.seconds]

    full_captures = [GainCapture(model=load_cached(g), control_position=g, label=f"G{g:g}") for g in gains]
    full_set = GainCaptureSet(full_captures)
    full_harness = run_ground_truth_harness(full_set, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)

    print("=== Part 1: adjacent-step measurements (full sweep, standard DI) ===")
    steps = compute_adjacent_metrics(full_harness, full_set)
    print(f"{'step':<14}{'level_Δ_db':>12}{'peak_Δ_db':>12}{'spectral_dist':>15}")
    for s in steps:
        print(f"{s.lower_label}->{s.upper_label:<8}{s.level_change_db:>12.3f}{s.peak_change_db:>12.3f}{s.spectral_distance:>15.4f}")

    print("\n=== Part 2: response axis (knob position -> cumulative coordinate) ===")
    axis_rms = build_response_axis(steps, first_position=gains[0], weights=RMS_ONLY_WEIGHTS)
    axis_combined = build_response_axis(steps, first_position=gains[0])
    print(f"{'knob':>8}{'rms_only_r':>14}{'level+spectral_r':>20}")
    for pos, r_rms, r_comb in zip(axis_rms.positions, axis_rms.coordinates, axis_combined.coordinates):
        print(f"{pos:>8.2f}{r_rms:>14.4f}{r_comb:>20.4f}")

    print("\n=== Part 4/6: capture-placement score (which step looks most in need of another capture) ===")
    score_rms = combined_step_scores(steps, weights=RMS_ONLY_WEIGHTS)
    score_combined = combined_step_scores(steps)
    print(f"{'step':<14}{'score_rms':>12}{'score_combined':>16}")
    for s, r, c in zip(steps, score_rms, score_combined):
        print(f"{s.lower_label}->{s.upper_label:<8}{r:>12.4f}{c:>16.4f}")
    print(f"RMS-only top step:        {steps[int(np.argmax(score_rms))].lower_label}->{steps[int(np.argmax(score_rms))].upper_label}")
    print(f"level+spectral top step:  {steps[int(np.argmax(score_combined))].lower_label}->{steps[int(np.argmax(score_combined))].upper_label}")

    print("\n=== Part 3: leave-one-out (interior positions only) ===")
    print(f"{'hidden':>8}{'neighbors':>18}  {'knob_esr':>10}{'response_esr':>14}{'oracle_mismatch':>17}")
    mismatches, knob_esrs = [], []
    for i in range(1, len(gains) - 1):
        hidden_gain = gains[i]
        captures = [
            GainCapture(model=load_cached(g), control_position=g, label=f"G{g:g}", hidden=(g == hidden_gain))
            for g in gains
        ]
        cs = GainCaptureSet(captures)
        harness = run_ground_truth_harness(cs, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)
        training_steps = compute_adjacent_metrics(harness, cs)
        axis = build_response_axis(training_steps, first_position=gains[0])

        hidden_capture = cs.hidden_captures()[0]
        pos = cs.normalized_position(hidden_capture)
        lower, upper = cs.neighbors(pos)

        knob_recon = interpolate_output(harness, cs, pos)
        response_recon = interpolate_output_response_coordinate(harness, cs, pos, axis)
        real = harness.by_label(hidden_capture.label).output
        n = min(len(knob_recon), len(response_recon), len(real))
        knob_esr = compute_esr_metrics(knob_recon[:n], real[:n])["raw_esr"]
        response_esr = compute_esr_metrics(response_recon[:n], real[:n])["raw_esr"]

        oracle = oracle_response_mismatch(harness, lower, hidden_capture, upper, training_steps)
        mismatches.append(oracle.mismatch)
        knob_esrs.append(knob_esr)

        print(f"{hidden_gain:>8.2f}{f'{lower.label}/{upper.label}':>18}  {knob_esr:>10.4f}{response_esr:>14.4f}{oracle.mismatch:>17.4f}")

    if len(mismatches) >= 3:
        corr = np.corrcoef(mismatches, knob_esrs)[0, 1]
        print(f"\nOracle mismatch vs knob-linear raw ESR correlation: {corr:.3f}  (n={len(mismatches)})")
    else:
        print("\nToo few interior points for a meaningful mismatch/ESR correlation.")


if __name__ == "__main__":
    main()
