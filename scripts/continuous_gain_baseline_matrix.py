"""Runs the Continuous Gain Model Phase 2 baseline error map described in
docs/CONTINUOUS_GAIN.md ("Critical Validation Strategy" / research
questions Q2/Q3) and docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md.

For a directory of real `.nam` gain-sweep captures (e.g. several fixed-knob
captures of one amp channel at different Gain settings, filenames containing
"G<n>"), this leaves out each interior gain one at a time, reconstructs it
with the Phase 2 baseline (linear output interpolation between its
bracketing captures), and scores the reconstruction against the withheld
real capture across multiple DI excitation types. It also runs the
wide-spacing and level-matched-oracle ablations from the same doc.

Requires the native nam_render tool to be built (native/nam_render/README.md)
-- this is real inference, not the fake-render unit tests in
tests/test_continuous_gain.py.

Usage:
    python scripts/continuous_gain_baseline_matrix.py <amp_dir> [--channels Hi,Lo] [--gains 2,4,6,8,10] [--seconds 6]

`<amp_dir>` must contain files matching "<channel> ... G<gain>.nam" for
every channel/gain combination, e.g. "JCM800 Hi P6 B8 M4 T7 G6.nam".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.audio_metrics import rms_dbfs
from hybrid.continuous_gain import (
    GainCapture,
    GainCaptureSet,
    interpolate_output,
    interpolate_output_level_matched,
    leave_one_out_validation,
    run_ground_truth_harness,
)
from hybrid.nam_loader import load_nam
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
DI_FILES = {
    "standard": _REPO_ROOT / "assets/di/moderate_brit.wav",
    "clean": _REPO_ROOT / "assets/di/clean_smooth.wav",
    "metalcore": _REPO_ROOT / "assets/di/high_metalcore.wav",
}


def _find_capture(amp_dir: Path, channel: str, gain: int, gain_prefix: str) -> Path:
    matches = [p for p in amp_dir.glob("*.nam") if channel in p.stem and f"{gain_prefix}{gain}" == p.stem.split()[-1]]
    if not matches:
        raise FileNotFoundError(f"No capture found for channel={channel!r} gain={gain} in {amp_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(f"Ambiguous capture for channel={channel!r} gain={gain} in {amp_dir}: {matches}")
    return matches[0]


def _load_di(name: str, seconds: int) -> tuple[np.ndarray, int]:
    dry, sr = sf.read(DI_FILES[name])
    dry = dry.astype(np.float32)
    return dry[: sr * seconds], sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("amp_dir", type=Path, help="Directory containing the gain-sweep .nam captures")
    parser.add_argument("--channels", default="Hi,Lo", help="Comma-separated channel name substrings (default: Hi,Lo)")
    parser.add_argument("--gains", default="2,4,6,8,10", help="Comma-separated gain positions (default: 2,4,6,8,10)")
    parser.add_argument("--gain-prefix", default="G", help='Filename label preceding the gain number, e.g. "G" or "V" (default: G)')
    parser.add_argument("--seconds", type=int, default=6, help="Seconds of each DI clip to render (default: 6)")
    args = parser.parse_args()

    channels = args.channels.split(",")
    gains = [int(g) for g in args.gains.split(",")]
    interior_gains = gains[1:-1]  # only interior positions can be leave-one-out validated

    model_cache: dict[tuple[str, int], object] = {}

    def load_cached(channel: str, gain: int):
        key = (channel, gain)
        if key not in model_cache:
            model_cache[key] = load_nam(_find_capture(args.amp_dir, channel, gain, args.gain_prefix))
        return model_cache[key]

    def full_set(channel: str, hidden_gain: int) -> GainCaptureSet:
        captures = [
            GainCapture(
                model=load_cached(channel, g), control_position=float(g),
                label=f"{channel}_{args.gain_prefix}{g}", hidden=(g == hidden_gain),
            )
            for g in gains
        ]
        return GainCaptureSet(captures)

    rows = []
    for channel in channels:
        for hidden_gain in interior_gains:
            for di_name in DI_FILES:
                capture_set = full_set(channel, hidden_gain)
                dry, sr = _load_di(di_name, args.seconds)
                harness = run_ground_truth_harness(
                    capture_set, dry, sr, calibration_mode="auto", output_normalization_amount=0.5,
                )
                result = leave_one_out_validation(harness, capture_set, sr)[0]
                rows.append({
                    "channel": channel, "hidden": hidden_gain, "neighbors": result.neighbor_labels,
                    "di": di_name, "spacing": "narrow", **result.metrics,
                })

    header = f"{'ch':<6}{'hid':>4}{'neighbors':>20}{'di':>11}{'spacing':>12}  raw_esr  gnorm_esr  rms_d   peak_d  lowΔdB  highΔdB"
    print(header)
    for row in rows:
        print(
            f"{row['channel']:<6}{row['hidden']:>4}{str(row['neighbors']):>20}{row['di']:>11}{row['spacing']:>12}  "
            f"{row['raw_esr']:.4f}   {row['gain_normalized_esr']:.4f}    {row['rms_difference']:.4f}  "
            f"{row['peak_difference']:.4f}  {row['low_freq_delta_db']:.2f}   {row['high_freq_delta_db']:.2f}"
        )

    # Wide-spacing ablation: reconstruct the middle interior gain from the
    # two extreme captures instead of its nearest neighbours.
    print("\nWide-spacing ablation (predict middle gain from extreme captures instead of nearest neighbours):")
    mid_gain = gains[len(gains) // 2]
    for channel in channels:
        capture_set = full_set(channel, mid_gain)
        dry, sr = _load_di("standard", args.seconds)
        harness = run_ground_truth_harness(capture_set, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)
        lo_capture = next(c for c in capture_set.training_captures() if c.control_position == gains[0])
        hi_capture = next(c for c in capture_set.training_captures() if c.control_position == gains[-1])
        result = leave_one_out_validation(harness, capture_set, sr, neighbors=(lo_capture, hi_capture))[0]
        print(
            f"  {channel} hidden={mid_gain} neighbors={result.neighbor_labels}: "
            f"raw_esr={result.metrics['raw_esr']:.4f}  gain_norm_esr={result.metrics['gain_normalized_esr']:.4f}"
        )

    # Level-matched oracle ablation on the first channel's middle gain.
    print(f"\nKnob-linear vs level-matched-oracle ablation ({channels[0]}, hidden={mid_gain}, standard DI):")
    capture_set = full_set(channels[0], mid_gain)
    dry, sr = _load_di("standard", args.seconds)
    harness = run_ground_truth_harness(capture_set, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)
    hidden_rendered = harness.by_label(f"{channels[0]}_{args.gain_prefix}{mid_gain}")
    real = hidden_rendered.output
    linear = interpolate_output(harness, capture_set, hidden_rendered.normalized_position)
    oracle = interpolate_output_level_matched(harness, capture_set, hidden_rendered.normalized_position, hidden_rendered.active_rms_dbfs)
    n = min(len(linear), len(real))
    m_linear = compute_esr_metrics(linear[:n], real[:n])
    m_oracle = compute_esr_metrics(oracle[:n], real[:n])
    print(f"  knob-linear blend:      raw_esr={m_linear['raw_esr']:.4f}  rms_diff={m_linear['rms_difference']:.4f}")
    print(f"  level-matched (oracle): raw_esr={m_oracle['raw_esr']:.4f}  rms_diff={m_oracle['rms_difference']:.4f}")
    print(f"  reconstructed active rms (linear) = {rms_dbfs(linear):.2f} dB   target (real) = {hidden_rendered.active_rms_dbfs:.2f} dB")


if __name__ == "__main__":
    main()
