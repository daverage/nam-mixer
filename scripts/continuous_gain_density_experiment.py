"""Runs the Continuous Gain Model Phase 3 capture-density test described in
docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md's "Phase 3 hypothesis": does denser
sampling across a gain-curve knee reduce reconstruction error more than
interpolation-side sophistication would?

Given a directory of real `.nam` captures at fine-grained gain steps (e.g. a
0.5-gain-step sweep), this:

1. Locates the knee (the region of fastest active-RMS change per gain step)
   using a reference DI clip.
2. Compares a SPARSE bracket (spacing 2x the sweep's step size) against a
   DENSE bracket (the sweep's native step size) for reconstructing the
   midpoint gain, via `hybrid.continuous_gain.leave_one_out_validation`.
3. Runs a local three-point test: predict each of the three knee-adjacent
   gains from its own immediate native-step neighbours.

Requires the native nam_render tool to be built (see
native/nam_render/README.md) -- this is real inference.

Usage:
    python scripts/continuous_gain_density_experiment.py <amp_dir> --pattern "jcm800-high-g{gain}-11.4dBu.nam" --step 0.5 --min-gain 1.0 --max-gain 10.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.audio_metrics import rms_dbfs
from hybrid.continuous_gain import GainCapture, GainCaptureSet, leave_one_out_validation, run_ground_truth_harness
from hybrid.coverage import ACTIVE_SIGNAL_THRESHOLD_DBFS, active_signal_mask
from hybrid.envelope import DEFAULT_BOUNDED_ENVELOPE_CONFIG, bounded_causal_envelope_db
from hybrid.nam_loader import load_nam
from hybrid.render import render

_REPO_ROOT = Path(__file__).resolve().parent.parent
DI_FILES = {
    "standard": _REPO_ROOT / "assets/di/moderate_brit.wav",
    "clean": _REPO_ROOT / "assets/di/clean_smooth.wav",
    "metalcore": _REPO_ROOT / "assets/di/high_metalcore.wav",
}


def _gain_path(amp_dir: Path, pattern: str, gain: float) -> Path:
    """Resolve one gain step to a file, trying the exact decimal first and
    then a couple of common alternate-max-gain spellings (e.g. "ga10")."""
    candidate = amp_dir / pattern.format(gain=f"{gain:.1f}")
    if candidate.exists():
        return candidate
    alt = amp_dir / pattern.format(gain=f"a{gain:.0f}")
    if alt.exists():
        return alt
    raise FileNotFoundError(f"No capture for gain={gain} in {amp_dir} (tried {candidate.name}, {alt.name})")


def _load_di(name: str, seconds: int) -> tuple[np.ndarray, int]:
    dry, sr = sf.read(DI_FILES[name])
    return dry.astype(np.float32)[: sr * seconds], sr


def find_knee(gains: list[float], loader, dry: np.ndarray, sample_rate: int) -> float:
    """Return the gain step with the largest active-RMS jump from its
    predecessor -- a simple, deterministic proxy for "where the gain curve
    changes fastest" (the doc's clean-to-breakup knee)."""
    envelope_db = bounded_causal_envelope_db(dry, sample_rate, DEFAULT_BOUNDED_ENVELOPE_CONFIG)
    mask = active_signal_mask(envelope_db, ACTIVE_SIGNAL_THRESHOLD_DBFS)
    levels = []
    for gain in gains:
        output = render(loader(gain), dry, sample_rate)
        n = min(len(output), len(mask))
        levels.append(rms_dbfs(output[:n][mask[:n]]))
    deltas = [abs(levels[i] - levels[i - 1]) for i in range(1, len(levels))]
    knee_index = int(np.argmax(deltas)) + 1  # the gain AFTER the biggest jump
    # Keep room for both a sparse (+/-2 step) and dense (+/-1 step) bracket
    # on each side, so an edge-adjacent knee doesn't collapse the two.
    knee_index = min(max(knee_index, 2), len(gains) - 3)
    return gains[knee_index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("amp_dir", type=Path)
    parser.add_argument("--pattern", required=True, help='Filename pattern with a "{gain}" placeholder, e.g. "jcm800-high-g{gain}-11.4dBu.nam"')
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--min-gain", type=float, default=1.0)
    parser.add_argument("--max-gain", type=float, default=10.0)
    parser.add_argument("--seconds", type=int, default=6)
    args = parser.parse_args()

    n_steps = round((args.max_gain - args.min_gain) / args.step) + 1
    all_gains = [round(args.min_gain + i * args.step, 6) for i in range(n_steps)]

    model_cache: dict[float, object] = {}

    def load_cached(gain: float):
        if gain not in model_cache:
            model_cache[gain] = load_nam(_gain_path(args.amp_dir, args.pattern, gain))
        return model_cache[gain]

    dry_standard, sr = _load_di("standard", args.seconds)
    knee_gain = find_knee(all_gains, load_cached, dry_standard, sr)
    knee_index = all_gains.index(knee_gain)
    print(f"Knee located at gain={knee_gain:g} (step index {knee_index})")

    lower_sparse = all_gains[max(0, knee_index - 2)]
    upper_sparse = all_gains[min(len(all_gains) - 1, knee_index + 2)]
    lower_dense = all_gains[max(0, knee_index - 1)]
    upper_dense = all_gains[min(len(all_gains) - 1, knee_index + 1)]

    def make_set(gains: list[float], hidden_gain: float) -> GainCaptureSet:
        captures = [
            GainCapture(model=load_cached(g), control_position=g, label=f"G{g:g}", hidden=(g == hidden_gain))
            for g in dict.fromkeys(gains)  # de-dupe while preserving order (knee near an edge can clamp two positions together)
        ]
        return GainCaptureSet(captures)

    print(f"\n{'DI':<11}{'bracket':<24}{'predicts':>10}  raw_esr  gnorm_esr  >3kHzΔdB")
    for di_name in DI_FILES:
        dry, sr = _load_di(di_name, args.seconds)

        sparse_set = make_set([lower_sparse, upper_sparse, knee_gain], hidden_gain=knee_gain)
        sparse_harness = run_ground_truth_harness(sparse_set, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)
        sparse_result = leave_one_out_validation(sparse_harness, sparse_set, sr)[0]

        dense_set = make_set([lower_sparse, lower_dense, upper_dense, upper_sparse, knee_gain], hidden_gain=knee_gain)
        dense_harness = run_ground_truth_harness(dense_set, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)
        dense_result = leave_one_out_validation(dense_harness, dense_set, sr)[0]

        print(f"{di_name:<11}{f'G{lower_sparse:g}/G{upper_sparse:g} (sparse)':<24}{knee_gain:>10.1f}  "
              f"{sparse_result.metrics['raw_esr']:.4f}   {sparse_result.metrics['gain_normalized_esr']:.4f}    "
              f"{sparse_result.metrics['high_freq_delta_db']:.2f}")
        print(f"{di_name:<11}{f'G{lower_dense:g}/G{upper_dense:g} (dense)':<24}{knee_gain:>10.1f}  "
              f"{dense_result.metrics['raw_esr']:.4f}   {dense_result.metrics['gain_normalized_esr']:.4f}    "
              f"{dense_result.metrics['high_freq_delta_db']:.2f}")

    print(f"\nLocal three-point test at native step ({args.step:g}) spacing around the knee:")
    # Needs a training point on both sides of whichever of the three middle
    # gains is hidden, so the full 5-point sparse..dense..sparse span is used
    # as context even though only the three interior gains are ever hidden.
    full_local = [lower_sparse, lower_dense, knee_gain, upper_dense, upper_sparse]
    local_targets = [lower_dense, knee_gain, upper_dense]
    dry, sr = _load_di("standard", args.seconds)
    for hidden in local_targets:
        cs = make_set(full_local, hidden_gain=hidden)
        harness = run_ground_truth_harness(cs, dry, sr, calibration_mode="auto", output_normalization_amount=0.5)
        result = leave_one_out_validation(harness, cs, sr)[0]
        print(f"  predict G{hidden:g} from {result.neighbor_labels}: raw_esr={result.metrics['raw_esr']:.4f}  "
              f"gnorm_esr={result.metrics['gain_normalized_esr']:.4f}")


if __name__ == "__main__":
    main()
