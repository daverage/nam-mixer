"""Offline Continuous Gain PRODUCTION profile builder / regression benchmark.

Builds a `hybrid.continuous_gain_profile.ContinuousGainProfile` from a real
capture sweep and validates it against withheld half-step captures, using
the SAME dense "Marshall JCM800 2203 - updated" dataset as
`scripts/continuous_gain_virtual_gain_benchmark.py` -- see
docs/CONTINUOUS_GAIN_PRODUCTION.md for what this reproduces and why.

Usage:
    python scripts/continuous_gain_profile_builder.py [--out work/continuous_gain/jcm800_profile.json]

Requires the native `nam_render` tool built (see native/nam_render/README.md)
and the real capture set at AMP_DIR below -- not part of the automated test
suite (tests/test_continuous_gain_profile.py covers the module with a fake
render()), this is the real-data regression benchmark the doc requirements
ask for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.continuous_gain_profile import (
    ContinuousGainRuntime,
    TrainingTarget,
    build_profile,
    render_with_input_gain,
)
from hybrid.nam_loader import load_nam
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
AMP_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Marshall JCM800 2203 - updated")
EVAL_DI_FILE = _REPO_ROOT / "assets/di/moderate_brit.wav"
EVAL_SECONDS = 6.0

TRAINING_GAINS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
HELD_OUT_GAINS = [round(1.0 + 0.5 * i, 1) for i in range(19) if (1.0 + 0.5 * i) % 1 != 0]


def gain_path(gain: float) -> Path:
    if gain == 10.0:
        return AMP_DIR / "jcm800-high-ga10-11.4dBu.nam"
    return AMP_DIR / f"jcm800-high-g{gain:.1f}-11.4dBu.nam"


def load_dry(sample_rate_hint: int = 48000) -> tuple[np.ndarray, int]:
    dry, sample_rate = sf.read(EVAL_DI_FILE, dtype="float32", always_2d=False)
    if dry.ndim > 1:
        dry = dry.mean(axis=1)
    n = int(EVAL_SECONDS * sample_rate)
    return dry[:n].astype(np.float32), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "work/continuous_gain/jcm800_profile.json")
    parser.add_argument("--max-anchors", type=int, default=3)
    args = parser.parse_args()

    if not AMP_DIR.exists():
        print(f"SKIP: capture directory not found: {AMP_DIR}")
        return

    dry, sample_rate = load_dry()
    nam_paths = {}
    targets = []
    for gain in TRAINING_GAINS:
        path = gain_path(gain)
        model = load_nam(path)
        label = f"g{gain:g}"
        nam_paths[label] = str(path)
        output = render_with_input_gain(model, dry, sample_rate, 0.0)
        targets.append(TrainingTarget(label=label, physical_position=gain, model=model, output=output))

    print(f"Building profile from {len(targets)} training captures (max_anchors={args.max_anchors})...")
    profile = build_profile(targets, dry, sample_rate, max_anchors=args.max_anchors, nam_paths=nam_paths)

    print(f"Selected {len(profile.anchors)} anchor(s): {[a.label for a in profile.anchors]}")
    print(f"Build-time quality: {profile.validation.quality} "
          f"(worst raw ESR {profile.validation.worst_raw_esr:.4f})")
    for region in profile.anchors:
        print(f"  {region.label}: range [{region.range_low:g}, {region.range_high:g}], "
              f"worst ESR in region {region.worst_raw_esr_in_region:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile.to_dict(), indent=2))
    print(f"Wrote profile to {args.out}")

    # Regression benchmark: validate against genuinely-real withheld
    # half-step captures, reproducing the intent of
    # scripts/continuous_gain_virtual_gain_benchmark.py's experiment C.
    models_by_label = {a.label: load_nam(Path(a.nam_path)) for a in profile.anchors}
    runtime = ContinuousGainRuntime(profile, models_by_label)

    print("\nRegression benchmark against held-out half-step captures:")
    esrs = []
    for gain in HELD_OUT_GAINS:
        path = gain_path(gain)
        if not path.exists():
            continue
        real_model = load_nam(path)
        real_output = render_with_input_gain(real_model, dry, sample_rate, 0.0)
        reconstructed = runtime.render_at(gain, dry, sample_rate)
        n = min(len(reconstructed), len(real_output))
        metrics = compute_esr_metrics(reconstructed[:n], real_output[:n])
        esrs.append(metrics["raw_esr"])
        print(f"  Gain {gain:g}: raw ESR {metrics['raw_esr']:.4f}")

    if esrs:
        print(f"\nMean held-out raw ESR: {np.mean(esrs):.4f}, worst: {np.max(esrs):.4f}")


if __name__ == "__main__":
    main()
