"""Regression-check the new finite `bounded_causal_envelope_db` against the
old infinite-memory `rms_envelope_db` on the bundled DI fixtures --
docs/phase3.md section 6.

Usage:
    python scripts/compare_envelopes.py

Reports p10/p50/p90 of each envelope's ACTIVE-signal distribution and the
Amp A / transition / Amp B coverage fractions at a representative crossover
setting, for both the old and new envelope, so a human can confirm the new
envelope is "the same musical idea, bounded causal memory" rather than a
radically different behavior (see hybrid/coverage.py for the coverage
math, reused unmodified here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.blend import CrossoverConfig, blend_weight  # noqa: E402
from hybrid.coverage import active_signal_mask  # noqa: E402
from hybrid.envelope import bounded_causal_envelope_db, rms_envelope_db  # noqa: E402

DI_DIR = Path(__file__).resolve().parent.parent / "assets" / "di"
FILES = ["moderate_brit.wav", "clean_smooth.wav", "high_thrash.wav", "bass_rollin.wav"]

# A representative crossover per docs/phase3.md section 6 -- percentile-based,
# same idea as hybrid.coverage.suggest_crossover_dbfs, computed per-file below.
TRANSITION_WIDTH_DB = 8.0


def _percentiles(active_db: np.ndarray) -> tuple[float, float, float]:
    if len(active_db) == 0:
        return (float("nan"),) * 3
    p10, p50, p90 = np.percentile(active_db, [10, 50, 90])
    return float(p10), float(p50), float(p90)


def _coverage(envelope_db: np.ndarray, crossover_dbfs: float) -> tuple[float, float, float]:
    mask = active_signal_mask(envelope_db)
    active = envelope_db[mask] if mask.any() else envelope_db
    config = CrossoverConfig(crossover_dbfs=crossover_dbfs, transition_width_db=TRANSITION_WIDTH_DB)
    t = blend_weight(active, config)
    a = float(np.mean(t <= 0.10))
    b = float(np.mean(t >= 0.90))
    transition = max(0.0, 1.0 - a - b)
    return a, transition, b


def main() -> None:
    for name in FILES:
        path = DI_DIR / name
        if not path.is_file():
            print(f"{name}: MISSING, skipping")
            continue
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        old_env = rms_envelope_db(audio, sr)
        new_env = bounded_causal_envelope_db(audio, sr)

        old_active = old_env[active_signal_mask(old_env)]
        new_active = new_env[active_signal_mask(new_env)]

        old_p10, old_p50, old_p90 = _percentiles(old_active)
        new_p10, new_p50, new_p90 = _percentiles(new_active)

        crossover = float(np.percentile(old_active, 62.5)) if len(old_active) else -30.0

        old_cov = _coverage(old_env, crossover)
        new_cov = _coverage(new_env, crossover)

        print(f"\n=== {name} (crossover={crossover:.1f} dBFS, transition={TRANSITION_WIDTH_DB} dB) ===")
        print(f"  old p10/p50/p90: {old_p10:6.1f} / {old_p50:6.1f} / {old_p90:6.1f} dBFS")
        print(f"  new p10/p50/p90: {new_p10:6.1f} / {new_p50:6.1f} / {new_p90:6.1f} dBFS")
        print(f"  old coverage A/transition/B: {old_cov[0]:.1%} / {old_cov[1]:.1%} / {old_cov[2]:.1%}")
        print(f"  new coverage A/transition/B: {new_cov[0]:.1%} / {new_cov[1]:.1%} / {new_cov[2]:.1%}")


if __name__ == "__main__":
    main()
