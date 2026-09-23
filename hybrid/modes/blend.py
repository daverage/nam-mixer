"""Dynamic crossfade between Amp A and Amp B, driven by the dry input's envelope.

The transition function (currently smoothstep) is intentionally isolated behind
`TransitionCurve` so other curves (linear, exponential, custom) can be tried
later without touching the blending math itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Suggested transition-width presets (dB), per the project brief.
TRANSITION_WIDTH_PRESETS_DB = {
    "very_hard": 2.0,
    "hard": 4.0,
    "medium": 8.0,
    "soft": 12.0,
    "very_soft": 18.0,
}
DEFAULT_TRANSITION_WIDTH_DB = 8.0


def smoothstep_curve(t: np.ndarray) -> np.ndarray:
    """Classic smoothstep: 3t^2 - 2t^3, t already clamped to [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


TransitionCurve = Callable[[np.ndarray], np.ndarray]


@dataclass
class CrossoverConfig:
    crossover_dbfs: float
    transition_width_db: float = DEFAULT_TRANSITION_WIDTH_DB
    curve: TransitionCurve = smoothstep_curve
    amp_a_trim_db: float = 0.0
    amp_b_trim_db: float = 0.0


def blend_weight(envelope_db: np.ndarray, config: CrossoverConfig) -> np.ndarray:
    """Compute the Amp B mix weight `t` in [0, 1] for each sample of envelope_db.

    t=0 -> pure Amp A (quiet input), t=1 -> pure Amp B (loud input).
    """
    width = max(config.transition_width_db, 1e-6)
    start = config.crossover_dbfs - width / 2.0
    t_linear = (envelope_db - start) / width
    t_linear = np.clip(t_linear, 0.0, 1.0)
    return config.curve(t_linear)


def blend(
    dry_envelope_db: np.ndarray,
    amp_a_render: np.ndarray,
    amp_b_render: np.ndarray,
    config: CrossoverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend two (already aligned, already trimmed-if-desired) amp renders.

    Uses a LINEAR amplitude crossfade (not equal-power) because amp_a_render and
    amp_b_render are correlated versions of the same guitar signal, not
    independent sources -- equal-power crossfades assume decorrelated content
    and would produce a level bump in the transition region here.

    Returns (hybrid_audio, t_curve) where t_curve is the per-sample Amp B weight
    actually used, useful for debugging/plotting the transition.
    """
    n = min(len(dry_envelope_db), len(amp_a_render), len(amp_b_render))
    envelope_db = dry_envelope_db[:n]
    a = amp_a_render[:n] * (10.0 ** (config.amp_a_trim_db / 20.0))
    b = amp_b_render[:n] * (10.0 ** (config.amp_b_trim_db / 20.0))

    t = blend_weight(envelope_db, config)
    hybrid = a * (1.0 - t) + b * t
    return hybrid, t
