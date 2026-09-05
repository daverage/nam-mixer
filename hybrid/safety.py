"""Safety checks and gain-staging for generated training targets vs. live preview.

Important distinction (see main README):
- The GENERATED TRAINING TARGET must never be limited. If it's too hot, we apply
  a single fixed gain reduction to the whole file instead, so the dynamic
  behavior we're trying to train into the A2 model isn't altered by a limiter.
- A limiter is only acceptable on the LIVE PREVIEW/PLAYBACK path, purely as a
  speaker/headphone safety net, and must never touch the file that gets used to
  build the training target.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TARGET_PEAK_DBFS = -3.0


@dataclass
class SafetyReport:
    has_nan_or_inf: bool
    peak_dbfs: float
    clipped_sample_count: int
    max_abs_sample_jump: float
    is_silent: bool
    gain_reduction_applied_db: float


def check_audio(audio: np.ndarray, clip_threshold: float = 0.999) -> SafetyReport:
    """Basic sanity checks used both by the app and by tests/regression scripts."""
    audio = np.asarray(audio, dtype=np.float64)
    has_nan_or_inf = bool(np.isnan(audio).any() or np.isinf(audio).any())

    finite = audio[np.isfinite(audio)]
    if len(finite) == 0:
        peak = -np.inf
    else:
        peak_amp = np.max(np.abs(finite))
        peak = 20.0 * np.log10(peak_amp) if peak_amp > 0 else -np.inf

    clipped = int(np.sum(np.abs(finite) >= clip_threshold)) if len(finite) else 0

    if len(finite) > 1:
        jumps = np.abs(np.diff(finite))
        max_jump = float(np.max(jumps))
    else:
        max_jump = 0.0

    is_silent = bool(len(finite) == 0 or np.max(np.abs(finite)) < 1e-6)

    return SafetyReport(
        has_nan_or_inf=has_nan_or_inf,
        peak_dbfs=float(peak),
        clipped_sample_count=clipped,
        max_abs_sample_jump=max_jump,
        is_silent=is_silent,
        gain_reduction_applied_db=0.0,
    )


def apply_peak_ceiling(audio: np.ndarray, target_peak_dbfs: float = DEFAULT_TARGET_PEAK_DBFS) -> tuple[np.ndarray, float]:
    """If `audio` exceeds target_peak_dbfs, apply a single fixed gain reduction
    to bring its peak down to that ceiling. NEVER a limiter -- see module
    docstring. If already under the ceiling, audio is returned unchanged (no
    makeup gain is applied; we only ever reduce, matching the brief).

    Returns (possibly-scaled audio, gain_reduction_db applied -- 0.0 if none).
    """
    audio = np.asarray(audio, dtype=np.float64)
    finite = audio[np.isfinite(audio)]
    if len(finite) == 0:
        return audio, 0.0

    peak_amp = np.max(np.abs(finite))
    if peak_amp <= 0:
        return audio, 0.0

    peak_dbfs = 20.0 * np.log10(peak_amp)
    if peak_dbfs <= target_peak_dbfs:
        return audio, 0.0

    reduction_db = peak_dbfs - target_peak_dbfs
    gain = 10.0 ** (-reduction_db / 20.0)
    return audio * gain, reduction_db


def preview_safety_limiter(audio: np.ndarray, ceiling_dbfs: float = -1.0) -> np.ndarray:
    """Simple hard-clip safety limiter for LIVE PLAYBACK ONLY.

    Must never be applied to a file that will be saved as (or used to derive) a
    training target -- use apply_peak_ceiling for that instead. This exists
    purely so a runaway gain-staging bug in the preview path can't blast the
    user's speakers/headphones.
    """
    ceiling_amp = 10.0 ** (ceiling_dbfs / 20.0)
    return np.clip(audio, -ceiling_amp, ceiling_amp)
