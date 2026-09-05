"""Envelope extraction from a dry guitar signal.

The hybrid crossover is driven by the level/envelope of the ORIGINAL DRY INPUT,
never by either processed amp's output -- see the main README for why. This
module is the single place that computes that envelope, so the method can be
swapped out later without touching blend.py.

Two implementations live here:

- `rms_envelope_db` -- the ORIGINAL envelope (causal RMS + one-pole
  attack/release). DEPRECATED for production use: its one-pole release has
  theoretically infinite memory, which is unacceptable for a signal that
  becomes the crossover control baked into a synthetic training target for a
  causal, finite-receptive-field A2 model (see docs/phase3.md and
  scripts/compare_envelopes.py). Kept only for the old regression tests and
  the old-vs-new comparison script.
- `bounded_causal_envelope_db` -- the PRODUCTION envelope used by
  `hybrid.pipeline.render_pair`. Same causal-RMS-then-smooth idea, but every
  stage is a finite window (FIR), so the total dry-input dependency of the
  output is a small, fixed, documented number of samples -- see
  `BoundedEnvelopeConfig.max_history_ms`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

_EPS = 1e-10


def rms_envelope_db(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: float = 20.0,
    attack_ms: float = 5.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    """Return a per-sample envelope of `audio` in dBFS, same length as `audio`.

    Method: a CAUSAL moving-RMS estimate (only the current and preceding
    `window_ms` of samples, never future ones) followed by asymmetric
    attack/release smoothing (fast attack, slower release) so the envelope
    tracks picking transients without chattering during decay. Causality
    matters here specifically because this envelope becomes the crossover
    control signal baked into the synthetic hybrid training target -- a
    non-causal (centered) window would let the target start moving toward
    Amp B before the louder input that justifies it has actually arrived,
    which a causal A2 model trained on that target could never reproduce.
    This is a reasonable default, not a claimed-optimal one -- replace this
    function's body to experiment with other envelope followers (peak-based,
    different window sizes, etc.) without changing callers, but keep any
    replacement causal for the same reason.
    """
    audio = np.asarray(audio, dtype=np.float64)
    n = len(audio)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    win = max(1, int(sample_rate * window_ms / 1000.0))
    squared = audio * audio
    # Causal windowed mean-of-squares via a zero-padded cumulative sum: sample i's
    # window covers only samples [i - win + 1, i] (missing history at the start
    # is treated as silence), never samples after i.
    padded = np.concatenate([np.zeros(win - 1, dtype=np.float64), squared])
    cumsum = np.concatenate([[0.0], np.cumsum(padded)])
    window_sum = cumsum[win:] - cumsum[:n]
    mean_sq = window_sum / win
    rms = np.sqrt(np.maximum(mean_sq, 0.0))

    attack_coef = _time_to_coef(attack_ms, sample_rate)
    release_coef = _time_to_coef(release_ms, sample_rate)

    smoothed = np.empty_like(rms)
    state = rms[0]
    for i in range(n):
        target = rms[i]
        coef = attack_coef if target > state else release_coef
        state = state + coef * (target - state)
        smoothed[i] = state

    db = 20.0 * np.log10(np.maximum(smoothed, _EPS))
    return db


def _time_to_coef(time_ms: float, sample_rate: int) -> float:
    """One-pole smoothing coefficient for a given time constant."""
    if time_ms <= 0:
        return 1.0
    tau_samples = (time_ms / 1000.0) * sample_rate
    return 1.0 - np.exp(-1.0 / max(tau_samples, 1e-6))


def _ms_to_samples(time_ms: float, sample_rate: int) -> int:
    return max(1, int(round(time_ms / 1000.0 * sample_rate)))


def _causal_moving_mean(db: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average of a dB-domain signal, edge-padded (repeats the
    first value backward) rather than zero-padded -- zero-padding a dB value
    would inject an artificial -inf/very-negative spike at the start, which a
    linear-amplitude RMS window doesn't have (silence really is 0 amplitude,
    but it is emphatically not the dB floor of whatever this signal already
    is at sample 0)."""
    n = len(db)
    if n == 0 or window <= 1:
        return db.copy()
    padded = np.concatenate([np.full(window - 1, db[0], dtype=np.float64), db])
    cumsum = np.concatenate([[0.0], np.cumsum(padded)])
    window_sum = cumsum[window:] - cumsum[:n]
    return window_sum / window


def _causal_sliding_max(values: np.ndarray, window: int, floor: float) -> np.ndarray:
    """max(values[i - window + 1 : i + 1]) for every i, treating samples
    before index 0 as `floor` -- a genuinely FIR (bounded-window) operation,
    O(n) via scipy's C implementation, not a recursive/IIR one."""
    n = len(values)
    if n == 0 or window <= 1:
        return values.copy()
    origin = (window - 1) // 2
    return maximum_filter1d(values, size=window, mode="constant", cval=floor, origin=origin)


@dataclass(frozen=True)
class BoundedEnvelopeConfig:
    """Window sizes for `bounded_causal_envelope_db`, in milliseconds.

    Total worst-case dry-input dependency of the output ("how many samples
    older than the current one can still change env_db[i]?") is
    `rms_window_ms + attack_avg_ms + release_window_ms` (each stage's finite
    window adds directly to the one before it, since it consumes the
    previous stage's already-bounded output) -- see
    `bounded_envelope_max_history_ms`/`bounded_envelope_max_history_samples`
    below, and tests/test_envelope_finite.py for the proof.
    """

    rms_window_ms: float = 20.0
    attack_avg_ms: float = 5.0
    release_window_ms: float = 55.0
    release_range_db: float = 60.0


DEFAULT_BOUNDED_ENVELOPE_CONFIG = BoundedEnvelopeConfig()

# The target ceiling from docs/phase3.md ("<= about 100 ms at 48 kHz"). The
# default config above sums to 80 ms, leaving 20 ms of margin against the A2
# receptive field check in hybrid/receptive_field.py.
MAX_HISTORY_MS_TARGET = 100.0


def bounded_envelope_max_history_ms(config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG) -> float:
    return config.rms_window_ms + config.attack_avg_ms + config.release_window_ms


def bounded_envelope_max_history_samples(
    sample_rate: int, config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG
) -> int:
    """Exact worst-case sample count: env_db[i] can depend on dry-input
    samples as old as `i - bounded_envelope_max_history_samples(...)`, and
    never anything older, and never anything from `i+1` onward."""
    w1 = _ms_to_samples(config.rms_window_ms, sample_rate)
    w2 = _ms_to_samples(config.attack_avg_ms, sample_rate)
    w3 = _ms_to_samples(config.release_window_ms, sample_rate)
    return (w1 - 1) + (w2 - 1) + (w3 - 1)


def bounded_causal_envelope_db(
    audio: np.ndarray,
    sample_rate: int,
    config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG,
) -> np.ndarray:
    """The PRODUCTION crossover envelope: strictly causal AND strictly
    bounded-memory (see module docstring for why `rms_envelope_db`'s one-pole
    release is not acceptable for this).

    Three cascaded FIR (non-recursive) stages, each a fixed causal window:

      1. Causal moving RMS over `rms_window_ms` (dB), exactly like
         `rms_envelope_db`'s first stage.
      2. Causal moving average over `attack_avg_ms` of stage 1's dB values --
         a bounded stand-in for the old one-pole attack smoothing, replacing
         "abrupt" transient jumps with a short, finite-length blur instead of
         instant response.
      3. Causal decaying-maximum ("peak hold with linear dB decay") over
         `release_window_ms`: sample i is
         `max_{k in [i-W+1, i]} (stage2_db[k] - release_range_db * (i-k) / W)`.
         This is the bounded replacement for a one-pole release: a loud peak's
         influence provably reaches exactly zero `release_window_ms` later,
         not asymptotically. Implemented as a shifted causal sliding-max (see
         `_causal_sliding_max`), which is the standard O(n) technique for a
         decaying peak-hold detector.

    Deterministic, causal (no sample depends on any `audio[j]` for `j > i`),
    and its total dependency on `audio` is bounded by
    `bounded_envelope_max_history_samples(sample_rate, config)` -- see
    tests/test_envelope_finite.py.
    """
    audio = np.asarray(audio, dtype=np.float64)
    n = len(audio)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    w1 = _ms_to_samples(config.rms_window_ms, sample_rate)
    squared = audio * audio
    padded = np.concatenate([np.zeros(w1 - 1, dtype=np.float64), squared])
    cumsum = np.concatenate([[0.0], np.cumsum(padded)])
    window_sum = cumsum[w1:] - cumsum[:n]
    rms = np.sqrt(np.maximum(window_sum / w1, 0.0))
    stage1_db = 20.0 * np.log10(np.maximum(rms, _EPS))

    w2 = _ms_to_samples(config.attack_avg_ms, sample_rate)
    stage2_db = _causal_moving_mean(stage1_db, w2)

    w3 = _ms_to_samples(config.release_window_ms, sample_rate)
    if w3 <= 1:
        return stage2_db

    decay_per_sample = config.release_range_db / w3
    idx = np.arange(n, dtype=np.float64)
    ramped = stage2_db + decay_per_sample * idx
    floor_db = float(stage2_db.min()) - config.release_range_db - 1.0
    floor_ramped = floor_db  # sentinel is below any real ramped value at idx=0
    windowed_max = _causal_sliding_max(ramped, w3, floor=floor_ramped)
    stage3_db = windowed_max - decay_per_sample * idx
    return stage3_db
