"""Envelope extraction from a dry guitar signal.

The hybrid crossover is driven by the level/envelope of the ORIGINAL DRY INPUT,
never by either processed amp's output -- see the main README for why. This
module is the single place that computes that envelope, so the method (currently
RMS-in-a-sliding-window with attack/release smoothing) can be swapped out later
without touching blend.py.
"""
from __future__ import annotations

import numpy as np

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
