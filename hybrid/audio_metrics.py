"""Small, dependency-light audio measurements shared by blend modes."""
from __future__ import annotations

import numpy as np


def rms_dbfs(audio: np.ndarray, floor_dbfs: float | None = None) -> float:
    """Return RMS level in dBFS, with an optional lower reporting floor."""
    if len(audio) == 0:
        return -np.inf if floor_dbfs is None else floor_dbfs
    rms = np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))
    level = 20.0 * np.log10(max(rms, 1e-10))
    return max(level, floor_dbfs) if floor_dbfs is not None else level


def band_energy_dbfs(audio: np.ndarray, sample_rate: int, low_hz: float = 0.0, high_hz: float | None = None) -> float:
    """RMS level (dBFS) of `audio` restricted to [low_hz, high_hz) via a hard
    FFT-domain band mask -- a coarse, deterministic band-energy measurement
    (not a proper filter design), good enough to compare how much of a
    difference between two renders is concentrated in a given frequency
    band without pulling in a separate spectral-analysis dependency."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) == 0:
        return -np.inf
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    mask = freqs >= low_hz
    if high_hz is not None:
        mask &= freqs < high_hz
    banded = np.zeros_like(spectrum)
    banded[mask] = spectrum[mask]
    return rms_dbfs(np.fft.irfft(banded, n=len(audio)))
