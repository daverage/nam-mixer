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


def spectral_magnitude_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two signals' log-magnitude spectra (a single
    whole-signal Hann-windowed FFT, not framed/averaged) -- a coarse
    "same spectral shape" measurement that stays meaningful even when the
    signals' SAMPLE-DOMAIN correlation collapses despite genuinely similar
    tonal/harmonic character.

    This matters specifically for heavily-driven/high-gain amp material:
    two renders that a listener would call nearly identical can have
    near-zero raw waveform correlation, because heavy nonlinear
    distortion/compression is hypersensitive to microscopic input
    differences at individual clipping instants -- a tiny gain difference
    shifts WHERE a sample clips, decorrelating the waveforms sample-for-
    sample, without the underlying harmonic/spectral content actually
    differing much. `hybrid.validation.compute_esr_metrics`'s raw ESR is a
    sample-domain metric and can therefore look catastrophic on such
    material even when this spectral correlation stays high -- use both
    together rather than raw ESR alone on heavily saturated captures.
    """
    n = min(len(a), len(b))
    if n < 8:
        return float("nan")
    window = np.hanning(n)
    log_mag_a = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(np.asarray(a[:n], dtype=np.float64) * window)), 1e-10))
    log_mag_b = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(np.asarray(b[:n], dtype=np.float64) * window)), 1e-10))
    if np.std(log_mag_a) < 1e-9 or np.std(log_mag_b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(log_mag_a, log_mag_b)[0, 1])


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
