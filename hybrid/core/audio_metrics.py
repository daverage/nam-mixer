"""Small, dependency-light audio measurements shared by blend modes."""
from __future__ import annotations

import numpy as np


# rms_dbfs floors digital silence at -200 dBFS instead of -inf, so a level
# difference against a silent region is a meaningless huge number, not a
# trim. Anything this quiet is treated as "no usable level".
SILENCE_DBFS = -120.0


def is_audible_dbfs(level_dbfs: float) -> bool:
    """True if `level_dbfs` is a real signal level, not silence/empty."""
    return bool(np.isfinite(level_dbfs) and level_dbfs > SILENCE_DBFS)


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
    differing much. `hybrid.training.validation.compute_esr_metrics`'s raw ESR is a
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


def _framed_log_magnitude(audio: np.ndarray, window_size: int, hop: int) -> np.ndarray | None:
    """Hann-windowed log-magnitude spectra of consecutive overlapping frames
    -- returns an (n_frames, n_bins) array, or None if `audio` is shorter
    than one frame."""
    n = len(audio)
    n_frames = max(0, (n - window_size) // hop + 1)
    if n_frames <= 0:
        return None
    window = np.hanning(window_size)
    audio = np.asarray(audio, dtype=np.float64)
    frames = np.stack([audio[i * hop:i * hop + window_size] * window for i in range(n_frames)])
    return 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(frames, axis=1)), 1e-10))


def framed_spectral_correlation(a: np.ndarray, b: np.ndarray, window_size: int = 4096) -> float:
    """Like `spectral_magnitude_correlation`, but correlates log-magnitude
    spectra frame-by-frame (50% overlap) rather than one whole-signal FFT --
    more sensitive to LOCAL spectral mismatches that a single long window
    can average away. See docs/history/Continuous Gain/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md's 5150
    metric-discrimination experiment: a useful metric must be able to tell
    an obviously-narrow-bracketed reconstruction from an obviously-wide one
    on the same heavily saturated material, which the whole-signal version
    was not sensitive enough to do reliably by itself.
    """
    n = min(len(a), len(b))
    if n < window_size:
        return float("nan")
    mag_a = _framed_log_magnitude(a[:n], window_size, window_size // 2)
    mag_b = _framed_log_magnitude(b[:n], window_size, window_size // 2)
    if mag_a is None or mag_b is None:
        return float("nan")
    m = min(len(mag_a), len(mag_b))
    flat_a, flat_b = mag_a[:m].ravel(), mag_b[:m].ravel()
    if np.std(flat_a) < 1e-9 or np.std(flat_b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(flat_a, flat_b)[0, 1])


def multi_resolution_log_spectral_distance(a: np.ndarray, b: np.ndarray, window_sizes: tuple[int, ...] = (1024, 4096)) -> float:
    """Mean absolute log-magnitude difference across framed spectra at
    several window sizes (short window for transient/harmonic detail, long
    window for tonal balance), averaged across sizes -- a simple,
    dependency-light stand-in for a full multi-resolution STFT loss. Lower
    is more similar (unlike `framed_spectral_correlation`, which is a
    similarity, not a distance)."""
    n = min(len(a), len(b))
    a64, b64 = np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)
    distances = []
    for window_size in window_sizes:
        mag_a = _framed_log_magnitude(a64, window_size, window_size // 2)
        mag_b = _framed_log_magnitude(b64, window_size, window_size // 2)
        if mag_a is None or mag_b is None:
            continue
        m = min(len(mag_a), len(mag_b))
        distances.append(float(np.mean(np.abs(mag_a[:m] - mag_b[:m]))))
    return float(np.mean(distances)) if distances else float("nan")


def envelope_error_db(a: np.ndarray, b: np.ndarray, sample_rate: int, frame_ms: float = 20.0) -> float:
    """RMS (over frames) of the per-frame dB level difference between two
    signals -- a coarse dynamics/envelope-shape distance, independent of the
    sample-domain phase alignment that `hybrid.training.validation.compute_esr_metrics`
    is sensitive to. Deliberately simple (non-overlapping fixed-size frames,
    no attack/release smoothing) per docs/history/Continuous Gain/CONTINUOUS_GAIN_RESPONSE_COORDINATE.md's
    instruction not to overengineer this."""
    n = min(len(a), len(b))
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = n // frame
    if n_frames == 0:
        return float("nan")
    a_frames = np.asarray(a[:n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    b_frames = np.asarray(b[:n_frames * frame], dtype=np.float64).reshape(n_frames, frame)
    env_a = rms_dbfs_per_frame(a_frames)
    env_b = rms_dbfs_per_frame(b_frames)
    return float(np.sqrt(np.mean((env_a - env_b) ** 2)))


def rms_dbfs_per_frame(frames: np.ndarray) -> np.ndarray:
    """Vectorized per-row `rms_dbfs`, for an (n_frames, frame_length) array."""
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))
