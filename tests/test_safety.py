import numpy as np

from hybrid.safety import apply_peak_ceiling, check_audio


def test_detects_nan():
    audio = np.array([0.1, np.nan, 0.2])
    report = check_audio(audio)
    assert report.has_nan_or_inf


def test_detects_clipping():
    audio = np.array([0.1, 1.0, -1.0, 0.2])
    report = check_audio(audio)
    assert report.clipped_sample_count == 2


def test_apply_peak_ceiling_reduces_gain_only_when_needed():
    audio = np.array([0.0, 1.0, -1.0])  # peak = 0 dBFS
    scaled, reduction = apply_peak_ceiling(audio, target_peak_dbfs=-3.0)
    assert reduction > 0
    peak = np.max(np.abs(scaled))
    assert peak <= 10 ** (-3.0 / 20) + 1e-6


def test_apply_peak_ceiling_leaves_quiet_audio_unchanged():
    audio = np.array([0.0, 0.1, -0.1])
    scaled, reduction = apply_peak_ceiling(audio, target_peak_dbfs=-3.0)
    assert reduction == 0.0
    assert np.array_equal(scaled, audio)
