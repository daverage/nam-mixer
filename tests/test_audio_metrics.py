import numpy as np
import pytest

from hybrid.core.audio_metrics import (
    envelope_error_db,
    framed_spectral_correlation,
    multi_resolution_log_spectral_distance,
    rms_dbfs,
)


def test_rms_dbfs_matches_existing_blend_semantics():
    assert np.isneginf(rms_dbfs(np.array([])))
    assert np.isclose(rms_dbfs(np.array([0.1, -0.1])), -20.0)
    assert rms_dbfs(np.array([0.0]), floor_dbfs=-90.0) == -90.0


def _tone(freq, n=8192, sr=48000, amplitude=0.5):
    t = np.arange(n) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_framed_spectral_correlation_is_high_for_identical_signal():
    x = _tone(440)
    assert framed_spectral_correlation(x, x) == pytest.approx(1.0, abs=1e-6)


def test_framed_spectral_correlation_is_low_for_very_different_tones():
    a = _tone(220)
    b = _tone(8000)
    assert framed_spectral_correlation(a, b) < 0.5


def test_framed_spectral_correlation_nan_for_short_signal():
    assert np.isnan(framed_spectral_correlation(np.zeros(100), np.zeros(100), window_size=4096))


def test_multi_resolution_log_spectral_distance_is_zero_for_identical_signal():
    x = _tone(440)
    assert multi_resolution_log_spectral_distance(x, x) == pytest.approx(0.0, abs=1e-6)


def test_multi_resolution_log_spectral_distance_is_larger_for_different_tones():
    a = _tone(220)
    b = _tone(8000)
    same = multi_resolution_log_spectral_distance(a, a)
    different = multi_resolution_log_spectral_distance(a, b)
    assert different > same


def test_envelope_error_db_is_zero_for_identical_signal():
    x = _tone(440)
    assert envelope_error_db(x, x, 48000) == pytest.approx(0.0, abs=1e-6)


def test_envelope_error_db_detects_a_level_difference():
    x = _tone(440)
    louder = x * 2.0
    assert envelope_error_db(x, louder, 48000) == pytest.approx(20.0 * np.log10(2.0), abs=1e-3)
