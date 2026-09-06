"""Tests for hybrid/cab_ir.py -- see docs/blend-mode.md "CAB IR PROCESSING"."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from hybrid.cab_ir import (
    CabIrError,
    apply_cab_ir,
    cab_design_from_prepared,
    get_prepared_cab_ir,
    load_and_prepare_cab_ir,
)


def _write_wav(path, data, sample_rate=48000):
    sf.write(path, np.asarray(data, dtype=np.float32), sample_rate, subtype="FLOAT")
    return path


def test_identity_ir_leaves_audio_unchanged(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", [1.0])  # single-tap identity IR
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    audio = np.array([0.1, -0.2, 0.3, 0.05], dtype=np.float32)
    out = apply_cab_ir(audio, prepared)
    assert len(out) == len(audio)
    np.testing.assert_allclose(out, audio, atol=1e-6)


def test_known_short_fir_matches_expected_causal_convolution(tmp_path):
    ir = [0.5, 0.5]  # simple 2-tap averaging-delay FIR
    ir_path = _write_wav(tmp_path / "ir.wav", ir)
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    audio = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = apply_cab_ir(audio, prepared)
    expected = np.convolve(audio, ir)[: len(audio)]
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_output_length_equals_input_length(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(0).uniform(-1, 1, 500))
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    for n in (1, 10, 4096, 50000):
        audio = np.zeros(n, dtype=np.float32)
        out = apply_cab_ir(audio, prepared)
        assert len(out) == n


def test_stereo_ir_downmix_is_deterministic(tmp_path):
    n = 200
    left = np.random.default_rng(1).uniform(-1, 1, n).astype(np.float32)
    right = np.random.default_rng(2).uniform(-1, 1, n).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    ir_path = tmp_path / "stereo_ir.wav"
    sf.write(ir_path, stereo, 48000, subtype="FLOAT")

    prepared_1 = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    prepared_2 = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    np.testing.assert_array_equal(prepared_1.samples, prepared_2.samples)
    # Deterministic channel-average downmix, not "pick channel 0".
    expected_mono = (left + right) / 2.0
    # (leading-silence trimming may shift the start; compare via full match
    # after re-padding with the trimmed prefix)
    reconstructed = np.concatenate([np.zeros(prepared_1.leading_samples_trimmed, dtype=np.float32), prepared_1.samples])
    np.testing.assert_allclose(reconstructed[: len(expected_mono)], expected_mono, atol=1e-6)


def test_sample_rate_conversion_changes_tap_count(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(3).uniform(-1, 1, 480), sample_rate=48000)
    prepared_same = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    prepared_half = load_and_prepare_cab_ir(ir_path, target_sample_rate=24000)
    assert prepared_half.sample_rate == 24000
    # Resampled to half the rate -> roughly half as many taps.
    assert prepared_half.prepared_frame_count < prepared_same.prepared_frame_count


def test_empty_ir_is_rejected(tmp_path):
    ir_path = tmp_path / "empty.wav"
    sf.write(ir_path, np.zeros((0,), dtype=np.float32), 48000, subtype="FLOAT")
    with pytest.raises(CabIrError):
        load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)


def test_silent_ir_is_rejected(tmp_path):
    ir_path = _write_wav(tmp_path / "silent.wav", np.zeros(1000))
    with pytest.raises(CabIrError):
        load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)


def test_non_finite_ir_is_rejected(tmp_path):
    ir_path = tmp_path / "nan.wav"
    data = np.array([1.0, float("nan"), 0.5], dtype=np.float32)
    sf.write(ir_path, data, 48000, subtype="FLOAT")
    with pytest.raises(CabIrError):
        load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(CabIrError):
        load_and_prepare_cab_ir(tmp_path / "nope.wav", target_sample_rate=48000)


def test_leading_silence_is_trimmed_and_recorded(tmp_path):
    ir = np.concatenate([np.zeros(100, dtype=np.float32), np.array([1.0, 0.5, 0.25], dtype=np.float32)])
    ir_path = _write_wav(tmp_path / "ir.wav", ir)
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    assert prepared.leading_samples_trimmed == 100
    assert prepared.prepared_frame_count == 3


def test_preview_and_bake_use_the_same_prepared_ir_path(tmp_path):
    """get_prepared_cab_ir (used by both preview and baked-target
    generation, see hybrid/training_target.py::maybe_bake_cab) must be
    cache-consistent: same file + same target rate -> identical result."""
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(4).uniform(-1, 1, 300))
    a = get_prepared_cab_ir(ir_path, 48000)
    b = get_prepared_cab_ir(ir_path, 48000)
    assert a is b  # served from cache
    np.testing.assert_array_equal(a.samples, b.samples)


def test_cab_design_from_prepared_records_fir_history_samples(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(5).uniform(-1, 1, 250))
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    design = cab_design_from_prepared(prepared, original_filename="ir.wav", preview_enabled=True, baked=True)
    assert design.selected is True
    assert design.baked is True
    assert design.fir_history_samples == prepared.prepared_frame_count - 1
    assert design.sha256 == prepared.sha256
