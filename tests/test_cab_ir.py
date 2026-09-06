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


# -- Cabinet energy diagnostics -- docs/blend-mode.md "CABINET ENERGY ANALYSIS" ---
# Purely informational: must never alter the actual FIR taps/convolution.

def test_energy_one_tap_ir_reports_immediate_effective_energy(tmp_path):
    """A single-tap (all energy in sample 0) IR should report every energy
    percentile as reached at sample 1 -- immediate effective energy, no long
    tail to speak of."""
    ir_path = _write_wav(tmp_path / "ir.wav", [1.0])
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    assert prepared.energy_99_samples == 1
    assert prepared.energy_999_samples == 1
    assert prepared.energy_9999_samples == 1
    assert prepared.energy_99_ms == pytest.approx(1 / 48000 * 1000)


def test_energy_percentile_indices_are_known_for_synthetic_ir():
    """Construct an IR where the energy distribution is exactly known: 100
    equal-energy samples followed by 100 near-zero (but not literally
    silent, so leading/trailing trimming doesn't interfere) samples --
    99%/99.9%/99.99% energy must all be reached within the first ~100
    "loud" samples, well before the tail is exhausted."""
    from hybrid.cab_ir import _energy_percentile_sample_counts

    loud = np.full(100, 1.0, dtype=np.float32)
    quiet = np.full(100, 1e-6, dtype=np.float32)  # negligible energy contribution
    ir = np.concatenate([loud, quiet])

    total_energy, counts = _energy_percentile_sample_counts(ir)
    assert total_energy == pytest.approx(100.0, rel=1e-3)
    assert counts["99"] <= 100
    assert counts["999"] <= 101
    assert counts["9999"] <= 105
    # Monotonic: reaching a higher percentile never needs FEWER samples.
    assert counts["99"] <= counts["999"] <= counts["9999"]


def test_energy_fractions_are_finite_and_bounded(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(6).uniform(-1, 1, 1000))
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    for window in (0, 1, 10, 500, 1000, 10000):
        frac = prepared.energy_fraction_within(window)
        assert np.isfinite(frac)
        assert 0.0 <= frac <= 1.0
    # The full prepared length must always contain (approximately) all the energy.
    assert prepared.energy_fraction_within(prepared.prepared_frame_count) == pytest.approx(1.0, abs=1e-9)


def test_energy_analysis_never_alters_prepared_samples_or_convolution(tmp_path):
    """Computing energy diagnostics must be side-effect-free -- the same
    prepared IR used for the energy report is the one actually convolved,
    and its taps must be bit-identical to a version with no energy analysis
    performed (there's only one code path, but this pins that invariant)."""
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(7).uniform(-1, 1, 2000))
    prepared_a = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    prepared_b = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    np.testing.assert_array_equal(prepared_a.samples, prepared_b.samples)

    audio = np.random.default_rng(8).uniform(-1, 1, 5000).astype(np.float32)
    out_a = apply_cab_ir(audio, prepared_a)
    out_b = apply_cab_ir(audio, prepared_b)
    np.testing.assert_array_equal(out_a, out_b)


def test_energy_reported_duration_matches_prepared_frame_count(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(9).uniform(-1, 1, 4800))
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    assert prepared.prepared_duration_ms == pytest.approx(prepared.prepared_frame_count / 48000 * 1000)


def test_energy_sample_rate_preparation_reports_durations_correctly(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(10).uniform(-1, 1, 480), sample_rate=48000)
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=24000)
    assert prepared.sample_rate == 24000
    assert prepared.energy_99_ms == pytest.approx(prepared.energy_99_samples / 24000 * 1000)
    assert prepared.prepared_duration_ms == pytest.approx(prepared.prepared_frame_count / 24000 * 1000)


def test_cab_design_from_prepared_carries_energy_diagnostics(tmp_path):
    ir_path = _write_wav(tmp_path / "ir.wav", np.random.default_rng(11).uniform(-1, 1, 1000))
    prepared = load_and_prepare_cab_ir(ir_path, target_sample_rate=48000)
    design = cab_design_from_prepared(prepared, original_filename="ir.wav", preview_enabled=True, baked=True)
    assert design.energy_99_samples == prepared.energy_99_samples
    assert design.energy_999_ms == pytest.approx(prepared.energy_999_ms)
    assert design.energy_9999_samples == prepared.energy_9999_samples
    assert design.prepared_duration_ms == pytest.approx(prepared.prepared_duration_ms)
