"""Integration tests for render_pair()'s input-profile/calibration wiring.

Uses a fake (identity) render() in place of the real NAM inference (which
shells out to the native nam_render tool) so these run everywhere, same as
the rest of the test suite -- see tests/test_render.py for the real-inference
tests that require the built native tool + a real .nam file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hybrid.pipeline as pipeline
from hybrid.envelope import bounded_causal_envelope_db
from hybrid.input_profiles import db_to_amplitude
from hybrid.nam_loader import NamModel
from hybrid.pipeline import render_pair


def _fake_model(input_level_dbu=None):
    return NamModel(path=Path("fake.nam"), raw={"input_level_dbu": input_level_dbu} if input_level_dbu is not None else {})


@pytest.fixture(autouse=True)
def identity_render(monkeypatch):
    """Replace real NAM inference with an identity pass-through so we can
    inspect exactly what signal each model actually received."""
    def fake_render(model, audio, sample_rate):
        return np.asarray(audio, dtype=np.float32).copy()
    monkeypatch.setattr(pipeline, "render", fake_render)


def _dry(n=4000, sample_rate=48000, amplitude=0.25):
    rng = np.random.default_rng(0)
    return (amplitude * rng.uniform(-1.0, 1.0, n)).astype(np.float32)


def test_input_profile_gain_doubles_amplitude_for_plus6db():
    dry = _dry(amplitude=0.25)
    pair = render_pair(
        _fake_model(), _fake_model(), dry, 48000,
        input_profile_gain_db=6.0, calibration_mode="raw",
    )
    ratio = np.max(np.abs(pair.amp_a)) / np.max(np.abs(dry))
    assert ratio == pytest.approx(db_to_amplitude(6.0), rel=1e-6)


def test_input_profile_gain_halves_amplitude_for_minus6db():
    dry = _dry(amplitude=0.25)
    pair = render_pair(
        _fake_model(), _fake_model(), dry, 48000,
        input_profile_gain_db=-6.0, calibration_mode="raw",
    )
    ratio = np.max(np.abs(pair.amp_a)) / np.max(np.abs(dry))
    assert ratio == pytest.approx(db_to_amplitude(-6.0), rel=1e-6)


def test_envelope_is_derived_from_profiled_dry_not_source_dry():
    dry = _dry()
    pair = render_pair(_fake_model(), _fake_model(), dry, 48000, input_profile_gain_db=6.0, calibration_mode="raw")
    expected = bounded_causal_envelope_db((dry * db_to_amplitude(6.0)).astype(np.float32), 48000)
    np.testing.assert_allclose(pair.envelope_db, expected)
    # And it must NOT match the un-profiled source envelope once gain != 0.
    assert not np.allclose(pair.envelope_db, pair.source_envelope_db)


def test_source_envelope_matches_raw_dry_regardless_of_profile_gain():
    dry = _dry()
    pair = render_pair(_fake_model(), _fake_model(), dry, 48000, input_profile_gain_db=6.0, calibration_mode="raw")
    np.testing.assert_allclose(pair.source_envelope_db, bounded_causal_envelope_db(dry, 48000))


def test_auto_calibration_applies_per_model_gain_both_calibrated():
    dry = _dry()
    pair = render_pair(
        _fake_model(input_level_dbu=8.0), _fake_model(input_level_dbu=12.0), dry, 48000,
        calibration_mode="auto", reference_input_level_dbu=12.0,
    )
    assert pair.calibration_applied is True
    assert pair.amp_a_calibration_gain_db == pytest.approx(4.0)
    assert pair.amp_b_calibration_gain_db == pytest.approx(0.0)
    # Amp A (needing +4dB compensation) must actually receive a hotter signal than Amp B.
    assert np.max(np.abs(pair.amp_a)) > np.max(np.abs(pair.amp_b))
    # But the crossover envelope must stay linked to the common profiled signal,
    # unaffected by either model's own per-model calibration split.
    np.testing.assert_allclose(pair.envelope_db, bounded_causal_envelope_db(pair.profiled_dry, 48000))


def test_auto_calibration_falls_back_to_raw_when_only_one_model_calibrated():
    dry = _dry()
    pair = render_pair(
        _fake_model(input_level_dbu=8.0), _fake_model(input_level_dbu=None), dry, 48000,
        calibration_mode="auto", reference_input_level_dbu=12.0,
    )
    assert pair.calibration_applied is False
    assert pair.calibration_warning is not None
    np.testing.assert_allclose(pair.amp_a, pair.amp_b)


def test_explicit_raw_mode_ignores_calibration_metadata():
    dry = _dry()
    pair = render_pair(
        _fake_model(input_level_dbu=8.0), _fake_model(input_level_dbu=12.0), dry, 48000,
        calibration_mode="raw", reference_input_level_dbu=12.0,
    )
    assert pair.calibration_applied is False
    np.testing.assert_allclose(pair.amp_a, pair.amp_b)


def test_input_peak_dbfs_reflects_profiled_signal():
    dry = np.full(1000, 0.5, dtype=np.float32)
    pair = render_pair(_fake_model(), _fake_model(), dry, 48000, input_profile_gain_db=6.0, calibration_mode="raw")
    expected_peak = 0.5 * db_to_amplitude(6.0)
    assert pair.input_peak_dbfs == pytest.approx(20.0 * np.log10(expected_peak), abs=1e-3)


def test_default_render_pair_is_backward_compatible_raw_zero_gain():
    dry = _dry()
    pair = render_pair(_fake_model(), _fake_model(), dry, 48000)
    # Defaults: guitar / vintage_humbucker / 0 dB gain / auto calibration with
    # uncalibrated models -> effectively raw, unchanged signal.
    np.testing.assert_allclose(pair.profiled_dry, dry)
    np.testing.assert_allclose(pair.amp_a, dry)
    np.testing.assert_allclose(pair.amp_b, dry)
