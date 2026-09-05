import numpy as np

from hybrid.envelope import rms_envelope_db


def test_envelope_length_matches_input():
    sr = 44100
    audio = np.random.uniform(-0.5, 0.5, sr * 2).astype(np.float32)
    env = rms_envelope_db(audio, sr)
    assert len(env) == len(audio)


def test_envelope_tracks_level_change():
    sr = 44100
    quiet = np.random.uniform(-0.01, 0.01, sr)
    loud = np.random.uniform(-0.5, 0.5, sr)
    audio = np.concatenate([quiet, loud]).astype(np.float64)
    env = rms_envelope_db(audio, sr, attack_ms=1.0, release_ms=1.0)
    # well after the transition, envelope should reflect the louder section
    assert env[-100:].mean() > env[:100].mean() + 10


def test_envelope_handles_silence():
    sr = 44100
    audio = np.zeros(sr)
    env = rms_envelope_db(audio, sr)
    assert np.all(np.isfinite(env))
