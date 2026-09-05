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


def test_envelope_is_causal_future_independent():
    """A sample's envelope must not depend on audio that comes after it.

    This becomes the crossover control signal baked into the synthetic
    training target, so it must be causal -- see hybrid/envelope.py's
    docstring. Regression test: two signals identical up to sample `split`
    but arbitrarily different after it must produce identical envelopes up
    to `split` (a centered/non-causal window would leak the future content
    backward and fail this).
    """
    sr = 44100
    rng = np.random.default_rng(4)
    split = sr // 2

    shared_head = rng.uniform(-0.3, 0.3, split)
    tail_a = np.zeros(sr - split)
    tail_b = rng.uniform(-1.0, 1.0, sr - split)

    audio_a = np.concatenate([shared_head, tail_a])
    audio_b = np.concatenate([shared_head, tail_b])

    env_a = rms_envelope_db(audio_a, sr)
    env_b = rms_envelope_db(audio_b, sr)

    np.testing.assert_array_equal(env_a[:split], env_b[:split])
