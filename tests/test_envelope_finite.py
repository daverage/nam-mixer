"""Tests for the bounded, finite-memory production envelope
(`hybrid.envelope.bounded_causal_envelope_db`) -- see docs/phase3.md section 4.

Unlike `rms_envelope_db`'s one-pole release (recursive, theoretically
infinite memory), this envelope must have a documented, provable, EXACT
maximum dependency on past dry-input samples -- see
`bounded_envelope_max_history_samples`.
"""
from __future__ import annotations

import numpy as np

from hybrid.envelope import (
    BoundedEnvelopeConfig,
    bounded_causal_envelope_db,
    bounded_envelope_max_history_ms,
    bounded_envelope_max_history_samples,
)


def test_bounded_envelope_length_matches_input():
    sr = 48000
    audio = np.random.uniform(-0.5, 0.5, sr * 2).astype(np.float32)
    env = bounded_causal_envelope_db(audio, sr)
    assert len(env) == len(audio)


def test_bounded_envelope_handles_silence():
    sr = 48000
    audio = np.zeros(sr)
    env = bounded_causal_envelope_db(audio, sr)
    assert np.all(np.isfinite(env))


def test_bounded_envelope_handles_empty_input():
    env = bounded_causal_envelope_db(np.zeros(0), 48000)
    assert len(env) == 0


def test_bounded_envelope_is_deterministic():
    sr = 48000
    rng = np.random.default_rng(1)
    audio = rng.uniform(-0.5, 0.5, sr).astype(np.float32)
    env1 = bounded_causal_envelope_db(audio, sr)
    env2 = bounded_causal_envelope_db(audio, sr)
    np.testing.assert_array_equal(env1, env2)


def test_bounded_envelope_tracks_level_change():
    sr = 48000
    quiet = np.random.uniform(-0.01, 0.01, sr)
    loud = np.random.uniform(-0.5, 0.5, sr)
    audio = np.concatenate([quiet, loud]).astype(np.float64)
    env = bounded_causal_envelope_db(audio, sr)
    assert env[-100:].mean() > env[:100].mean() + 10


def test_bounded_envelope_is_causal_future_independent():
    """Same regression as the old envelope's causality test (see
    tests/test_envelope.py): two signals sharing a common head but arbitrarily
    different tails must produce identical envelopes over the shared head."""
    sr = 48000
    rng = np.random.default_rng(4)
    split = sr // 2

    shared_head = rng.uniform(-0.3, 0.3, split)
    tail_a = np.zeros(sr - split)
    tail_b = rng.uniform(-1.0, 1.0, sr - split)

    audio_a = np.concatenate([shared_head, tail_a])
    audio_b = np.concatenate([shared_head, tail_b])

    env_a = bounded_causal_envelope_db(audio_a, sr)
    env_b = bounded_causal_envelope_db(audio_b, sr)

    np.testing.assert_allclose(env_a[:split], env_b[:split])


def test_bounded_envelope_has_exact_bounded_past_dependence():
    """The core new requirement: construct two long signals that are
    completely different in their distant past but identical for the last N
    samples (N = the declared maximum history + a margin). At the comparison
    point, the envelopes must match exactly once every differing sample is
    older than the declared maximum history -- proving the dependency window
    is not just small in practice but exactly bounded.
    """
    sr = 48000
    config = BoundedEnvelopeConfig()
    max_history = bounded_envelope_max_history_samples(sr, config)

    rng = np.random.default_rng(7)
    tail_len = 20000
    lead_len = max_history + 5000  # plenty of differing history before the shared tail

    tail = rng.uniform(-0.4, 0.4, tail_len)
    lead_a = rng.uniform(-1.0, 1.0, lead_len)
    lead_b = rng.uniform(-0.01, 0.01, lead_len)  # wildly different from lead_a

    audio_a = np.concatenate([lead_a, tail]).astype(np.float64)
    audio_b = np.concatenate([lead_b, tail]).astype(np.float64)

    env_a = bounded_causal_envelope_db(audio_a, sr, config)
    env_b = bounded_causal_envelope_db(audio_b, sr, config)

    # Once we're `max_history` samples past the join point, no dependency on
    # the differing lead sections should remain.
    compare_from = lead_len + max_history + 1
    np.testing.assert_allclose(
        env_a[compare_from:], env_b[compare_from:], atol=1e-9,
        err_msg="envelope depends on samples older than the declared maximum history",
    )
    # Sanity: without waiting out the full history, the two signals SHOULD
    # still differ (otherwise this test would pass trivially).
    assert not np.allclose(env_a[lead_len : lead_len + 10], env_b[lead_len : lead_len + 10])


def test_bounded_envelope_max_history_is_within_100ms_target_at_48khz():
    sr = 48000
    ms = bounded_envelope_max_history_ms()
    assert ms <= 100.0
    samples = bounded_envelope_max_history_samples(sr)
    assert samples <= round(100.0 / 1000.0 * sr)


def test_bounded_envelope_config_is_respected():
    sr = 48000
    small = BoundedEnvelopeConfig(rms_window_ms=5.0, attack_avg_ms=2.0, release_window_ms=10.0)
    big = BoundedEnvelopeConfig(rms_window_ms=20.0, attack_avg_ms=5.0, release_window_ms=55.0)
    assert bounded_envelope_max_history_samples(sr, small) < bounded_envelope_max_history_samples(sr, big)
