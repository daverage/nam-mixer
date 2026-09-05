import numpy as np

from hybrid.coverage import analyse_profile_coverage, envelope_percentiles, suggest_crossover_dbfs


def _synthetic_envelope(n=10000):
    # Spans well below and above a typical crossover band, all "active"
    # (above the -50 dBFS active-signal threshold).
    return np.linspace(-45.0, -5.0, n)


def test_coverage_monotonic_across_increasing_profile_gain():
    envelope_db = _synthetic_envelope()
    profiles = [
        ("vintage_single", -7.0),
        ("vintage_humbucker", 0.0),
        ("hot_humbucker", 4.5),
        ("extreme_passive", 6.0),
    ]
    results = analyse_profile_coverage(envelope_db, profiles, crossover_dbfs=-20.0, transition_width_db=8.0)

    b_fractions = [r.amp_b_fraction for r in results]
    a_fractions = [r.amp_a_fraction for r in results]
    assert b_fractions == sorted(b_fractions)
    assert a_fractions == sorted(a_fractions, reverse=True)
    # Fractions must be a real probability split of the active signal.
    for r in results:
        assert abs(r.amp_a_fraction + r.transition_fraction + r.amp_b_fraction - 1.0) < 1e-9


def test_coverage_all_below_crossover_is_all_amp_a():
    envelope_db = np.full(1000, -60.0)
    results = analyse_profile_coverage(envelope_db, [("x", 0.0)], crossover_dbfs=-10.0, transition_width_db=4.0)
    assert results[0].amp_a_fraction == 1.0
    assert results[0].amp_b_fraction == 0.0


def test_coverage_ignores_silence_below_active_threshold():
    # Mostly silence, with a short loud burst well above the crossover.
    envelope_db = np.concatenate([np.full(9000, -80.0), np.full(1000, -5.0)])
    results = analyse_profile_coverage(envelope_db, [("x", 0.0)], crossover_dbfs=-20.0, transition_width_db=4.0)
    # Silence samples are excluded, so the "active" set is 100% the loud burst.
    assert results[0].amp_b_fraction == 1.0


def test_envelope_percentiles_ordering():
    envelope_db = np.linspace(-60.0, 0.0, 1000)
    p = envelope_percentiles(envelope_db)
    assert p["p10"] < p["p25"] < p["p50"] < p["p75"] < p["p90"]


def test_suggest_crossover_within_active_range():
    envelope_db = _synthetic_envelope()
    suggestion = suggest_crossover_dbfs(envelope_db)
    assert -45.0 < suggestion < -5.0


def test_suggest_crossover_none_when_fully_silent():
    assert suggest_crossover_dbfs(np.full(1000, -80.0)) is None
