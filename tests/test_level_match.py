import numpy as np

from hybrid.level_match import compute_crossover_trim


def test_suggested_trim_matches_known_offset():
    n = 44100 * 5
    sr = 44100
    envelope_db = np.full(n, -22.0)  # entire signal sits at the crossover point
    rng = np.random.default_rng(0)
    amp_a = rng.uniform(-1, 1, n) * (10 ** (-16.3 / 20))
    amp_b = rng.uniform(-1, 1, n) * (10 ** (-12.7 / 20))
    result = compute_crossover_trim(envelope_db, amp_a, amp_b, crossover_dbfs=-22.0, transition_width_db=8.0)
    # amp_b is louder than amp_a in this region; suggested trim should be negative
    assert result.suggested_b_trim_db < 0
    assert result.n_samples_in_region == n


def test_widens_region_when_too_few_samples():
    n = 44100
    envelope_db = np.linspace(-40, 0, n)  # only a few samples near -22 dBFS
    amp_a = np.ones(n) * 0.1
    amp_b = np.ones(n) * 0.2
    result = compute_crossover_trim(envelope_db, amp_a, amp_b, crossover_dbfs=-22.0, transition_width_db=0.01)
    assert result.n_samples_in_region > 0
