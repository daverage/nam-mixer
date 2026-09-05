import numpy as np

from hybrid.blend import CrossoverConfig, blend, blend_weight, smoothstep_curve


def test_smoothstep_endpoints():
    t = np.array([0.0, 1.0])
    out = smoothstep_curve(t)
    assert np.allclose(out, [0.0, 1.0])


def test_blend_weight_quiet_is_amp_a():
    envelope_db = np.full(100, -40.0)
    config = CrossoverConfig(crossover_dbfs=-22.0, transition_width_db=8.0)
    t = blend_weight(envelope_db, config)
    assert np.allclose(t, 0.0)


def test_blend_weight_loud_is_amp_b():
    envelope_db = np.full(100, 0.0)
    config = CrossoverConfig(crossover_dbfs=-22.0, transition_width_db=8.0)
    t = blend_weight(envelope_db, config)
    assert np.allclose(t, 1.0)


def test_blend_output_is_pure_a_when_quiet():
    n = 1000
    envelope_db = np.full(n, -40.0)
    amp_a = np.ones(n)
    amp_b = np.full(n, 5.0)
    config = CrossoverConfig(crossover_dbfs=-22.0, transition_width_db=8.0)
    hybrid, t = blend(envelope_db, amp_a, amp_b, config)
    assert np.allclose(hybrid, amp_a)
    assert np.allclose(t, 0.0)


def test_blend_no_discontinuity_across_transition():
    n = 10000
    envelope_db = np.linspace(-40, 0, n)
    amp_a = np.ones(n)
    amp_b = np.ones(n) * 1.2
    config = CrossoverConfig(crossover_dbfs=-22.0, transition_width_db=8.0)
    hybrid, _ = blend(envelope_db, amp_a, amp_b, config)
    jumps = np.abs(np.diff(hybrid))
    assert jumps.max() < 0.01
