import numpy as np
import pytest

from hybrid.continuous_gain.multi_blend import GainChain, chain_weights, multi_blend


def test_weights_sum_to_one_and_only_adjacent_mix():
    lv = (-52.0, -32.0, -16.0)
    env = np.linspace(-70, 0, 200)
    w = chain_weights(env, lv)
    assert np.allclose(w.sum(0), 1.0)
    assert np.all((w > 1e-12).sum(0) <= 2)
    assert np.all(w[2][env <= -32] == 0) and np.all(w[0][env >= -32] == 0)


def test_endpoints_and_exact_capture_levels():
    lv = (-52.0, -32.0, -16.0)
    w = chain_weights(np.array([-90.0, -52.0, -32.0, -16.0, 5.0]), lv)
    assert np.allclose(w, [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])


def test_multi_blend_and_designated_gain():
    c = GainChain((1.0, 10.0), (-50.0, -20.0), -30.0)
    assert c.designated_gain_db(0) == -20.0 and c.input_scale_db(1) == -10.0
    out = multi_blend(np.array([-60.0, -20.0]), [np.ones(2), 3 * np.ones(2)], c)
    assert np.allclose(out, [1.0, 3.0])


def test_rejects_unsorted_levels():
    with pytest.raises(ValueError):
        GainChain((1.0, 2.0), (-20.0, -30.0), -30.0)
