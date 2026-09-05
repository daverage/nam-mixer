import numpy as np

from hybrid.align import align_to_reference, estimate_offset


def test_estimate_offset_detects_known_shift():
    rng = np.random.default_rng(1)
    n = 5000
    signal = rng.uniform(-1, 1, n)
    shift = 37
    shifted = np.concatenate([np.zeros(shift), signal])[:n]
    offset = estimate_offset(signal, shifted, max_lag=100)
    assert offset == shift



def test_align_to_reference_matches_length():
    rng = np.random.default_rng(2)
    n = 2000
    reference = rng.uniform(-1, 1, n)
    other = np.concatenate([np.zeros(10), reference])[:n]
    aligned, offset = align_to_reference(reference, other, max_lag=50)
    assert len(aligned) == len(reference)
