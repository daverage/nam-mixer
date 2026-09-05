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
    aligned, offset = align_to_reference(reference, other, max_lag=50, enabled=True)
    assert len(aligned) == len(reference)


def test_align_to_reference_disabled_by_default():
    """Regression test: alignment must be opt-in, not applied automatically.

    See hybrid/align.py's module docstring -- cross-correlating two tonally
    dissimilar amp renders can misread a real tonal/phase difference as
    latency, so correction must not happen unless the caller explicitly asks
    for it via enabled=True.
    """
    rng = np.random.default_rng(3)
    n = 2000
    reference = rng.uniform(-1, 1, n)
    shift = 37
    other = np.concatenate([np.zeros(shift), reference])[:n]

    aligned, offset = align_to_reference(reference, other, max_lag=100)

    assert offset == 0
    assert len(aligned) == n
    np.testing.assert_array_equal(aligned, other[:n])
