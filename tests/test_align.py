import numpy as np

from hybrid.core.align import align_to_reference, estimate_offset


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

    See hybrid/core/align.py's module docstring -- cross-correlating two tonally
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


def _reference_loop_offset(reference, other, max_lag):
    """The original per-lag loop estimate_offset replaced; the FFT version must
    pick exactly the same lag (including the first-maximum tie-break)."""
    reference = np.asarray(reference, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    n = min(len(reference), len(other))
    if n == 0:
        return 0
    ref = reference[:n] - reference[:n].mean()
    oth = other[:n] - other[:n].mean()
    max_lag = min(max_lag, n - 1)
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        a, b = (ref[: n - lag], oth[lag:]) if lag >= 0 else (ref[-lag:], oth[: n + lag])
        if len(a) < 2:
            continue
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            continue
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag


def test_fft_offset_matches_the_original_loop():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(3, 4000))
        x = rng.standard_normal(n)
        y = np.roll(x, int(rng.integers(-300, 300))) * rng.uniform(0.2, 3) + rng.standard_normal(n) * rng.uniform(0, 2)
        max_lag = int(rng.integers(0, 2500))
        for a, b in ((x, y), (y, x), (x[: n // 2], y), (np.zeros(n), y)):
            assert estimate_offset(a, b, max_lag) == _reference_loop_offset(a, b, max_lag)
