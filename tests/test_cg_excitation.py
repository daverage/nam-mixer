"""The single-file Continuous Gain excitation: deterministic, level-swept across the requested range, calibrated to a reference level."""
import numpy as np

from hybrid.cg_excitation import SR, active_rms, level_swept_excitation


def test_deterministic_and_sized():
    a = level_swept_excitation(12.0, seed=3, period_s=6.0)
    b = level_swept_excitation(12.0, seed=3, period_s=6.0)
    assert a.dtype == np.float32 and len(a) == 12 * SR and np.array_equal(a, b) and np.isfinite(a).all()
    assert not np.array_equal(a, level_swept_excitation(12.0, seed=4, period_s=6.0))


def test_level_sweeps_the_whole_range_and_is_calibrated():
    x = level_swept_excitation(40.0, seed=1, lo_db=-30.0, hi_db=18.0, period_s=20.0, reference_rms=0.05)
    frame = SR // 2
    rms_db = 20 * np.log10(np.sqrt(np.mean(x[: len(x) // frame * frame].reshape(-1, frame) ** 2, axis=1)) + 1e-9)
    assert rms_db.max() - rms_db.min() > 40                                    # the sweep is really there
    t = (np.arange(len(rms_db)) + 0.5) * 0.5
    mid = (t % 20 > 9) & (t % 20 < 11)                                         # around the triangle peak (hi_db)
    low = (t % 20 < 1) | (t % 20 > 19)                                         # around the triangle floor (lo_db)
    assert np.median(rms_db[mid]) - np.median(rms_db[low]) > 38
    assert abs(active_rms(level_swept_excitation(30.0, seed=2, lo_db=0.0, hi_db=0.0, reference_rms=0.05)) - 0.05) < 0.012


def test_material_is_guitar_like_not_noise():
    x = level_swept_excitation(20.0, seed=5, lo_db=0.0, hi_db=0.0)
    spec = np.abs(np.fft.rfft(x[SR:] * np.hanning(len(x) - SR)))
    f = np.fft.rfftfreq(len(x) - SR, 1 / SR)
    low, high = spec[(f > 70) & (f < 1500)].sum(), spec[(f > 6000) & (f < 16000)].sum()
    assert low > 20 * high                                                     # guitar range with a natural top-end roll-off (the bundled DIs measure 55-200x)


def test_top_bias_spends_more_time_loud_and_default_is_unchanged():
    kw = dict(seed=9, lo_db=-30.0, hi_db=20.0, period_s=10.0)
    plain = level_swept_excitation(20.0, **kw)
    assert np.array_equal(plain, level_swept_excitation(20.0, top_bias=1.0, **kw))
    biased = level_swept_excitation(20.0, top_bias=0.5, **kw)
    frame = SR // 2
    db = lambda x: 20 * np.log10(np.sqrt(np.mean(x[: len(x) // frame * frame].reshape(-1, frame) ** 2, axis=1)) + 1e-9)
    assert np.median(db(biased)) > np.median(db(plain)) + 3
