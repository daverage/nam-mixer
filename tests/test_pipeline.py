import numpy as np

from hybrid.pipeline import RenderedPair, build_hybrid


def _synthetic_pair(n=10000, sample_rate=48000):
    rng = np.random.default_rng(0)
    dry = np.linspace(-1.0, 1.0, n).astype(np.float32) * rng.uniform(0.5, 1.0, n).astype(np.float32)
    amp_a = np.full(n, 1.0, dtype=np.float32)
    amp_b = np.full(n, 2.0, dtype=np.float32)
    return RenderedPair(dry=dry, amp_a=amp_a, amp_b=amp_b, sample_rate=sample_rate)


def test_build_hybrid_manual_trim_no_auto_level():
    pair = _synthetic_pair()
    result = build_hybrid(
        pair,
        crossover_dbfs=-22.0,
        transition_width_db=8.0,
        auto_level=False,
        manual_b_trim_db=0.0,
    )
    assert result.level_match is None
    assert result.auto_trim_db == 0.0
    assert len(result.hybrid) == len(pair.dry)
    assert len(result.blend_curve) == len(pair.dry)


def test_build_hybrid_auto_level_returns_level_match_result():
    pair = _synthetic_pair()
    result = build_hybrid(
        pair,
        crossover_dbfs=-22.0,
        transition_width_db=8.0,
        auto_level=True,
    )
    assert result.level_match is not None
    assert np.isfinite(result.auto_trim_db)


def test_build_hybrid_is_cheap_to_call_repeatedly_on_same_pair():
    """Different crossover points on the same RenderedPair shouldn't require
    re-rendering -- this is the whole point of splitting render_pair out."""
    pair = _synthetic_pair()
    r1 = build_hybrid(pair, crossover_dbfs=-30.0, transition_width_db=8.0, auto_level=False)
    r2 = build_hybrid(pair, crossover_dbfs=-10.0, transition_width_db=8.0, auto_level=False)
    assert not np.allclose(r1.blend_curve, r2.blend_curve)
