import numpy as np

from hybrid.envelope import rms_envelope_db
from hybrid.pipeline import RenderedPair, build_hybrid


def _synthetic_pair(n=10000, sample_rate=48000):
    rng = np.random.default_rng(0)
    dry = np.linspace(-1.0, 1.0, n).astype(np.float32) * rng.uniform(0.5, 1.0, n).astype(np.float32)
    amp_a = np.full(n, 1.0, dtype=np.float32)
    amp_b = np.full(n, 2.0, dtype=np.float32)
    envelope_db = rms_envelope_db(dry, sample_rate)
    return RenderedPair(dry=dry, amp_a=amp_a, amp_b=amp_b, envelope_db=envelope_db, sample_rate=sample_rate)


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
    assert result.manual_trim_db == 0.0
    assert result.effective_b_trim_db == 0.0
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
    assert result.effective_b_trim_db == result.auto_trim_db


def test_build_hybrid_manual_trim_combines_with_auto_trim():
    """Auto-match gives a starting point; the manual tweak should add to it,
    not replace it."""
    pair = _synthetic_pair()
    auto_only = build_hybrid(
        pair,
        crossover_dbfs=-22.0,
        transition_width_db=8.0,
        auto_level=True,
    )
    auto_plus_manual = build_hybrid(
        pair,
        crossover_dbfs=-22.0,
        transition_width_db=8.0,
        auto_level=True,
        manual_b_trim_db=0.7,
    )
    assert auto_plus_manual.auto_trim_db == auto_only.auto_trim_db
    assert auto_plus_manual.manual_trim_db == 0.7
    assert auto_plus_manual.effective_b_trim_db == auto_only.auto_trim_db + 0.7


def test_build_hybrid_dry_gain_db_shifts_envelope_and_pushes_toward_amp_b():
    """dry_gain_db is the test-only "pretend I played louder" control -- it
    should shift the whole envelope up by exactly that many dB (an exact
    log-domain identity, not an approximation) and therefore push the blend
    weight toward Amp B for a crossover point the unshifted signal never
    reaches."""
    pair = _synthetic_pair()
    quiet = build_hybrid(pair, crossover_dbfs=0.0, transition_width_db=2.0, auto_level=False)
    loud = build_hybrid(pair, crossover_dbfs=0.0, transition_width_db=2.0, auto_level=False, dry_gain_db=40.0)

    np.testing.assert_allclose(loud.envelope_db, pair.envelope_db + 40.0)
    assert quiet.blend_curve.max() < 0.5
    assert loud.blend_curve.max() > 0.5


def test_build_hybrid_is_cheap_to_call_repeatedly_on_same_pair():
    """Different crossover points on the same RenderedPair shouldn't require
    re-rendering -- this is the whole point of splitting render_pair out."""
    pair = _synthetic_pair()
    r1 = build_hybrid(pair, crossover_dbfs=-30.0, transition_width_db=8.0, auto_level=False)
    r2 = build_hybrid(pair, crossover_dbfs=-10.0, transition_width_db=8.0, auto_level=False)
    assert not np.allclose(r1.blend_curve, r2.blend_curve)
