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


def _two_region_pair(n=20000, sample_rate=48000):
    """A pair where the dry signal is quiet in the first half and loud in the
    second half, and Amp B's rendered loudness relative to Amp A is DIFFERENT
    in each half (10x in the quiet half, 1x -- i.e. matched -- in the loud
    half). This makes it possible to tell which half of the recording
    auto-level-match actually measured from."""
    dry = np.concatenate([
        np.full(n // 2, 0.01, dtype=np.float32),   # quiet half: envelope ~ -40 dBFS
        np.full(n - n // 2, 0.56, dtype=np.float32),  # loud half: envelope ~ -5 dBFS
    ])
    amp_a = np.full(n, 1.0, dtype=np.float32)
    amp_b = np.concatenate([
        np.full(n // 2, 10.0, dtype=np.float32),   # quiet half: Amp B 10x hotter than Amp A
        np.full(n - n // 2, 1.0, dtype=np.float32),  # loud half: Amp B matches Amp A
    ])
    envelope_db = rms_envelope_db(dry, sample_rate)
    return RenderedPair(dry=dry, amp_a=amp_a, amp_b=amp_b, envelope_db=envelope_db, sample_rate=sample_rate)


def test_auto_trim_ignores_dry_gain_db_and_measures_the_real_input_level():
    """Auto-level-match must always measure Amp A/B's loudness at the REAL
    crossover input level, never at the dry_gain_db-shifted one -- otherwise
    sweeping the test-only input gain would silently change which stretch of
    the recording gets used for level matching, producing a trim that has
    nothing to do with how the amps actually sound at that input level (see
    the build_hybrid docstring)."""
    pair = _two_region_pair()

    # crossover_dbfs=-5 sits in the LOUD half in real terms, where Amp A/B
    # are matched (0 dB trim expected) regardless of dry_gain_db.
    no_gain = build_hybrid(pair, crossover_dbfs=-5.0, transition_width_db=2.0, auto_level=True)
    with_gain = build_hybrid(
        pair, crossover_dbfs=-5.0, transition_width_db=2.0, auto_level=True, dry_gain_db=35.0
    )

    assert no_gain.auto_trim_db == with_gain.auto_trim_db
    assert abs(no_gain.auto_trim_db) < 0.5

    # Sanity check: at dry_gain_db=35 the BLEND (not the trim) does move,
    # since the shifted envelope is what should drive the crossfade.
    quiet_blend = build_hybrid(pair, crossover_dbfs=-5.0, transition_width_db=2.0, auto_level=True)
    loud_blend = build_hybrid(
        pair, crossover_dbfs=-5.0, transition_width_db=2.0, auto_level=True, dry_gain_db=35.0
    )
    assert not np.allclose(quiet_blend.blend_curve, loud_blend.blend_curve)


def test_build_hybrid_is_cheap_to_call_repeatedly_on_same_pair():
    """Different crossover points on the same RenderedPair shouldn't require
    re-rendering -- this is the whole point of splitting render_pair out."""
    pair = _synthetic_pair()
    r1 = build_hybrid(pair, crossover_dbfs=-30.0, transition_width_db=8.0, auto_level=False)
    r2 = build_hybrid(pair, crossover_dbfs=-10.0, transition_width_db=8.0, auto_level=False)
    assert not np.allclose(r1.blend_curve, r2.blend_curve)
