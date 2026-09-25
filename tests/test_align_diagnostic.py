"""analyse_alignment: synthetic signals with exactly known offsets."""
import numpy as np
import pytest

from hybrid.core import align_diagnostic as ad
from hybrid.core.align_diagnostic import analyse_alignment, select_windows

SR = 48000
NOTE_SPACING = int(0.5 * SR)
N_NOTES = 10


def _notes(seed=0, n_notes=N_NOTES):
    """A DI-like signal: decaying noisy plucks separated by near-silence."""
    rng = np.random.default_rng(seed)
    dry = np.zeros(NOTE_SPACING * (n_notes + 1))
    t = np.arange(int(0.4 * SR)) / SR
    for k in range(n_notes):
        start = NOTE_SPACING * (k + 1) - NOTE_SPACING // 2
        pluck = rng.standard_normal(len(t)) * np.exp(-t * 8) * 0.3
        pluck += 0.2 * np.sin(2 * np.pi * (110 + 20 * k) * t) * np.exp(-t * 4)
        dry[start:start + len(t)] += pluck
    return dry.astype(np.float32)


def _shift(x, samples):
    """Delay (positive) or advance (negative) by an exact number of samples."""
    if samples >= 0:
        return np.concatenate([np.zeros(samples, dtype=x.dtype), x])[: len(x)]
    return np.concatenate([x[-samples:], np.zeros(-samples, dtype=x.dtype)])


def _amp_pair(shift_b, seed=0):
    dry = _notes(seed)
    amp_a = np.tanh(3 * dry).astype(np.float32)
    amp_b = _shift(0.7 * np.tanh(5 * dry), shift_b).astype(np.float32)
    return dry, amp_a, amp_b


@pytest.mark.parametrize("shift", [7, 23, 100])
def test_known_positive_shift_is_a_fixed_offset(shift):
    dry, a, b = _amp_pair(shift)
    result = analyse_alignment(a, b, dry, SR)
    assert result.status == "fixed_offset"
    assert result.recommended_offset_samples == shift
    assert set(result.per_window_offsets) == {shift}
    assert result.agreement_fraction == 1.0
    assert "lags" in result.reason


@pytest.mark.parametrize("shift", [-5, -40])
def test_known_negative_shift_is_a_fixed_offset(shift):
    dry, a, b = _amp_pair(shift)
    result = analyse_alignment(a, b, dry, SR)
    assert result.status == "fixed_offset"
    assert result.recommended_offset_samples == shift
    assert "leads" in result.reason


def test_zero_shift_is_aligned():
    dry, a, b = _amp_pair(0)
    result = analyse_alignment(a, b, dry, SR)
    assert result.status == "aligned"
    assert result.recommended_offset_samples == 0
    assert set(result.per_window_offsets) == {0}


def test_one_sample_is_within_the_aligned_tolerance():
    dry, a, b = _amp_pair(ad.ZERO_OFFSET_TOLERANCE_SAMPLES)
    result = analyse_alignment(a, b, dry, SR)
    assert result.status == "aligned"
    assert result.recommended_offset_samples == 0


def test_several_independent_windows_agree():
    dry, a, b = _amp_pair(12)
    result = analyse_alignment(a, b, dry, SR)
    assert result.windows_analysed >= ad.MIN_WINDOWS
    assert result.windows_reliable == result.windows_analysed
    starts = [w.start_sample for w in result.windows]
    assert all(later - earlier >= w.length_samples for earlier, later, w in zip(starts, starts[1:], result.windows))
    assert set(result.per_window_offsets) == {12}


def test_inconsistent_windows_are_ambiguous():
    """Each note of B is shifted by a different amount: no single fixed delay."""
    dry = _notes()
    a = np.tanh(3 * dry).astype(np.float32)
    b = np.zeros_like(a)
    shifts = [3, 30, -20, 60, 0, -45, 15, 90, -8, 40]
    for k, s in enumerate(shifts):
        seg = slice(NOTE_SPACING * k, NOTE_SPACING * (k + 1))
        b[seg] = _shift(a, s)[seg]
    result = analyse_alignment(a, b, dry, SR)
    assert result.status == "ambiguous"
    assert result.recommended_offset_samples == 0
    assert len(set(result.per_window_offsets)) > 3


def test_uncorrelated_renders_are_ambiguous_not_corrected():
    dry = _notes()
    rng = np.random.default_rng(5)
    a = np.tanh(3 * dry).astype(np.float32)
    b = (rng.standard_normal(len(dry)) * np.abs(dry)).astype(np.float32)
    result = analyse_alignment(a, b, dry, SR)
    assert result.status == "ambiguous"
    assert result.recommended_offset_samples == 0


def test_offset_beyond_the_search_range_is_not_recommended():
    dry, a, b = _amp_pair(ad.DIAGNOSTIC_MAX_LAG_SAMPLES + 200)
    result = analyse_alignment(a, b, dry, SR)
    assert result.status != "fixed_offset"
    assert result.recommended_offset_samples == 0


def test_silence_is_insufficient_signal():
    silence = np.zeros(SR * 5, dtype=np.float32)
    result = analyse_alignment(silence, silence, silence, SR)
    assert result.status == "insufficient_signal"
    assert result.windows_analysed == 0
    assert result.recommended_offset_samples == 0


def test_too_few_notes_is_insufficient_signal():
    dry = _notes(n_notes=ad.MIN_WINDOWS - 1)
    a = np.tanh(3 * dry).astype(np.float32)
    result = analyse_alignment(a, _shift(a, 10), dry, SR)
    assert result.status == "insufficient_signal"


def test_short_input_is_insufficient_signal():
    tiny = np.ones(100, dtype=np.float32) * 0.1
    assert analyse_alignment(tiny, tiny, tiny, SR).status == "insufficient_signal"


def test_diagnostic_does_not_modify_audio():
    dry, a, b = _amp_pair(9)
    copies = [x.copy() for x in (dry, a, b)]
    analyse_alignment(a, b, dry, SR)
    for original, copy in zip((dry, a, b), copies):
        np.testing.assert_array_equal(original, copy)


def test_window_selection_is_deterministic():
    dry = _notes(3)
    assert select_windows(dry, SR) == select_windows(dry.copy(), SR)
    assert len(select_windows(dry, SR)) == N_NOTES


def test_result_serialises_with_per_window_offsets():
    dry, a, b = _amp_pair(7)
    data = analyse_alignment(a, b, dry, SR).to_dict()
    assert data["status"] == "fixed_offset"
    assert data["per_window_offsets"] == [w["offset_samples"] for w in data["windows"]]
    assert data["whole_render_offset_samples"] == 7


def test_production_default_stays_unaligned():
    """The diagnostic changes nothing about what the app renders."""
    import inspect

    from hybrid.core.align import align_to_reference
    from hybrid.core.pipeline import build_hybrid
    from hybrid.modes.fixed_blend import build_fixed_blend

    for fn in (align_to_reference,):
        assert inspect.signature(fn).parameters["enabled"].default is False
    for fn in (build_hybrid, build_fixed_blend):
        assert inspect.signature(fn).parameters["align_enabled"].default is False


def test_consistent_but_weakly_correlated_offset_is_not_recommended():
    """Real captures showed regions agreeing on a lag at ~0.6 correlation that
    moved with the DI material -- tone/phase, not delay. Weak correlation is
    not evidence, however consistent."""
    dry = _notes()
    rng = np.random.default_rng(9)
    a = np.tanh(3 * dry).astype(np.float32)
    noise = rng.standard_normal(len(a)).astype(np.float32) * np.abs(a) * 1.2
    b = (_shift(a, 5) + noise).astype(np.float32)
    result = analyse_alignment(a, b, dry, SR)
    assert set(result.per_window_offsets) == {5}
    assert all(w.correlation < ad.MIN_WINDOW_CORRELATION for w in result.windows)
    assert result.status == "ambiguous"
    assert result.recommended_offset_samples == 0


def test_even_split_between_two_lags_recommends_the_smaller():
    assert ad._round_half_toward_zero(-1.5) == -1
    assert ad._round_half_toward_zero(2.5) == 2
    assert ad._round_half_toward_zero(2.6) == 3
    assert ad._round_half_toward_zero(-7.0) == -7
