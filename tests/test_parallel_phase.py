import numpy as np

from hybrid.core.parallel_phase import (
    analyse_parallel_compatibility,
    analyse_parallel_with_polarity_choice,
)


SR = 48_000


def _signal(seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.45 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)


def test_in_phase_parallel_sum_is_safe_and_both_present():
    x = _signal()
    report = analyse_parallel_compatibility(0.5 * x, 0.5 * x, SR)
    assert report.status == "safe"
    assert report.presence_status == "both_present"
    assert report.worst_band is not None
    assert report.worst_band.interaction_db > 2.5


def test_reversed_polarity_is_a_problem_and_flip_is_recommended():
    x = _signal()
    report = analyse_parallel_with_polarity_choice(0.5 * x, -0.5 * x, SR)
    assert report.status == "problem"
    assert report.polarity_recommended is True
    assert report.recommended_polarity == "inverted"
    assert report.alternative_status == "safe"


def test_selected_flip_reports_the_improved_sum():
    x = _signal()
    report = analyse_parallel_with_polarity_choice(0.5 * x, -0.5 * x, SR, polarity_inverted=True)
    assert report.status == "safe"
    assert report.polarity_inverted is True
    assert report.polarity_recommended is True


def test_bad_selected_flip_recommends_restoring_original():
    x = _signal()
    report = analyse_parallel_with_polarity_choice(0.5 * x, 0.5 * x, SR, polarity_inverted=True)
    assert report.status == "problem"
    assert report.recommended_polarity == "original"


def test_mix_endpoint_reports_absent_source_without_false_phase_failure():
    x = _signal()
    report = analyse_parallel_compatibility(x, np.zeros_like(x), SR)
    assert report.presence_status == "amp_b_absent"
    assert report.status == "insufficient_overlap"


def test_different_frequency_content_is_not_called_phase_failure():
    t = np.arange(SR) / SR
    a = np.sin(2 * np.pi * 180 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 1800 * t).astype(np.float32)
    report = analyse_parallel_compatibility(0.5 * a, 0.5 * b, SR)
    assert report.status != "problem"
