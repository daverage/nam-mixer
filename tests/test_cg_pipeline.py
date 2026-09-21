"""Continuous Gain measurement pipeline (probe -> audit -> profile -> selection -> anchors) on synthetic amps."""
from __future__ import annotations

import numpy as np
import pytest

from hybrid.cg_anchors import effective_min_sep, fixed_ladder_anchors, mapping_table, position_input_gain_db, response_anchors
from hybrid.cg_audit import alignment_shift, audit_captures
from hybrid.cg_probe import FIT_DIS, probe_capture
from hybrid.cg_profile import build_profile
from hybrid.cg_selection import (GROUPS, MAX_EXHAUSTIVE_ELIGIBLE, PHYS, evaluate_set, resolve_selection, select_captures)
from tests.cg_synth import SR, amp_render, synth_di

GAINS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _load_di(name):
    return synth_di(name, 4.0)


@pytest.fixture(scope="module")
def measured():
    renders = {g: amp_render(g * 1.5) for g in GAINS}
    probes = {g: probe_capture(renders[g], _load_di, seconds=4) for g in GAINS}
    meta = {g: {"loudness": None, "gear_make": "x", "gear_model": "y", "sample_rate": SR, "input_level_dbu": None} for g in GAINS}
    audit = audit_captures(GAINS, probes, meta, lambda g, x: renders[g](x), _load_di)
    return renders, probes, audit, build_profile(probes, audit)


def test_probe_bank_shape(measured):
    _, probes, _, _ = measured
    p = probes[3.0]
    assert set(p) == {"click_lo", "click_hi", "silence", "tones", "music"}
    assert len(p["tones"]) == 12 and len(p["music"]) == len(FIT_DIS) * 4
    assert {"rms_db", "hf3k_db", "crest_db", "dyn_range_db", "eq_mid_db", "clipped_frac", "nan"} <= set(p["music"]["clean_smooth@0"])


def test_probe_measures_more_drive_as_more_saturation(measured):
    _, probes, _, _ = measured
    crest = [probes[g]["music"]["clean_smooth@0"]["crest_db"] for g in GAINS]
    assert crest[0] > crest[-1]                      # a harder-driven amp has a lower crest factor


def test_clean_synthetic_set_audits_valid(measured):
    _, _, audit, _ = measured
    assert {v["status"] for v in audit["captures"].values()} <= {"VALID", "CORRECTED", "SUSPECT"}
    assert audit["captures"]["3"]["status"] in ("VALID", "SUSPECT")
    assert all(alignment_shift(v) == 0 for v in audit["captures"].values())


def test_a_verified_timing_offset_is_corrected_and_the_shift_realigns_it():
    delays = {g: (40 if g == 3.0 else 0) for g in GAINS}
    renders = {g: amp_render(g * 1.5, delay=delays[g]) for g in GAINS}
    probes = {g: probe_capture(renders[g], _load_di, seconds=4) for g in GAINS}
    meta = {g: {"sample_rate": SR} for g in GAINS}
    audit = audit_captures(GAINS, probes, meta, lambda g, x: renders[g](x), _load_di)
    a = audit["captures"]["3"]
    assert a["status"] == "CORRECTED" and a["correction"]["samples"] == -40
    sh = alignment_shift(a)
    x = _load_di("clean_smooth")
    y = renders[3.0](x)
    ref = amp_render(3.0 * 1.5)(x)
    fixed = np.concatenate([y[sh:], np.zeros(sh, y.dtype)])
    assert np.allclose(fixed[: len(x) - 100], ref[: len(x) - 100], atol=1e-6)
    assert all(audit["captures"][k]["status"] == "VALID" for k in ("1", "2", "4", "5", "6") if audit["captures"][k]["status"] != "SUSPECT")


def test_a_noise_spike_makes_a_capture_suspect_and_analysis_only():
    renders = {g: amp_render(g * 1.5, noise_db=(-45 if g == 4.0 else -100)) for g in GAINS}
    probes = {g: probe_capture(renders[g], _load_di, seconds=4) for g in GAINS}
    audit = audit_captures(GAINS, probes, {g: {"sample_rate": SR} for g in GAINS}, lambda g, x: renders[g](x), _load_di)
    assert audit["captures"]["4"]["status"] == "SUSPECT"
    profile = build_profile(probes, audit)
    assert 4.0 in profile["quarantine"]["noise"]
    an = select_captures(profile, audit)
    assert 4.0 not in an["eligible"] and an["ineligible"]["4"] == "SUSPECT"


def test_profile_series_cover_every_selection_measure(measured):
    _, _, _, profile = measured
    needed = {n for g in GROUPS.values() for n in g} | {n for names, _ in PHYS.values() for n in names}
    assert needed <= set(profile["series"])
    assert all(len(s["values"]) == len(GAINS) for s in profile["series"].values())


def test_selection_search_invariants(measured):
    _, _, audit, profile = measured
    an = select_captures(profile, audit)
    js = [an["by_k"][str(k)]["J"] for k in sorted(map(int, an["by_k"]))]
    assert all(b <= a + 1e-12 for a, b in zip(js, js[1:]))          # more captures never cover worse
    assert an["by_k"][str(len(GAINS))]["J"] == 0.0
    assert an["exhaustive"] and an["response_coordinate"]["arc"][0] == 0 and abs(an["response_coordinate"]["arc"][-1] - 1) < 1e-9
    assert an["greedy_order"] and len(an["greedy_order"]) == len(GAINS)


def _rough_profile(n, statuses=None):
    rng = np.random.default_rng(3)
    gains = [float(i) for i in range(1, n + 1)]
    names = {x for g in GROUPS.values() for x in g} | {x for names, _ in PHYS.values() for x in names}
    series = {nm: {"dimension": "tone", "unit": "dB", "values": (rng.standard_normal(n) * 10).tolist(), "reliable": [True] * n} for nm in names}
    audit = {"captures": {f"{g:g}": {"status": (statuses or {}).get(g, "VALID"), "evidence": [], "correction": None} for g in gains}}
    return {"gains": gains, "series": series}, audit


def test_no_subset_satisfying_the_coverage_rule_is_reported_honestly_with_a_fallback():
    profile, audit = _rough_profile(7)
    an = select_captures(profile, audit)
    assert an["k_star_all_within_tolerance"] is None
    r = resolve_selection(an, profile, audit, "automatic")
    assert r["fallback"] is True and r["selected"] == an["eligible"] and "No optimum is claimed" in r["notes"][0]


def test_too_many_captures_for_an_exhaustive_search_never_invents_an_optimum():
    n = MAX_EXHAUSTIVE_ELIGIBLE + 2
    profile, audit = _rough_profile(n)
    an = select_captures(profile, audit)
    assert not an["exhaustive"] and an["by_k"] == {}
    r = resolve_selection(an, profile, audit, "automatic")
    assert r["fallback"] and r["selected"] == an["eligible"] and "exhaustive-search limit" in r["notes"][0]


def test_modes_use_all_and_custom(measured):
    _, _, audit, profile = measured
    an = select_captures(profile, audit)
    assert resolve_selection(an, profile, audit, "use_all")["selected"] == an["eligible"]
    c = resolve_selection(an, profile, audit, "custom", [1, 3, 6])
    assert c["selected"] == [1.0, 3.0, 6.0] and c["coverage"]["reasons"]["1"]["role"] == "selected" and c["coverage"]["reasons"]["2"]["role"] == "omitted"
    with pytest.raises(ValueError):
        resolve_selection(an, profile, audit, "custom", [1])
    with pytest.raises(ValueError):
        resolve_selection(an, profile, audit, "custom", [1, 99])
    with pytest.raises(ValueError):
        resolve_selection(an, profile, audit, "bogus")


def test_custom_may_include_a_suspect_capture_only_with_a_warning():
    profile, audit = _rough_profile(5, {3.0: "SUSPECT"})
    an = select_captures(profile, audit)
    assert 3.0 not in an["eligible"]
    r = resolve_selection(an, profile, audit, "custom", [1, 3, 5])
    assert any("not VALID/CORRECTED" in n for n in r["notes"])
    ev = evaluate_set(profile, audit, [1, 5])
    assert ev["reasons"]["3"]["role"] == "needs_review"


# ---- anchors
RC = {"gains": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "arc": [0, 0.2, 0.32, 0.4, 0.48, 0.55, 0.62, 0.7, 0.85, 1.0]}


def test_fc_anchors_span_the_plugin_range_and_keep_separation():
    pos, anc = response_anchors(RC, [1, 2, 4, 10])
    assert pos == [1.0, 2.0, 4.0, 10.0] and anc[0] == -20.0 and anc[-1] == 14.0
    assert all(b - a >= 4.0 - 1e-9 for a, b in zip(anc, anc[1:])) and all(round(a, 1) == a for a in anc)


def test_fc_anchor_rule_is_scale_free_in_the_response_coordinate():
    rc2 = {"gains": RC["gains"], "arc": [a * 7 for a in RC["arc"]]}
    assert response_anchors(RC, [1, 3, 6, 10]) == response_anchors(rc2, [1, 3, 6, 10])


def test_more_than_nine_captures_relax_the_separation_only_as_needed():
    assert effective_min_sep(9) == 4.0 and effective_min_sep(10) == pytest.approx(34 / 9)
    _, anc = response_anchors(RC, list(range(1, 11)))
    assert anc[0] == -20.0 and anc[-1] == 14.0 and all(b > a for a, b in zip(anc, anc[1:]))
    with pytest.raises(ValueError):
        response_anchors(RC, [1])


def test_fixed_ladder_is_the_explicit_v3_rule():
    assert fixed_ladder_anchors([1, 5, 10]) == ([1.0, 5.0, 10.0], [-22.0, -6.0, 14.0])


def test_mapping_marks_anchors_and_interpolates_the_rest():
    pos, anc = response_anchors(RC, [1, 4, 10])
    rows = mapping_table(RC, pos, anc, [1, 2, 4, 7, 10])
    assert [r["kind"] for r in rows] == ["training_anchor", "interpolated", "training_anchor", "interpolated", "training_anchor"]
    ig = [r["input_gain_db"] for r in rows]
    assert ig == sorted(ig) and ig[0] == -20.0 and ig[-1] == 14.0
    assert position_input_gain_db(RC, pos, anc, 4) == anc[1]
