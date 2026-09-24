"""FC training bundle: teacher construction, alignment, one peak-ceiling scale, files and manifest."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from hybrid.continuous_gain.bundle import (FC_RECIPE, FcRecipe, build_training_audio, bundle_manifest_core, make_chain, sha256_f32, write_bundle)
from hybrid.core.safety import apply_peak_ceiling
from tests.cg_synth import SR, synth_di

TINY = FcRecipe(train_dis=("a", "b"), val_dis=("v",), train_offsets_db=(-6.0, 6.0), val_offsets_db=(0.0,), di_seconds=1, val_seconds=1)
POS, ANCH = [1.0, 2.0, 5.0], [-20.0, -6.0, 14.0]


def _di(name):
    return synth_di(name, 2.0)


def _official():
    return synth_di("official", 2.0)


def _renderers(delays=None):
    delays = delays or {}
    def render(g, x):
        y = np.tanh((1 + g) * x.astype(np.float64)).astype(np.float32)
        d = delays.get(g, 0)
        return np.concatenate([np.zeros(d, np.float32), y[:len(y) - d]]) if d else y
    return render


def test_chain_levels_follow_the_designated_gain():
    chain = make_chain(POS, ANCH)
    assert chain.levels_db == (-50.0, -36.0, -16.0) and chain.reference_db == -30.0
    assert [chain.designated_gain_db(k) for k in range(3)] == ANCH


def test_target_layout_split_and_single_ceiling_scale():
    chain = make_chain(POS, ANCH)
    b = build_training_audio(chain, _renderers(), {}, _official(), _di, TINY)
    seg = b.segments
    assert seg[0]["source"] == "official" and [s["split"] for s in seg] == ["train"] * 5 + ["val"]
    assert all(s["stop"] == seg[i + 1]["start"] for i, s in enumerate(seg[:-1])) and seg[-1]["stop"] == len(b.input) == len(b.target)
    assert b.train_stop == seg[4]["stop"] == seg[5]["start"]
    assert np.max(np.abs(b.target)) <= 10 ** (-0.2 / 20) + 1e-6           # one fixed peak ceiling, never above it
    assert b.output_scale_c == pytest.approx(10 ** (-b.reduction_db / 20) if b.reduction_db else 1.0)


def test_ceiling_scale_is_a_single_constant_not_a_limiter():
    chain = make_chain(POS, ANCH)
    hot = lambda g, x: (x * 6).astype(np.float32)             # far above the ceiling
    b = build_training_audio(chain, hot, {}, _official(), _di, TINY)
    assert b.reduction_db > 0 and b.output_scale_c < 1.0
    raw = build_training_audio(chain, hot, {}, _official(), _di, FcRecipe(**{**TINY.__dict__, "ceiling_dbfs": 40.0}))
    assert raw.reduction_db == 0.0
    ratio = b.target.astype(np.float64)[SR: SR + 5000] / raw.target.astype(np.float64)[SR: SR + 5000]
    assert np.allclose(ratio[np.isfinite(ratio)], b.output_scale_c, rtol=1e-4)     # the same gain on every sample


def test_verified_alignment_shift_is_applied_to_that_capture_only():
    chain = make_chain(POS, ANCH)
    base = build_training_audio(chain, _renderers(), {}, _official(), _di, TINY)
    delayed = build_training_audio(chain, _renderers({2.0: 25}), {2.0: 25}, _official(), _di, TINY)
    assert np.array_equal(base.target, delayed.target)
    unaligned = build_training_audio(chain, _renderers({2.0: 25}), {}, _official(), _di, TINY)
    assert not np.array_equal(base.target, unaligned.target)


def test_the_default_recipe_is_the_frozen_fc_recipe():
    assert FC_RECIPE.train_offsets_db == (-32.0, -24.0, -16.0, -8.0, 0.0, 8.0, 14.0, 20.0) and FC_RECIPE.val_offsets_db == (-24.0, 0.0, 8.0, 20.0)
    assert FC_RECIPE.train_dis == ("clean_smooth", "moderate_hotrod", "high_thrash") and FC_RECIPE.val_dis == ("high_metalcore",)
    assert (FC_RECIPE.di_seconds, FC_RECIPE.val_seconds, FC_RECIPE.reference_db, FC_RECIPE.ceiling_dbfs) == (20, 8, -30.0, -0.2)


def test_written_bundle_is_a_standard_a2_bundle_with_a_declared_custom_split(tmp_path):
    chain = make_chain(POS, ANCH)
    b = build_training_audio(chain, _renderers(), {}, _official(), _di, TINY)
    mp = write_bundle(tmp_path / "d", b, chain, ANCH, {}, sources=[], model_name="M", artifact_stem="M", design={"kind": "x"}, receptive_field={"branch_samples": {"G1": 10}})
    m = json.loads(mp.read_text())
    d = mp.parent
    assert m["mode"] == "continuous_gain" and m["artifact_filename"] == "M.nam" and m["cab"]["baked"] is False
    ti = m["training_input"]
    assert ti["custom_split"] and ti["train_stop_samples"] == b.train_stop and ti["sample_rate"] == SR
    import hashlib
    assert ti["sha256"] == hashlib.sha256((d / "input.wav").read_bytes()).hexdigest()
    assert m["target"]["final_sha256"] == hashlib.sha256((d / "hybrid_target.wav").read_bytes()).hexdigest()
    x, sr = sf.read(d / "input.wav", dtype="float32"); y, _ = sf.read(d / "hybrid_target.wav", dtype="float32")
    assert sr == SR and len(x) == len(y) and np.array_equal(y, b.target)
    core = m["core"]
    assert core["target_audio_sha256"] == sha256_f32(b.target) and core["gains"] == POS and core["anchors_designated_gain_db"] == ANCH
    assert core == bundle_manifest_core(b, chain, ANCH, {})


def test_apply_peak_ceiling_semantics_the_bundle_relies_on():
    y, red = apply_peak_ceiling(np.array([0.0, 2.0, -1.0]), -0.2)
    assert red > 0 and np.max(np.abs(y)) == pytest.approx(10 ** (-0.2 / 20))


def test_manifest_records_the_ceiling_the_recipe_actually_applied(tmp_path):
    chain = make_chain(POS, ANCH)
    custom = FcRecipe(**{**TINY.__dict__, "ceiling_dbfs": -3.0})
    b = build_training_audio(chain, _renderers(), {}, _official(), _di, custom)
    mp = write_bundle(tmp_path / "d", b, chain, ANCH, {}, sources=[], model_name="M", artifact_stem="M",
                      design={"kind": "x"}, receptive_field={"branch_samples": {"G1": 10}})
    assert json.loads(mp.read_text())["target"]["ceiling_dbfs"] == -3.0
    assert np.max(np.abs(b.target)) <= 10 ** (-3.0 / 20) + 1e-6


def test_source_records_carry_each_captures_gear_type(tmp_path):
    """So a Continuous Gain export of full-rig captures is labelled [Full Rig]/amp_cab."""
    from hybrid.continuous_gain.bundle import source_records

    paths = {}
    for position, gear in ((1.0, "amp_cab"), (5.0, "amp_cab")):
        path = tmp_path / f"g{position:g}.nam"
        path.write_text(json.dumps({"architecture": "WaveNet", "config": {}, "weights": [], "sample_rate": 48000,
                                    "metadata": {"gear_type": gear}}))
        paths[position] = path
    audit = {"captures": {"1": {"status": "VALID", "correction": None}, "5": {"status": "VALID", "correction": None}}}
    recs = source_records([1.0, 5.0], paths, [-10.0, 10.0], make_chain([1.0, 5.0], [-10.0, 10.0]), audit)
    assert [r["gear_type"] for r in recs] == ["amp_cab", "amp_cab"]


def test_source_records_use_already_loaded_captures_instead_of_reloading(monkeypatch, tmp_path):
    import types

    import hybrid.continuous_gain.bundle as bundle_mod
    from hybrid.continuous_gain.bundle import make_chain, source_records

    def no_reload(path):
        raise AssertionError(f"reloaded {path}")
    monkeypatch.setattr(bundle_mod, "load_nam", no_reload)
    paths = {}
    for p in (1.0, 5.0):
        paths[p] = tmp_path / f"G{p:g}.nam"; paths[p].write_bytes(b"{}")
    audit = {"captures": {f"{p:g}": {"status": "verified", "correction": None, "evidence": {}} for p in paths}}
    models = {1.0: types.SimpleNamespace(gear_type="amp"), 5.0: types.SimpleNamespace(gear_type="amp_cab")}
    recs = source_records([1.0, 5.0], paths, [-10.0, 10.0], make_chain([1.0, 5.0], [-10.0, 10.0]), audit, models)
    assert [r["gear_type"] for r in recs] == ["amp", "amp_cab"]
