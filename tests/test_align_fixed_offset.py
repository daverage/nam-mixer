"""Fixed A/B timing correction: estimation and application are separate.

A chosen integer offset is applied verbatim by preview builders, frozen into
designs, and reused by target generation and teacher reconstruction. The fake
renders here delay Amp B by TRUE_DELAY while designs freeze a DIFFERENT
integer, so any stage that re-measured instead of reusing the frozen value
would produce TRUE_DELAY and fail.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import hybrid.core.align as align
import hybrid.core.align_diagnostic as align_diagnostic
import hybrid.modes.blend_training_target as blend_training_target
import hybrid.modes.training_target as training_target
import hybrid.training.validation as validation
from hybrid.core.align import (
    FIXED_FROZEN_OFFSET_METHOD,
    LegacyAlignmentDesignError,
    apply_fixed_offset,
    frozen_alignment_offset,
    resolve_alignment_request,
)
from hybrid.core.pipeline import build_hybrid
from hybrid.modes.design import HybridDesign, freeze_design
from hybrid.modes.fixed_blend import BlendDesign, build_fixed_blend, freeze_blend_design

SR = 48000
TRUE_DELAY = 7
FROZEN = 5  # deliberately not TRUE_DELAY


def _delay(x, n):
    return np.concatenate([np.zeros(n, dtype=np.float32), np.asarray(x, dtype=np.float32)])[: len(x)]


def _fake_render(model, audio, sample_rate, **_kwargs):
    audio = np.asarray(audio, dtype=np.float32)
    return _delay(audio, TRUE_DELAY) if "b.nam" in str(model.path) else audio.copy()


@pytest.fixture(autouse=True)
def delayed_b_render_and_no_estimation(monkeypatch):
    for module in (training_target, blend_training_target, validation):
        monkeypatch.setattr(module, "render", _fake_render)
    monkeypatch.setattr(training_target, "_md5_file", lambda path: training_target.OFFICIAL_V3_INPUT_MD5)

    def forbidden(*args, **kwargs):
        raise AssertionError("an offset was re-estimated where a frozen one must be applied")
    monkeypatch.setattr(align, "estimate_offset", forbidden)
    monkeypatch.setattr(align_diagnostic, "estimate_offset", forbidden)


def _write_nam(path):
    path.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000.0}), encoding="utf-8")
    return path


def _signal(n=SR, seed=0):
    return (0.3 * np.random.default_rng(seed).uniform(-1.0, 1.0, n)).astype(np.float32)


def _pair(n=SR):
    dry = _signal(n)
    return SimpleNamespace(
        dry=dry, profiled_dry=dry, amp_a=dry.copy(), amp_b=_delay(dry, TRUE_DELAY),
        envelope_db=np.full(n, -10.0), sample_rate=SR,
        instrument_type="guitar", input_profile_id="vintage_humbucker", input_profile_gain_db=0.0,
        calibration_mode="raw", reference_input_level_dbu=12.0, calibration_applied=False,
        amp_a_model_input_level_dbu=None, amp_b_model_input_level_dbu=None,
        amp_a_calibration_gain_db=0.0, amp_b_calibration_gain_db=0.0, calibration_warning=None,
        amp_a_input_gain_db=0.0, amp_b_input_gain_db=0.0,
    )


# --- apply_fixed_offset -------------------------------------------------------

def test_positive_offset_trims_the_start_of_b():
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(apply_fixed_offset(x, 3, 10), [3, 4, 5, 6, 7, 8, 9, 0, 0, 0])


def test_negative_offset_pads_the_start_of_b():
    x = np.arange(1, 11, dtype=np.float32)
    np.testing.assert_array_equal(apply_fixed_offset(x, -3, 10), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])


def test_zero_offset_only_length_matches():
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(apply_fixed_offset(x, 0, 10), x)
    np.testing.assert_array_equal(apply_fixed_offset(x, 0, 12), [*x, 0, 0])
    np.testing.assert_array_equal(apply_fixed_offset(x, 0, 4), x[:4])


@pytest.mark.parametrize("offset", [-40, -1, 0, 1, 7, 40])
@pytest.mark.parametrize("length", [0, 5, 100, 200])
def test_output_is_always_exactly_output_length(offset, length):
    assert len(apply_fixed_offset(_signal(100), offset, length)) == length


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_is_preserved(dtype):
    x = np.ones(50, dtype=dtype)
    for offset in (-5, 0, 5):
        assert apply_fixed_offset(x, offset, 50).dtype == dtype


def test_positive_correction_exactly_removes_a_known_delay():
    x = _signal(1000)
    corrected = apply_fixed_offset(_delay(x, TRUE_DELAY), TRUE_DELAY, len(x))
    np.testing.assert_array_equal(corrected[: -TRUE_DELAY], x[: -TRUE_DELAY])


def test_offset_must_be_an_integer():
    for bad in (1.5, 2.0, True, "3", None):
        with pytest.raises(TypeError):
            apply_fixed_offset(np.zeros(10), bad, 10)
    assert len(apply_fixed_offset(np.zeros(10), np.int64(2), 10)) == 10


def test_input_is_not_modified():
    x = _signal(100)
    before = x.copy()
    apply_fixed_offset(x, 5, 100)
    apply_fixed_offset(x, -5, 100)
    np.testing.assert_array_equal(x, before)


def test_offset_with_alignment_disabled_is_rejected():
    assert resolve_alignment_request(False, 0) == 0
    assert resolve_alignment_request(True, -3) == -3
    with pytest.raises(ValueError):
        resolve_alignment_request(False, 4)


# --- frozen design offset / legacy guard -------------------------------------

def test_frozen_offset_is_zero_when_disabled_and_verbatim_when_enabled():
    assert frozen_alignment_offset(SimpleNamespace(alignment_enabled=False, alignment_offset_samples=9)) == 0
    enabled = SimpleNamespace(alignment_enabled=True, alignment_offset_samples=-4, alignment_method=FIXED_FROZEN_OFFSET_METHOD)
    assert frozen_alignment_offset(enabled) == -4


def test_legacy_enabled_design_is_refused_not_reinterpreted():
    legacy = HybridDesign(amp_a_path="a.nam", amp_b_path="b.nam", crossover_dbfs=-20.0,
                          alignment_enabled=True, alignment_offset_samples=12)
    assert legacy.alignment_method is None
    with pytest.raises(LegacyAlignmentDesignError):
        frozen_alignment_offset(legacy)


def test_old_design_json_without_new_fields_still_loads(tmp_path):
    design = BlendDesign(amp_a_path="a.nam", amp_b_path="b.nam", mix_b=0.5)
    data = design.to_dict()
    for key in ("alignment_method", "alignment_diagnostic"):
        data.pop(key)
    path = tmp_path / "blend_design.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = BlendDesign.read_json(path)
    assert loaded.alignment_enabled is False and loaded.alignment_method is None
    assert frozen_alignment_offset(loaded) == 0


# --- preview builders ---------------------------------------------------------

@pytest.mark.parametrize("offset", [FROZEN, -FROZEN])
def test_parallel_preview_applies_exactly_the_frozen_offset(offset):
    pair = _pair()
    result = build_fixed_blend(pair, 0.5, auto_level=False, align_enabled=True, alignment_offset_samples=offset)
    expected = 0.5 * pair.amp_a + 0.5 * apply_fixed_offset(pair.amp_b, offset, len(pair.amp_a))
    np.testing.assert_array_equal(result.blend, expected)
    assert result.alignment_offset_samples == offset


@pytest.mark.parametrize("offset", [FROZEN, -FROZEN])
def test_hybrid_preview_applies_exactly_the_frozen_offset(offset):
    pair = _pair()
    corrected = build_hybrid(pair, -100.0, 2.0, auto_level=False, align_enabled=True, alignment_offset_samples=offset)
    # Crossover far below the (constant -10 dB) envelope: fully Amp B.
    np.testing.assert_allclose(corrected.hybrid, apply_fixed_offset(pair.amp_b, offset, len(pair.amp_a)), atol=1e-6)
    assert corrected.alignment_offset_samples == offset


def test_original_preview_is_unchanged():
    pair = _pair()
    original = build_fixed_blend(pair, 0.5, auto_level=False)
    np.testing.assert_array_equal(original.blend, 0.5 * pair.amp_a + 0.5 * pair.amp_b)
    assert original.alignment_offset_samples == 0
    hybrid = build_hybrid(pair, -100.0, 2.0, auto_level=False)
    np.testing.assert_allclose(hybrid.hybrid, pair.amp_b, atol=1e-6)


def test_correcting_the_true_delay_makes_b_match_a():
    pair = _pair()
    result = build_fixed_blend(pair, 1.0, auto_level=False, align_enabled=True, alignment_offset_samples=TRUE_DELAY)
    np.testing.assert_array_equal(result.blend[: -TRUE_DELAY], pair.amp_a[: -TRUE_DELAY])


# --- freezing -------------------------------------------------------------------

def test_hybrid_design_freezes_the_exact_offset_and_provenance():
    pair = _pair()
    result = build_hybrid(pair, -20.0, 8.0, auto_level=False, align_enabled=True, alignment_offset_samples=FROZEN)
    provenance = {"diagnostic_method": "multi-region-v1", "status": "fixed_offset"}
    design = freeze_design(pair, result, "a.nam", "b.nam", -20.0, 8.0, alignment_enabled=True,
                           alignment_diagnostic=provenance)
    assert (design.alignment_enabled, design.alignment_offset_samples) == (True, FROZEN)
    assert design.alignment_method == FIXED_FROZEN_OFFSET_METHOD
    assert design.alignment_diagnostic == provenance


def test_parallel_design_freezes_the_exact_offset():
    pair = _pair()
    result = build_fixed_blend(pair, 0.4, auto_level=False, align_enabled=True, alignment_offset_samples=-FROZEN)
    design = freeze_blend_design(pair, result, "a.nam", "b.nam", alignment_enabled=True)
    assert (design.alignment_enabled, design.alignment_offset_samples) == (True, -FROZEN)
    assert design.alignment_method == FIXED_FROZEN_OFFSET_METHOD


def test_unaligned_designs_record_no_method():
    pair = _pair()
    design = freeze_blend_design(pair, build_fixed_blend(pair, 0.5), "a.nam", "b.nam", alignment_enabled=False)
    assert (design.alignment_enabled, design.alignment_offset_samples, design.alignment_method) == (False, 0, None)


def test_freezing_rejects_a_choice_that_differs_from_what_was_auditioned():
    pair = _pair()
    with pytest.raises(ValueError):
        freeze_blend_design(pair, build_fixed_blend(pair, 0.5), "a.nam", "b.nam", alignment_enabled=True)


# --- target generation ----------------------------------------------------------

def _training_input(tmp_path):
    path = tmp_path / "input.wav"
    sf.write(path, _signal(SR, seed=3), SR, subtype="FLOAT")
    return path


def _blend_design(tmp_path, **overrides):
    fields = dict(amp_a_path=str(_write_nam(tmp_path / "a.nam")), amp_b_path=str(_write_nam(tmp_path / "b.nam")),
                  mix_b=0.5, calibration_mode="raw", output_gain_mode="manual", manual_output_gain_db=0.0,
                  alignment_enabled=True, alignment_offset_samples=FROZEN, alignment_method=FIXED_FROZEN_OFFSET_METHOD,
                  alignment_diagnostic={"status": "fixed_offset", "recommended_offset_samples": FROZEN})
    fields.update(overrides)
    return BlendDesign(**fields)


def _hybrid_design(tmp_path, **overrides):
    fields = dict(amp_a_path=str(_write_nam(tmp_path / "a.nam")), amp_b_path=str(_write_nam(tmp_path / "b.nam")),
                  crossover_dbfs=-200.0, transition_width_db=2.0, calibration_mode="raw",
                  output_gain_mode="manual", manual_output_gain_db=0.0,
                  alignment_enabled=True, alignment_offset_samples=FROZEN, alignment_method=FIXED_FROZEN_OFFSET_METHOD)
    fields.update(overrides)
    return HybridDesign(**fields)


def test_parallel_target_uses_the_exact_frozen_offset(tmp_path):
    design = _blend_design(tmp_path)
    bundle = blend_training_target.generate_blend_training_bundle(design, _training_input(tmp_path), tmp_path / "out")
    x, _ = sf.read(tmp_path / "out" / "input.wav", dtype="float32")
    raw, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")
    expected = 0.5 * x + 0.5 * apply_fixed_offset(_delay(x, TRUE_DELAY), FROZEN, len(x))
    np.testing.assert_allclose(raw, expected, atol=1e-7)
    correction = bundle.manifest["alignment_correction"]
    assert correction == {**correction, "applied": True, "offset_samples": FROZEN, "method": FIXED_FROZEN_OFFSET_METHOD}
    assert "phase" not in correction["description"].replace("not a phase correction", "")
    assert bundle.manifest["design"]["alignment_offset_samples"] == FROZEN
    assert bundle.manifest["alignment_diagnostic"]["recommended_offset_samples"] == FROZEN


def test_hybrid_target_uses_the_exact_frozen_offset(tmp_path):
    design = _hybrid_design(tmp_path)
    bundle = training_target.generate_training_bundle(design, _training_input(tmp_path), tmp_path / "out")
    x, _ = sf.read(tmp_path / "out" / "input.wav", dtype="float32")
    raw, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")
    np.testing.assert_allclose(raw, apply_fixed_offset(_delay(x, TRUE_DELAY), FROZEN, len(x)), atol=1e-6)
    assert bundle.manifest["alignment_correction"]["offset_samples"] == FROZEN
    assert bundle.manifest["alignment_correction"]["applied"] is True


def test_unaligned_target_records_no_correction(tmp_path):
    design = _blend_design(tmp_path, alignment_enabled=False, alignment_offset_samples=0, alignment_method=None,
                           alignment_diagnostic=None)
    bundle = blend_training_target.generate_blend_training_bundle(design, _training_input(tmp_path), tmp_path / "out")
    x, _ = sf.read(tmp_path / "out" / "input.wav", dtype="float32")
    raw, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")
    np.testing.assert_allclose(raw, 0.5 * x + 0.5 * _delay(x, TRUE_DELAY), atol=1e-7)
    assert bundle.manifest["alignment_correction"]["applied"] is False
    assert bundle.manifest["alignment_correction"]["offset_samples"] == 0


@pytest.mark.parametrize("generate,make", [
    (lambda d, i, o: training_target.generate_training_bundle(d, i, o), _hybrid_design),
    (lambda d, i, o: blend_training_target.generate_blend_training_bundle(d, i, o), _blend_design),
])
def test_generators_refuse_a_legacy_enabled_design(tmp_path, generate, make):
    with pytest.raises(LegacyAlignmentDesignError):
        generate(make(tmp_path, alignment_method=None), _training_input(tmp_path), tmp_path / "out")


# --- teacher reconstruction / validation -----------------------------------------

def test_hybrid_teacher_reconstruction_uses_the_frozen_offset(tmp_path):
    design = _hybrid_design(tmp_path)
    dry = _signal(SR, seed=4)
    result = validation.render_reference_hybrid(design, dry, SR)
    assert result.alignment_offset_samples == FROZEN
    np.testing.assert_allclose(result.hybrid, apply_fixed_offset(_delay(dry, TRUE_DELAY), FROZEN, len(dry)), atol=1e-6)


def test_parallel_teacher_reconstruction_uses_the_frozen_offset(tmp_path):
    design = _blend_design(tmp_path)
    dry = _signal(SR, seed=4)
    result = validation.render_reference_blend(design, dry, SR)
    assert result.alignment_offset_samples == FROZEN
    expected = 0.5 * dry + 0.5 * apply_fixed_offset(_delay(dry, TRUE_DELAY), FROZEN, len(dry))
    np.testing.assert_allclose(result.hybrid, expected, atol=1e-7)


def test_teacher_matches_the_generated_target_exactly(tmp_path):
    """The reconstruction used for validation and completed-model comparison
    reproduces the training target sample for sample."""
    design = _blend_design(tmp_path)
    bundle = blend_training_target.generate_blend_training_bundle(design, _training_input(tmp_path), tmp_path / "out")
    x, _ = sf.read(tmp_path / "out" / "input.wav", dtype="float32")
    raw, _ = sf.read(bundle.hybrid_target_raw_path, dtype="float32")
    teacher = validation.render_reference_blend(design, x, SR).hybrid
    np.testing.assert_allclose(teacher, raw, atol=1e-7)


def test_no_production_module_calls_an_estimator():
    """Only the measurement modules may call estimate_offset/align_to_reference."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    allowed = {root / "hybrid/core/align.py", root / "hybrid/core/align_diagnostic.py"}
    offenders = []
    for path in [*root.glob("hybrid/**/*.py"), root / "app.py", *root.glob("routes/*.py"), *root.glob("scripts/*.py")]:
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "estimate_offset" in text or "align_to_reference" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
