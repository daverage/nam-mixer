"""Unit tests for scripts/train_a2.py's orchestration logic -- mocks the
official trainer entry point so these run without Torch/neural-amp-modeler
installed (see docs/phase3.md section 37: "mock the official trainer for
normal unit tests... have a separate integration/manual test path for real
A2 training").
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_a2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_a2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_a2"] = module
    spec.loader.exec_module(module)
    return module


train_a2 = _load_module()


def _write_nam(path, input_level_dbu=None):
    raw = {"architecture": "WaveNet", "sample_rate": 48000.0}
    if input_level_dbu is not None:
        raw["input_level_dbu"] = input_level_dbu
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _make_bundle(tmp_path, n=4800, sample_rate=48000):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    rng = np.random.default_rng(0)
    input_audio = (0.3 * rng.uniform(-1, 1, n)).astype(np.float32)
    target_audio = (0.2 * rng.uniform(-1, 1, n)).astype(np.float32)
    sf.write(bundle_dir / "input.wav", input_audio, sample_rate, subtype="FLOAT")
    sf.write(bundle_dir / "hybrid_target.wav", target_audio, sample_rate, subtype="FLOAT")

    manifest = {
        "training_input": {
            "sample_rate": sample_rate,
            "sha256": train_a2._sha256_file(bundle_dir / "input.wav"),
        },
        "target": {
            "final_sha256": train_a2._sha256_file(bundle_dir / "hybrid_target.wav"),
        },
        "design": {"envelope_max_history_ms": 80.0},
    }
    manifest_path = bundle_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path, bundle_dir


def test_load_and_validate_manifest_ok(tmp_path):
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    manifest = train_a2.load_and_validate_manifest(manifest_path)
    assert manifest["_bundle_dir"] == str(bundle_dir)


def test_load_and_validate_manifest_rejects_hash_mismatch(tmp_path):
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["training_input"]["sha256"] = "0" * 64
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(train_a2.TrainingAbort):
        train_a2.load_and_validate_manifest(manifest_path)


def test_load_and_validate_manifest_rejects_frame_count_mismatch(tmp_path):
    manifest_path, bundle_dir = _make_bundle(tmp_path, n=4800)
    # Overwrite target with a different length, then patch its hash to match
    # so only the frame-count check fires.
    target_path = bundle_dir / "hybrid_target.wav"
    sf.write(target_path, np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["target"]["final_sha256"] = train_a2._sha256_file(target_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    with pytest.raises(train_a2.TrainingAbort, match="frame count mismatch"):
        train_a2.load_and_validate_manifest(manifest_path)


def test_load_and_validate_manifest_missing_file_aborts(tmp_path):
    with pytest.raises(train_a2.TrainingAbort):
        train_a2.load_and_validate_manifest(tmp_path / "nope.json")


def test_check_receptive_field_aborts_without_training_env(tmp_path):
    """No training environment is installed here -- the receptive-field
    check must abort loudly rather than silently skip it."""
    try:
        import nam  # noqa: F401
        pytest.skip("neural-amp-modeler installed; unavailable-path not exercised here")
    except ImportError:
        pass
    manifest = {"design": {"envelope_max_history_ms": 80.0}}
    with pytest.raises(train_a2.TrainingAbort):
        train_a2.check_receptive_field(manifest, 48000)


def _write_nam_with_config(path, kernel_sizes, dilations):
    raw = {
        "architecture": "WaveNet",
        "sample_rate": 48000.0,
        "config": {"layers": [{"kernel_sizes": kernel_sizes, "dilations": dilations}]},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_check_receptive_field_uses_max_across_envelope_and_amp_branches(tmp_path, monkeypatch):
    """docs/phase3.md review: the complete CORE dependency is
    max(envelope, Amp A, Amp B) since the branches run in parallel -- prove
    check_receptive_field actually computes that max and passes IT (not just
    the envelope) to the A2 fits-check."""
    # Envelope: 80ms @ 48kHz = 3840 samples. Amp A: small RF. Amp B: RF
    # larger than the envelope -- must become the effective max.
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # RF = 1 + 2*1 = 3 samples
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [6, 6], [100, 200])  # RF = 1 + 5*100 + 5*200 = 1501 samples... too small
    # Make Amp B's RF clearly exceed the 3840-sample envelope history.
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [6] * 5, [1, 100, 500, 1000, 2000])

    manifest = {
        "design": {"envelope_max_history_ms": 80.0},
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
    }

    captured = {}

    def fake_assert_fits(samples, sample_rate, margin_fraction=0.0):
        captured["samples"] = samples
        class FakeRF:
            receptive_field_samples = samples + 1000
            submodel_names = ["fake"]
        return FakeRF()

    monkeypatch.setattr(train_a2, "assert_required_history_fits", fake_assert_fits)

    result = train_a2.check_receptive_field(manifest, 48000)

    envelope_samples = int(round(80.0 / 1000.0 * 48000))
    amp_b_rf = train_a2.compute_source_nam_receptive_field(train_a2.load_nam(amp_b))
    assert amp_b_rf > envelope_samples
    assert captured["samples"] == amp_b_rf
    assert result["hard_required_samples"] == amp_b_rf


def test_check_receptive_field_warns_but_continues_when_amp_path_missing(tmp_path, monkeypatch):
    manifest = {"design": {"envelope_max_history_ms": 80.0}}  # no amp_a/amp_b paths
    monkeypatch.setattr(
        train_a2, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": samples + 1, "submodel_names": ["x"]})(),
    )
    train_a2.check_receptive_field(manifest, 48000)  # must not raise


def test_check_receptive_field_blend_mode_ignores_envelope_and_uses_amp_max(tmp_path, monkeypatch):
    """docs/blend-mode.md: Blend mode's CORE dependency is max(Amp A, Amp B)
    only -- no envelope branch at all, even if envelope_max_history_ms were
    (incorrectly) present in the manifest."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # small RF
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [6] * 3, [1, 10, 100])  # larger RF

    manifest = {
        "mode": "blend",
        "design": {"envelope_max_history_ms": 999999.0},  # must be ignored for Blend
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
    }

    captured = {}

    def fake_assert_fits(samples, sample_rate, margin_fraction=0.0):
        captured["samples"] = samples
        return type("R", (), {"receptive_field_samples": samples + 1000, "submodel_names": ["fake"]})()

    monkeypatch.setattr(train_a2, "assert_required_history_fits", fake_assert_fits)
    train_a2.check_receptive_field(manifest, 48000)

    amp_b_rf = train_a2.compute_source_nam_receptive_field(train_a2.load_nam(amp_b))
    assert captured["samples"] == amp_b_rf  # NOT inflated by the bogus envelope_max_history_ms


def test_check_receptive_field_baked_cab_formal_total_is_core_plus_history(tmp_path, monkeypatch):
    """The formal total (advisory only) is still calculated as core + (L-1),
    but it must NOT be what's passed to the hard fits-check -- only the
    CORE dependency (3 samples here) is."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # RF = 3
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])  # RF = 3

    manifest = {
        "mode": "blend",
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
        "cab": {"baked": True, "fir_history_samples": 500},
    }

    captured = {}

    def fake_assert_fits(samples, sample_rate, margin_fraction=0.0):
        captured["samples"] = samples
        return type("R", (), {"receptive_field_samples": samples + 1000, "submodel_names": ["fake"]})()

    monkeypatch.setattr(train_a2, "assert_required_history_fits", fake_assert_fits)
    result = train_a2.check_receptive_field(manifest, 48000)

    assert captured["samples"] == 3  # HARD check only ever sees the core dependency
    assert result["hard_required_samples"] == 3
    assert result["cab_fir_history_samples"] == 500
    assert result["formal_total_required_samples"] == 3 + 500  # advisory total, reported not gated
    assert result["cab_requires_approximation"] is False  # A2 RF in this fake is 1003, formal total 503 fits


def test_check_receptive_field_preview_only_cab_adds_no_approximation_warning(tmp_path, monkeypatch):
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])

    manifest = {
        "mode": "blend",
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
        "cab": {"baked": False, "fir_history_samples": 500},  # preview-only
    }

    captured = {}

    def fake_assert_fits(samples, sample_rate, margin_fraction=0.0):
        captured["samples"] = samples
        return type("R", (), {"receptive_field_samples": samples + 1000, "submodel_names": ["fake"]})()

    monkeypatch.setattr(train_a2, "assert_required_history_fits", fake_assert_fits)
    result = train_a2.check_receptive_field(manifest, 48000)
    assert captured["samples"] == 3  # cab not baked -- zero added, not part of the hard check either
    assert result["cab_baked"] is False
    assert result["cab_requires_approximation"] is False
    assert result["formal_total_required_samples"] == 3


def test_check_receptive_field_baked_cab_over_capacity_does_not_abort(tmp_path, monkeypatch):
    """The core acceptance criterion of this policy: a baked cab whose
    formal total exceeds the A2 receptive field must NOT abort training --
    only the CORE dependency exceeding it does (see the sibling test
    test_check_receptive_field_core_over_capacity_aborts)."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # RF = 3
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])  # RF = 3

    manifest = {
        "mode": "blend",
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
        "cab": {"baked": True, "fir_history_samples": 5000},
    }

    # Core (3 samples) fits comfortably inside a small fake A2 RF (100), but
    # the formal total (3 + 5000 = 5003) does not -- this must NOT raise.
    monkeypatch.setattr(
        train_a2, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": 100, "submodel_names": ["fake"]})(),
    )

    result = train_a2.check_receptive_field(manifest, 48000)  # must NOT raise
    assert result["hard_required_samples"] == 3
    assert result["formal_total_required_samples"] == 3 + 5000
    assert result["cab_requires_approximation"] is True


def test_check_receptive_field_core_over_capacity_aborts(tmp_path, monkeypatch):
    """Unlike a baked cab's formal overflow, the CORE dependency exceeding
    A2's receptive field must still abort exactly as before this policy."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])

    manifest = {
        "mode": "blend",
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
    }

    def fake_assert_fits(samples, sample_rate, margin_fraction=0.0):
        raise ValueError(f"required {samples} exceeds available 1")
    monkeypatch.setattr(train_a2, "assert_required_history_fits", fake_assert_fits)

    with pytest.raises(train_a2.TrainingAbort, match="REFUSING"):
        train_a2.check_receptive_field(manifest, 48000)


def test_check_receptive_field_exact_fit_boundary_still_permitted(tmp_path, monkeypatch):
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # RF = 3
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])  # RF = 3
    manifest = {"mode": "blend", "amp_a": {"path": str(amp_a)}, "amp_b": {"path": str(amp_b)}}

    monkeypatch.setattr(
        train_a2, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": samples, "submodel_names": ["x"]})(),
    )
    result = train_a2.check_receptive_field(manifest, 48000)  # exact fit -- must not raise
    assert result["core_status"] == "EXACT FIT"


def test_check_receptive_field_real_world_bug_report_scenario_trains(tmp_path, monkeypatch):
    """Acceptance criterion #1 (docs/blend-mode.md): a Blend whose Amp A and
    Amp B each require 6332 samples (the real installed A2's own receptive
    field, per hybrid/receptive_field.py's module docstring) can still train
    with a baked 500ms/24000-sample cabinet IR -- the exact real-world
    scenario that used to be incorrectly refused."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [6332], [1])  # RF = 1 + 6331*1 = 6332
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [6332], [1])  # RF = 6332

    manifest = {
        "mode": "blend",
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
        "cab": {"baked": True, "fir_history_samples": 24000},
    }

    monkeypatch.setattr(
        train_a2, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": 6332, "submodel_names": ["channels_8"]})(),
    )

    result = train_a2.check_receptive_field(manifest, 48000)  # must NOT raise/abort

    assert result["hard_required_samples"] == 6332
    assert result["a2_receptive_field_samples"] == 6332
    assert result["core_status"] == "EXACT FIT"
    assert result["formal_total_required_samples"] == 6332 + 24000
    assert result["cab_requires_approximation"] is True


def test_check_receptive_field_legacy_manifest_base_is_hard_total_is_formal(tmp_path, monkeypatch):
    """A manifest written by the PRE-policy code has
    receptive_field.base_required_samples/cab_fir_serial_samples/
    total_required_samples but no receptive_field.cab sub-record. The hard
    gate must still only ever see the CORE (base) dependency, never the
    (formerly-hard, now-formal-only) total -- see docs/blend-mode.md
    "BACKWARDS COMPATIBILITY"."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # RF = 3
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])  # RF = 3

    manifest = {
        "mode": "blend",
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
        "cab": {"baked": True},  # no fir_history_samples on the CabDesign itself
        "receptive_field": {
            "mode": "blend",
            "branch_samples": {"amp_a": 3, "amp_b": 3},
            "base_required_samples": 3,
            "cab_fir_serial_samples": 500,
            "total_required_samples": 503,
        },
    }

    captured = {}

    def fake_assert_fits(samples, sample_rate, margin_fraction=0.0):
        captured["samples"] = samples
        return type("R", (), {"receptive_field_samples": samples + 1000, "submodel_names": ["fake"]})()

    monkeypatch.setattr(train_a2, "assert_required_history_fits", fake_assert_fits)
    result = train_a2.check_receptive_field(manifest, 48000)

    assert captured["samples"] == 3  # hard gate sees the CORE amp dependency only
    assert result["cab_fir_history_samples"] == 500  # resolved from the legacy cab_fir_serial_samples key
    assert result["formal_total_required_samples"] == 503


def test_validate_exported_nam_runs_native_render(tmp_path, monkeypatch):
    nam_path = _write_nam(tmp_path / "model.nam")
    input_path = tmp_path / "input.wav"
    audio = np.zeros(1000, dtype=np.float32)
    sf.write(input_path, audio, 48000, subtype="FLOAT")

    def fake_render(model, a, sr, **kwargs):
        return np.asarray(a, dtype=np.float32).copy()
    monkeypatch.setattr(train_a2, "render", fake_render)

    result = train_a2.validate_exported_nam(nam_path, input_path, 48000)
    assert result["frame_count"] == 1000
    assert np.all(np.isfinite(result["rendered"]))


def test_validate_exported_nam_rejects_non_finite(tmp_path, monkeypatch):
    nam_path = _write_nam(tmp_path / "model.nam")
    input_path = tmp_path / "input.wav"
    sf.write(input_path, np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")

    def bad_render(model, a, sr, **kwargs):
        out = np.asarray(a, dtype=np.float32).copy()
        out[0] = np.nan
        return out
    monkeypatch.setattr(train_a2, "render", bad_render)

    with pytest.raises(train_a2.TrainingAbort):
        train_a2.validate_exported_nam(nam_path, input_path, 48000)


def test_check_full_low_level_response_delegates_to_shared_helper_and_prints_verdict(tmp_path, monkeypatch, capsys):
    """The substantive sweep/comparison logic (docs/blend-mode-fixes.md,
    Phases 10-11) lives in hybrid.character_training_target.check_full_low_
    level_response, shared with hybrid.kaggle_training.validate_downloaded_
    model -- see tests/test_character_training_target.py for that logic.
    This only checks train_a2.py's wrapper delegates and reports a verdict."""
    manifest = {"mode": "character", "low_level_response": {}}
    nam_path, input_path = tmp_path / "model.nam", tmp_path / "input.wav"

    canned = {"max_error_db": 3.0, "pass": True, "dead_zone_detected": False}
    monkeypatch.setattr(train_a2.character_training_target, "check_full_low_level_response", lambda *a, **k: canned)

    result = train_a2.check_full_low_level_response(manifest, nam_path, input_path, 48000)
    assert result is canned
    assert "PASS" in capsys.readouterr().out

    monkeypatch.setattr(train_a2.character_training_target, "check_full_low_level_response", lambda *a, **k: None)
    assert train_a2.check_full_low_level_response(manifest, nam_path, input_path, 48000) is None


def test_compare_to_target_metrics(tmp_path):
    target_path = tmp_path / "target.wav"
    target_audio = np.full(1000, 0.5, dtype=np.float32)
    sf.write(target_path, target_audio, 48000, subtype="FLOAT")

    identical = np.full(1000, 0.5, dtype=np.float32)
    metrics = train_a2.compare_to_target(identical, target_path)
    assert metrics["raw_esr"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["peak_difference"] == pytest.approx(0.0, abs=1e-6)


def test_official_trainer_api_contract_matches_what_run_official_trainer_assumes():
    """Guards against neural-amp-modeler API drift: if a future installed
    version renames/removes `nam.train.core.train` or `BaseNet.export`,
    this fails loudly here instead of scripts/train_a2.py silently calling
    the wrong thing during a real (slow) training run."""
    try:
        import nam.train.core as core
        from nam.models.base import BaseNet
    except ImportError:
        pytest.skip("requires the neural-amp-modeler training package")

    import inspect
    train_params = set(inspect.signature(core.train).parameters)
    for required in ("input_path", "output_path", "train_path", "epochs", "latency", "fast_dev_run"):
        assert required in train_params, f"nam.train.core.train() no longer has a '{required}' parameter"

    export_params = set(inspect.signature(BaseNet.export).parameters)
    for required in ("outdir", "basename"):
        assert required in export_params, f"BaseNet.export() no longer has a '{required}' parameter"


def test_main_end_to_end_with_mocked_trainer(tmp_path, monkeypatch):
    """Full orchestration with the official-trainer call and receptive-field
    check both mocked out -- proves main() wires load -> train -> validate ->
    compare -> manifest-update together correctly without needing Torch."""
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    exported_nam = _write_nam(tmp_path / "trained.nam")

    monkeypatch.setattr(train_a2, "check_receptive_field", lambda manifest, sr: None)
    monkeypatch.setattr(train_a2, "_run_official_trainer", lambda *a, **k: exported_nam)

    def fake_render(model, a, sr, **kwargs):
        return np.asarray(a, dtype=np.float32).copy()
    monkeypatch.setattr(train_a2, "render", fake_render)

    rc = train_a2.main([str(manifest_path)])
    assert rc == 0

    with open(manifest_path) as f:
        updated = json.load(f)
    assert updated["training"]["output_nam_path"] == str(exported_nam)
    assert "full_metrics_vs_target" in updated["training"]
    assert "_bundle_dir" not in updated


# --- --epoch-preset (draft=20 / standard=60 / high_def=120) --------------

def test_main_records_default_epoch_preset_in_manifest(tmp_path, monkeypatch):
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    exported_nam = _write_nam(tmp_path / "trained.nam")

    monkeypatch.setattr(train_a2, "check_receptive_field", lambda manifest, sr: None)
    monkeypatch.setattr(train_a2, "_run_official_trainer", lambda *a, **k: exported_nam)
    monkeypatch.setattr(train_a2, "render", lambda model, a, sr, **kwargs: np.asarray(a, dtype=np.float32).copy())

    rc = train_a2.main([str(manifest_path)])
    assert rc == 0

    with open(manifest_path) as f:
        updated = json.load(f)
    assert updated["training"]["epoch_preset"] == "standard"
    assert updated["training"]["epochs"] == 60


@pytest.mark.parametrize("preset,expected_epochs", [("draft", 20), ("standard", 60), ("high_def", 120)])
def test_main_epoch_preset_flag_selects_settings_passed_to_trainer(tmp_path, monkeypatch, preset, expected_epochs):
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    exported_nam = _write_nam(tmp_path / "trained.nam")
    captured = {}

    def fake_run_official_trainer(input_path, target_path, output_dir, settings, device, manifest):
        captured["settings"] = settings
        return exported_nam

    monkeypatch.setattr(train_a2, "check_receptive_field", lambda manifest, sr: None)
    monkeypatch.setattr(train_a2, "_run_official_trainer", fake_run_official_trainer)
    monkeypatch.setattr(train_a2, "render", lambda model, a, sr, **kwargs: np.asarray(a, dtype=np.float32).copy())

    rc = train_a2.main([str(manifest_path), "--epoch-preset", preset])
    assert rc == 0
    assert captured["settings"].epochs == expected_epochs
    assert captured["settings"].fast_dev_run is False

    with open(manifest_path) as f:
        updated = json.load(f)
    assert updated["training"]["epoch_preset"] == preset
    assert updated["training"]["epochs"] == expected_epochs


def test_main_quick_flag_overrides_epoch_preset(tmp_path, monkeypatch):
    """--quick is a distinct 1-epoch smoke test, not one of the quality
    presets -- passing both must still run the smoke test, never a preset."""
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    exported_nam = _write_nam(tmp_path / "trained.nam")
    captured = {}

    def fake_run_official_trainer(input_path, target_path, output_dir, settings, device, manifest):
        captured["settings"] = settings
        return exported_nam

    monkeypatch.setattr(train_a2, "check_receptive_field", lambda manifest, sr: None)
    monkeypatch.setattr(train_a2, "_run_official_trainer", fake_run_official_trainer)
    monkeypatch.setattr(train_a2, "render", lambda model, a, sr, **kwargs: np.asarray(a, dtype=np.float32).copy())

    rc = train_a2.main([str(manifest_path), "--quick", "--epoch-preset", "high_def"])
    assert rc == 0
    assert captured["settings"].epochs == 1
    assert captured["settings"].fast_dev_run is True

    with open(manifest_path) as f:
        updated = json.load(f)
    assert updated["training"]["quick_mode"] is True
    assert updated["training"]["epoch_preset"] is None  # not a preset run


def test_epoch_preset_flag_rejects_unknown_value(tmp_path):
    manifest_path, bundle_dir = _make_bundle(tmp_path)
    with pytest.raises(SystemExit):
        train_a2.main([str(manifest_path), "--epoch-preset", "ultra"])


def test_run_official_trainer_forwards_settings_epochs_to_core_train(tmp_path, monkeypatch):
    """Direct test of _run_official_trainer itself: whatever A2TrainingSettings
    it's given, `epochs` must reach nam.train.core.train() unchanged --
    proves the draft/standard/high_def epoch count actually controls the
    real trainer call, not just what's recorded in the manifest."""
    import types

    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    output_dir = tmp_path / "out"
    sf.write(input_path, np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
    sf.write(target_path, np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")

    captured = {}

    class FakeNet:
        def export(self, export_dir, basename, user_metadata, other_metadata):
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            (Path(export_dir) / f"{basename}.nam").write_text("{}", encoding="utf-8")

    class FakeModel:
        net = FakeNet()

    class FakeMetadata:
        def model_dump(self):
            return {}

    class FakeTrainOutput:
        model = FakeModel()
        metadata = FakeMetadata()

    def fake_train(**kwargs):
        captured.update(kwargs)
        return FakeTrainOutput()

    class FakeGearType:
        AMP = "amp"

    class FakeUserMetadata:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "nam", types.ModuleType("nam"))
    monkeypatch.setitem(sys.modules, "nam.train", types.ModuleType("nam.train"))
    monkeypatch.setitem(sys.modules, "nam.train.core", types.SimpleNamespace(train=fake_train))
    monkeypatch.setitem(sys.modules, "nam.train.metadata", types.SimpleNamespace(TRAINING_KEY="training"))
    monkeypatch.setitem(sys.modules, "nam.models", types.ModuleType("nam.models"))
    monkeypatch.setitem(sys.modules, "nam.models.metadata", types.SimpleNamespace(
        GearType=FakeGearType, UserMetadata=FakeUserMetadata,
    ))

    settings = train_a2.settings_for_preset("high_def")
    manifest = {"amp_a": {"filename": "A.nam"}, "amp_b": {"filename": "B.nam"}, "calibration": {"applied": False}}

    nam_path = train_a2._run_official_trainer(input_path, target_path, output_dir, settings, "auto", manifest)

    assert captured["epochs"] == 120
    assert nam_path.is_file()
