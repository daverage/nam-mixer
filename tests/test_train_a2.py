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
    """docs/phase3.md review: the complete dependency is
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

    monkeypatch.setattr(train_a2, "assert_envelope_history_fits", fake_assert_fits)

    train_a2.check_receptive_field(manifest, 48000)

    envelope_samples = int(round(80.0 / 1000.0 * 48000))
    amp_b_rf = train_a2.compute_source_nam_receptive_field(train_a2.load_nam(amp_b))
    assert amp_b_rf > envelope_samples
    assert captured["samples"] == amp_b_rf


def test_check_receptive_field_warns_but_continues_when_amp_path_missing(tmp_path, monkeypatch):
    manifest = {"design": {"envelope_max_history_ms": 80.0}}  # no amp_a/amp_b paths
    monkeypatch.setattr(
        train_a2, "assert_envelope_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": samples + 1, "submodel_names": ["x"]})(),
    )
    train_a2.check_receptive_field(manifest, 48000)  # must not raise


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
