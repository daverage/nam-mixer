"""Parity assertions between the shared hybrid/a2_training_settings.py
constants, the local trainer (scripts/train_a2.py), and the cloud worker
(cloud/kaggle/train_a2_cloud.py) -- see docs/kaggle_training.md. This is the
mechanism that makes local/cloud training-hyperparameter drift a test
failure instead of a silent divergence.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from hybrid.a2_training_settings import (
    A2_EPOCH_PRESETS,
    A2_QUICK_SETTINGS,
    A2_TRAINING_SETTINGS,
    DEFAULT_EPOCH_PRESET,
    NEURAL_AMP_MODELER_VERSION,
    OFFICIAL_V3_INPUT_MD5,
    settings_for,
    settings_for_preset,
    user_metadata_kwargs,
)
from hybrid.training_target import OFFICIAL_V3_INPUT_MD5 as TRAINING_TARGET_MD5

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_cloud_module():
    path = REPO_ROOT / "cloud" / "kaggle" / "train_a2_cloud.py"
    spec = importlib.util.spec_from_file_location("train_a2_cloud", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_epoch_presets_match_brief():
    assert A2_EPOCH_PRESETS == {"draft": 20, "standard": 60, "high_def": 120}
    assert DEFAULT_EPOCH_PRESET == "standard"


def test_settings_for_preset_only_epochs_differ():
    draft = settings_for_preset("draft")
    standard = settings_for_preset("standard")
    high_def = settings_for_preset("high_def")

    assert draft.epochs == 20
    assert standard.epochs == 60
    assert high_def.epochs == 120

    for settings in (draft, standard, high_def):
        assert settings.batch_size == 16
        assert settings.ny == 8192
        assert settings.seed == 0
        assert settings.latency == 0
        assert settings.ignore_checks is False
        assert settings.fast_dev_run is False
        assert settings.silent is True


def test_settings_for_preset_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown A2 epoch preset"):
        settings_for_preset("ultra")


def test_full_settings_match_brief():
    """A2_TRAINING_SETTINGS is settings_for_preset(DEFAULT_EPOCH_PRESET) --
    kept for callers that don't need preset selection."""
    assert A2_TRAINING_SETTINGS == settings_for_preset(DEFAULT_EPOCH_PRESET)
    assert A2_TRAINING_SETTINGS.epochs == 60
    assert A2_TRAINING_SETTINGS.batch_size == 16
    assert A2_TRAINING_SETTINGS.ny == 8192
    assert A2_TRAINING_SETTINGS.seed == 0
    assert A2_TRAINING_SETTINGS.latency == 0
    assert A2_TRAINING_SETTINGS.ignore_checks is False
    assert A2_TRAINING_SETTINGS.fast_dev_run is False
    assert A2_TRAINING_SETTINGS.silent is True


def test_quick_settings_are_smoke_test_only():
    assert A2_QUICK_SETTINGS.epochs == 1
    assert A2_QUICK_SETTINGS.fast_dev_run is True
    # Everything else matches the normal run -- only epochs/fast_dev_run change.
    assert A2_QUICK_SETTINGS.batch_size == A2_TRAINING_SETTINGS.batch_size
    assert A2_QUICK_SETTINGS.ny == A2_TRAINING_SETTINGS.ny
    assert A2_QUICK_SETTINGS.seed == A2_TRAINING_SETTINGS.seed
    assert A2_QUICK_SETTINGS.latency == A2_TRAINING_SETTINGS.latency


def test_settings_for():
    assert settings_for(quick=False) is A2_TRAINING_SETTINGS
    assert settings_for(quick=True) is A2_QUICK_SETTINGS


def test_official_v3_md5_matches_training_target_module():
    assert OFFICIAL_V3_INPUT_MD5 == TRAINING_TARGET_MD5


def test_cloud_worker_settings_match_shared_constants():
    cloud = _load_cloud_module()
    assert cloud.TRAINING_SETTINGS["epochs"] == A2_TRAINING_SETTINGS.epochs
    assert cloud.TRAINING_SETTINGS["batch_size"] == A2_TRAINING_SETTINGS.batch_size
    assert cloud.TRAINING_SETTINGS["ny"] == A2_TRAINING_SETTINGS.ny
    assert cloud.TRAINING_SETTINGS["seed"] == A2_TRAINING_SETTINGS.seed
    assert cloud.TRAINING_SETTINGS["latency"] == A2_TRAINING_SETTINGS.latency
    assert cloud.TRAINING_SETTINGS["ignore_checks"] == A2_TRAINING_SETTINGS.ignore_checks
    assert cloud.TRAINING_SETTINGS["fast_dev_run"] == A2_TRAINING_SETTINGS.fast_dev_run

    assert cloud.QUICK_SETTINGS["epochs"] == A2_QUICK_SETTINGS.epochs
    assert cloud.QUICK_SETTINGS["fast_dev_run"] == A2_QUICK_SETTINGS.fast_dev_run

    assert cloud.NEURAL_AMP_MODELER_VERSION == NEURAL_AMP_MODELER_VERSION
    assert cloud.OFFICIAL_V3_INPUT_MD5 == OFFICIAL_V3_INPUT_MD5


def test_cloud_worker_epoch_presets_match_shared_constants():
    cloud = _load_cloud_module()
    assert cloud.EPOCH_PRESETS == A2_EPOCH_PRESETS
    assert cloud.DEFAULT_EPOCH_PRESET == DEFAULT_EPOCH_PRESET
    for preset in A2_EPOCH_PRESETS:
        cloud_settings = cloud.settings_for_preset(preset)
        local_settings = settings_for_preset(preset)
        assert cloud_settings["epochs"] == local_settings.epochs
        assert cloud_settings["batch_size"] == local_settings.batch_size
        assert cloud_settings["ny"] == local_settings.ny
        assert cloud_settings["seed"] == local_settings.seed
        assert cloud_settings["latency"] == local_settings.latency


def test_cloud_worker_user_metadata_matches_shared_helper():
    cloud = _load_cloud_module()
    manifest = {
        "amp_a": {"filename": "JCM800.nam"},
        "amp_b": {"filename": "FenderSuperReverb1977_Clean.nam"},
        "calibration": {"applied": True, "reference_input_level_dbu": 12.0},
    }
    assert cloud.user_metadata_kwargs(manifest) == user_metadata_kwargs(manifest)

    manifest_raw = {**manifest, "calibration": {"applied": False}}
    kwargs = user_metadata_kwargs(manifest_raw)
    assert kwargs["input_level_dbu"] is None
    assert cloud.user_metadata_kwargs(manifest_raw) == kwargs


@pytest.mark.parametrize("preset,expected_epochs", [("draft", 20), ("standard", 60), ("high_def", 120)])
def test_cloud_worker_run_training_uses_requested_epoch_preset(tmp_path, monkeypatch, preset, expected_epochs):
    """Direct test of the cloud worker's run_training(): whichever preset is
    requested, its epoch count must reach nam.train.core.train() unchanged."""
    import json
    import types

    import numpy as np
    import soundfile as sf

    cloud = _load_cloud_module()

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    sf.write(bundle_dir / "input.wav", np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
    sf.write(bundle_dir / "hybrid_target.wav", np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
    (bundle_dir / "training_manifest.json").write_text(json.dumps({
        "amp_a": {"filename": "A.nam"}, "amp_b": {"filename": "B.nam"}, "calibration": {"applied": False},
    }))

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

    train_result = cloud.run_training(bundle_dir, tmp_path / "out", quick=False, epoch_preset=preset)

    assert captured["epochs"] == expected_epochs
    assert train_result["settings"]["epochs"] == expected_epochs
    assert train_result["nam_path"].is_file()


def test_cloud_worker_run_training_rejects_unknown_epoch_preset(tmp_path):
    cloud = _load_cloud_module()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "training_manifest.json").write_text("{}")
    with pytest.raises(cloud.CloudTrainingError, match="unknown A2 epoch preset"):
        cloud.run_training(bundle_dir, tmp_path / "out", quick=False, epoch_preset="ultra")
