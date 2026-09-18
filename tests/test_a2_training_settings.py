"""Parity assertions between the shared hybrid/a2_training_settings.py
constants, the local trainer (scripts/train_a2.py), and the cloud worker
(cloud/kaggle/train_a2_cloud.py) -- see docs/kaggle_training.md. This is the
mechanism that makes local/cloud training-hyperparameter drift a test
failure instead of a silent divergence.
"""
import importlib.util
import re
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


def test_template_epoch_preset_hints_match_shared_constants():
    """templates/index.html's radio-button hint text ("20 epochs, fastest"
    etc.) is a THIRD hardcoded copy of A2_EPOCH_PRESETS' actual numbers --
    app.py's /api/generate response already sends the real dict as
    `epoch_presets`, but the frontend never reads it back into these hints,
    so nothing previously caught the HTML drifting from the Python values
    it describes.
    """
    template_text = (REPO_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    for preset, epochs in A2_EPOCH_PRESETS.items():
        pattern = rf'id="a2-preset-{re.escape(preset)}"[^>]*>[^<]*<span class="hint">\(({epochs}) epochs'
        assert re.search(pattern, template_text), (
            f"templates/index.html's hint for preset {preset!r} does not say "
            f"'{epochs} epochs' -- it has drifted from A2_EPOCH_PRESETS[{preset!r}] == {epochs}"
        )


def test_requirements_training_pin_matches_shared_constant():
    """requirements-training.txt is a THIRD independent place this version
    is written (pip cannot import a Python constant) -- the local/cloud
    parity test above only compares the two Python copies to each other, so
    a bump to this file alone would silently leave local training on a
    different neural-amp-modeler release than Kaggle. Catch that here.
    """
    requirements_text = (REPO_ROOT / "requirements-training.txt").read_text(encoding="utf-8")
    match = re.search(r"^neural-amp-modeler==([0-9.]+)\s*$", requirements_text, re.MULTILINE)
    assert match, "requirements-training.txt must pin neural-amp-modeler==<version>"
    assert match.group(1) == NEURAL_AMP_MODELER_VERSION


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


def test_explicit_model_name_is_embedded_in_nam_metadata():
    # No cab selected (no "cab" key) -> the automatic "[Amp Only]" export
    # suffix applies, same as an explicit export_mode="none".
    manifest = {
        "model_name": "Mesa + JCM800 Studio Blend",
        "amp_a": {"filename": "Mesa.nam"},
        "amp_b": {"filename": "JCM800.nam"},
    }
    assert user_metadata_kwargs(manifest)["name"] == "Mesa + JCM800 Studio Blend [Amp Only]"


def test_learned_cab_export_appends_cabinet_and_suffix():
    manifest = {
        "model_name": "British American High Gain",
        "amp_a": {"filename": "Mesa.nam"},
        "amp_b": {"filename": "JCM800.nam"},
        "cab": {"export_mode": "learned", "display_name": "Modern Boutique 4x12"},
    }
    assert user_metadata_kwargs(manifest)["name"] == "British American High Gain + Modern Boutique 4x12 [Learned Cab]"


def test_embedded_cab_export_leaves_head_name_unsuffixed():
    # The SlimmableContainer head isn't the final deliverable for an
    # embedded export -- hybrid/sequential_nam.py's packaged Sequential
    # file carries its own "[Embedded Cab · Full]" suffix instead.
    manifest = {
        "model_name": "British American High Gain",
        "amp_a": {"filename": "Mesa.nam"},
        "amp_b": {"filename": "JCM800.nam"},
        "cab": {"export_mode": "embedded", "display_name": "Modern Boutique 4x12"},
    }
    assert user_metadata_kwargs(manifest)["name"] == "British American High Gain"


def test_tone_type_copied_only_when_sources_agree():
    manifest = {
        "amp_a": {"filename": "A.nam", "tone_type": "hi_gain"},
        "amp_b": {"filename": "B.nam", "tone_type": "hi_gain"},
    }
    assert user_metadata_kwargs(manifest)["tone_type"] == "hi_gain"

    manifest["amp_b"]["tone_type"] = "clean"
    assert user_metadata_kwargs(manifest)["tone_type"] is None

    manifest["amp_b"]["tone_type"] = "hi_gain"
    manifest["amp_a"]["tone_type"] = "bogus_value"
    assert user_metadata_kwargs(manifest)["tone_type"] is None


def test_cloud_worker_export_naming_matches_shared_helper():
    cloud = _load_cloud_module()
    for manifest in (
        {"model_name": "X", "amp_a": {"filename": "A.nam"}, "amp_b": {"filename": "B.nam"}},
        {"model_name": "X", "amp_a": {"filename": "A.nam"}, "amp_b": {"filename": "B.nam"},
         "cab": {"export_mode": "learned", "display_name": "Cab"}},
        {"model_name": "X", "amp_a": {"filename": "A.nam"}, "amp_b": {"filename": "B.nam"},
         "cab": {"export_mode": "embedded", "display_name": "Cab"}},
    ):
        assert cloud.user_metadata_kwargs(manifest) == user_metadata_kwargs(manifest)


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

    class FakeToneType:
        pass

    class FakeUserMetadata:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "nam", types.ModuleType("nam"))
    monkeypatch.setitem(sys.modules, "nam.train", types.ModuleType("nam.train"))
    monkeypatch.setitem(sys.modules, "nam.train.core", types.SimpleNamespace(train=fake_train))
    monkeypatch.setitem(sys.modules, "nam.train.metadata", types.SimpleNamespace(TRAINING_KEY="training"))
    monkeypatch.setitem(sys.modules, "nam.models", types.ModuleType("nam.models"))
    monkeypatch.setitem(sys.modules, "nam.models.metadata", types.SimpleNamespace(
        GearType=FakeGearType, ToneType=FakeToneType, UserMetadata=FakeUserMetadata,
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
