"""Parity assertions between the shared hybrid/a2_training_settings.py
constants, the local trainer (scripts/train_a2.py), and the cloud worker
(cloud/kaggle/train_a2_cloud.py) -- see docs/kaggle_training.md. This is the
mechanism that makes local/cloud training-hyperparameter drift a test
failure instead of a silent divergence.
"""
import importlib.util
import sys
from pathlib import Path

from hybrid.a2_training_settings import (
    A2_QUICK_SETTINGS,
    A2_TRAINING_SETTINGS,
    NEURAL_AMP_MODELER_VERSION,
    OFFICIAL_V3_INPUT_MD5,
    settings_for,
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


def test_full_settings_match_brief():
    assert A2_TRAINING_SETTINGS.epochs == 100
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
