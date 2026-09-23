"""Local/cloud parity for the Continuous Gain bundle (custom training input, mode "continuous_gain").

The FC models were trained with a data-config patch (the input is the official file PLUS DI segments, with an explicit
train/validation boundary). Both A2 trainers -- scripts/train_a2.py and the self-contained
cloud/kaggle/train_a2_cloud.py -- must apply the SAME patch, read the boundary the same way, name the export the
same way and reach the same receptive-field conclusion (tests/test_receptive_field_parity.py is the model).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from hybrid.training.a2_training_settings import custom_split_train_stop, user_metadata_kwargs

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def local_mod():
    return _load("train_a2_local_for_cg_parity", REPO_ROOT / "scripts" / "train_a2.py")


@pytest.fixture(scope="module")
def cloud_mod():
    return _load("train_a2_cloud_for_cg_parity", REPO_ROOT / "cloud" / "kaggle" / "train_a2_cloud.py")


CG_MANIFEST = {"mode": "continuous_gain", "model_name": "My Amp", "training_input": {"custom_split": True, "train_stop_samples": 1234, "sample_rate": 48000},
               "receptive_field": {"branch_samples": {"G1": 3000, "G10": 6332, "envelope": 3837}, "cab": {"baked": False}}, "cab": {"baked": False, "export_mode": "none"}}


def test_custom_split_boundary_is_read_identically(cloud_mod):
    assert custom_split_train_stop(CG_MANIFEST) == 1234 == cloud_mod.custom_split_train_stop(CG_MANIFEST)
    for m in ({}, {"training_input": {}}, {"training_input": {"sha256": "x"}}):
        assert custom_split_train_stop(m) is None and cloud_mod.custom_split_train_stop(m) is None
    bad = {"training_input": {"custom_split": True, "train_stop_samples": 0}}
    with pytest.raises(ValueError):
        custom_split_train_stop(bad)
    with pytest.raises(ValueError):
        cloud_mod.custom_split_train_stop(bad)


def _stub_nam(monkeypatch):
    md = types.ModuleType("nam.train.metadata")
    md.DataChecks = lambda **k: ("DataChecks", k)
    md.Latency = lambda **k: types.SimpleNamespace(**k)
    md.LatencyCalibration = lambda **k: ("LatencyCalibration", k)
    md.LatencyCalibrationWarnings = lambda **k: ("Warnings", k)
    train = types.ModuleType("nam.train"); train.metadata = md
    nam = types.ModuleType("nam"); nam.train = train
    for name, mod in (("nam", nam), ("nam.train", train), ("nam.train.metadata", md)):
        monkeypatch.setitem(sys.modules, name, mod)
    core = types.SimpleNamespace(_Version=lambda *a: a)
    return core


def test_both_trainers_install_the_same_data_config(local_mod, cloud_mod, monkeypatch):
    cores = []
    for apply in (local_mod._apply_custom_split_patch, cloud_mod.apply_custom_split_patch):
        core = _stub_nam(monkeypatch)
        apply(core, 1234)
        cores.append(core)
    a, b = cores
    assert a._get_data_config("v", "in.wav", "out.wav", 8192, 0) == b._get_data_config("v", "in.wav", "out.wav", 8192, 0)
    cfg = a._get_data_config("v", "in.wav", "out.wav", 8192, 0)
    assert cfg["train"] == {"ny": 8192, "stop_samples": 1234} and cfg["validation"] == {"ny": None, "start_samples": 1234}
    assert cfg["common"]["allow_unequal_lengths"] is True and cfg["joint"] == []
    for core in cores:
        assert core._detect_input_version("x") == ((4, 0, 0), False)
        assert core._check_data("a", b=1) == ("DataChecks", {"version": 1, "passed": True})
        assert core._analyze_latency(0, "x").manual == 0
        assert core._get_final_latency(core._analyze_latency(7)) == 7


def test_export_name_is_the_users_model_name_without_a_hybrid_suffix(cloud_mod):
    assert user_metadata_kwargs(CG_MANIFEST)["name"] == "My Amp"
    assert cloud_mod.user_metadata_kwargs(CG_MANIFEST)["name"] == "My Amp"
    unnamed = {"mode": "continuous_gain", "cab": {"baked": False}}
    assert user_metadata_kwargs(unnamed)["name"] == cloud_mod.user_metadata_kwargs(unnamed)["name"] == "Continuous Gain"


def test_receptive_field_policy_matches_for_continuous_gain(local_mod, cloud_mod, monkeypatch):
    class Rf:
        receptive_field_samples, submodel_names = 6332, ["full"]
    monkeypatch.setattr(local_mod, "assert_required_history_fits", lambda hard, sr, margin_fraction=0.0: Rf())
    local = local_mod.check_receptive_field(CG_MANIFEST, 48000)
    cloud = cloud_mod.check_receptive_field(CG_MANIFEST, 48000) if _cloud_can_run(cloud_mod) else None
    assert local["hard_required_samples"] == 6332 and local["mode"] == "continuous_gain"
    assert set(local["branch_samples"]) == {"G1", "G10", "envelope"}
    if cloud is not None:
        assert cloud["hard_required_samples"] == local["hard_required_samples"]


def _cloud_can_run(cloud_mod) -> bool:
    try:
        import nam.train  # noqa: F401
        return True
    except Exception:
        return False


def test_core_dependency_beyond_the_a2_receptive_field_still_aborts(local_mod, monkeypatch):
    def refuse(hard, sr, margin_fraction=0.0):
        raise ValueError("too long")
    monkeypatch.setattr(local_mod, "assert_required_history_fits", refuse)
    with pytest.raises(local_mod.TrainingAbort):
        local_mod.check_receptive_field(CG_MANIFEST, 48000)


def test_cloud_input_check_uses_the_recorded_sha_for_a_custom_input(cloud_mod, tmp_path):
    import numpy as np
    import soundfile as sf
    sr = 48000
    x = (np.random.default_rng(0).standard_normal(sr) * 0.1).astype("float32")
    sf.write(tmp_path / "input.wav", x, sr, subtype="FLOAT"); sf.write(tmp_path / "hybrid_target.wav", x * 0.5, sr, subtype="FLOAT")
    man = {"training_input": {"custom_split": True, "train_stop_samples": 100, "sha256": cloud_mod._sha256_file(tmp_path / "input.wav")}}
    (tmp_path / "training_manifest.json").write_text(json.dumps(man))
    assert cloud_mod.validate_inputs(tmp_path)["frame_count"] == sr
    man["training_input"]["sha256"] = "0" * 64
    (tmp_path / "training_manifest.json").write_text(json.dumps(man))
    with pytest.raises(cloud_mod.CloudTrainingError):
        cloud_mod.validate_inputs(tmp_path)
    (tmp_path / "training_manifest.json").write_text(json.dumps({}))      # no custom split -> official MD5 required
    with pytest.raises(cloud_mod.CloudTrainingError, match="official NAM V3"):
        cloud_mod.validate_inputs(tmp_path)
