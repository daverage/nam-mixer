"""Local/cloud parity for the receptive-field policy: CORE dependency is a
HARD gate, a baked cabinet's FORMAL total is advisory-only -- see
docs/blend-mode.md's cabinet-approximation-policy section. Both
`scripts/train_a2.py::check_receptive_field` (imports hybrid/, recomputes
Amp A/B RF from the actual .nam files) and
`cloud/kaggle/train_a2_cloud.py::check_receptive_field` (self-contained,
reads only the manifest) MUST implement the same policy even though they
source their branch samples differently -- this is exactly the failure mode
that caused a real Kaggle training run to be refused for a cab-only formal
overflow while this repo's intent was to allow it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def local_mod():
    return _load_module("train_a2_local_for_parity", REPO_ROOT / "scripts" / "train_a2.py")


@pytest.fixture(scope="module")
def cloud_mod():
    return _load_module("train_a2_cloud_for_parity", REPO_ROOT / "cloud" / "kaggle" / "train_a2_cloud.py")


def _write_nam_with_config(path, kernel_sizes, dilations):
    raw = {
        "architecture": "WaveNet",
        "sample_rate": 48000.0,
        "config": {"layers": [{"kernel_sizes": kernel_sizes, "dilations": dilations}]},
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _matching_manifests(tmp_path, mode="blend", cab_baked=False, cab_fir_history=0):
    """Build a local-style manifest (amp .nam file paths for
    scripts/train_a2.py to recompute RF from) and an equivalent cloud-style
    manifest (only the manifest's own recorded branch_samples, no files) that
    describe the SAME core dependency, so both check_receptive_field
    implementations should reach the same conclusion."""
    amp_a = _write_nam_with_config(tmp_path / "a.nam", [3], [1])  # RF = 3
    amp_b = _write_nam_with_config(tmp_path / "b.nam", [3], [1])  # RF = 3

    cab = {"baked": cab_baked, "fir_history_samples": cab_fir_history} if cab_baked else {"baked": False}

    local_manifest = {
        "mode": mode,
        "amp_a": {"path": str(amp_a)},
        "amp_b": {"path": str(amp_b)},
        "cab": cab,
    }
    cloud_manifest = {
        "mode": mode,
        "cab": cab,
        "receptive_field": {
            "mode": mode,
            "branch_samples": {"amp_a": 3, "amp_b": 3},
        },
    }
    return local_manifest, cloud_manifest


def test_both_permit_baked_cab_formal_overflow_when_core_fits(tmp_path, local_mod, cloud_mod, monkeypatch):
    """The exact scenario from the bug report: Amp A/Amp B each need 6332
    samples (here scaled down to 3 for a fast synthetic test), core fits the
    fake A2 exactly, but a baked cab's formal total does not -- NEITHER
    worker should abort."""
    local_manifest, cloud_manifest = _matching_manifests(tmp_path, cab_baked=True, cab_fir_history=500)

    monkeypatch.setattr(
        local_mod, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": 3, "submodel_names": ["fake"]})(),
    )
    local_result = local_mod.check_receptive_field(local_manifest, 48000)

    monkeypatch.setattr(cloud_mod, "_a2_receptive_field_for_test", None, raising=False)

    def fake_load_packed_config():
        return {"net": {"config": {"submodels": [{"name": "fake", "config": {"layers": [{"kernel_sizes": [3], "dilations": [1]}]}}]}}}

    import importlib.resources as importlib_resources

    class _FakeResource:
        def read_text(self, encoding="utf-8"):
            return json.dumps(fake_load_packed_config())

    monkeypatch.setattr(
        importlib_resources, "files",
        lambda pkg: type("F", (), {"joinpath": lambda self, name: _FakeResource()})(),
    )

    cloud_result = cloud_mod.check_receptive_field(cloud_manifest, 48000)

    for label, result in (("local", local_result), ("cloud", cloud_result)):
        assert result["hard_required_samples"] == 3, label
        assert result["cab_requires_approximation"] is True, label
        assert result["formal_total_required_samples"] == 3 + 500, label


def test_both_reject_core_overflow_regardless_of_cab(tmp_path, local_mod, cloud_mod, monkeypatch):
    """A core dependency that itself exceeds A2's receptive field must abort
    on BOTH workers, cab or no cab -- semantics must match even though the
    exception types differ (TrainingAbort vs CloudTrainingError)."""
    local_manifest, cloud_manifest = _matching_manifests(tmp_path, cab_baked=False)

    monkeypatch.setattr(
        local_mod, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: (_ for _ in ()).throw(ValueError("required 3 exceeds available 1")),
    )
    with pytest.raises(local_mod.TrainingAbort):
        local_mod.check_receptive_field(local_manifest, 48000)

    class _FakeResource:
        def read_text(self, encoding="utf-8"):
            # A2 RF of 1 sample -- smaller than the 3-sample core dependency.
            return json.dumps({"net": {"config": {"submodels": [
                {"name": "fake", "config": {"layers": [{"kernel_sizes": [1], "dilations": [1]}]}}
            ]}}})

    import importlib.resources as importlib_resources

    monkeypatch.setattr(
        importlib_resources, "files",
        lambda pkg: type("F", (), {"joinpath": lambda self, name: _FakeResource()})(),
    )

    with pytest.raises(cloud_mod.CloudTrainingError):
        cloud_mod.check_receptive_field(cloud_manifest, 48000)


def test_both_report_no_approximation_when_formal_total_also_fits(tmp_path, local_mod, cloud_mod, monkeypatch):
    local_manifest, cloud_manifest = _matching_manifests(tmp_path, cab_baked=True, cab_fir_history=2)

    monkeypatch.setattr(
        local_mod, "assert_required_history_fits",
        lambda samples, sr, margin_fraction=0.0: type("R", (), {"receptive_field_samples": 1000, "submodel_names": ["fake"]})(),
    )
    local_result = local_mod.check_receptive_field(local_manifest, 48000)

    class _FakeResource:
        def read_text(self, encoding="utf-8"):
            return json.dumps({"net": {"config": {"submodels": [
                {"name": "fake", "config": {"layers": [{"kernel_sizes": [1000], "dilations": [1]}]}}
            ]}}})

    import importlib.resources as importlib_resources

    monkeypatch.setattr(
        importlib_resources, "files",
        lambda pkg: type("F", (), {"joinpath": lambda self, name: _FakeResource()})(),
    )
    cloud_result = cloud_mod.check_receptive_field(cloud_manifest, 48000)

    for label, result in (("local", local_result), ("cloud", cloud_result)):
        assert result["cab_requires_approximation"] is False, label
