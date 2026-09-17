import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.cab_ir import (
    CabDesign, CabIrError, EXPORT_MODE_EMBEDDED, EXPORT_MODE_LEARNED,
    EXPORT_MODE_NONE, PREPARATION_PRESERVE_ORIGINAL_TIMING,
    PREPARATION_TRIM_INITIAL_SILENCE, get_frozen_prepared_cab_ir,
    load_and_prepare_cab_ir,
)
from hybrid.sequential_nam import SequentialNamError, build_embedded_sequential, package_embedded_sequential, package_embedded_artifacts
from hybrid.render import NamRenderError, find_sequential_nam_render_exe


def _a2_head(sample_rate=48000):
    def wave(label):
        return {"version": "0.7.0", "architecture": "WaveNet", "config": {"label": label}, "weights": [1], "sample_rate": sample_rate}
    return {"version": "0.7.0", "architecture": "SlimmableContainer", "config": {"submodels": [
        {"max_value": .5, "model": wave("full")}, {"max_value": 1.0, "model": wave("lite")}
    ]}, "weights": [], "sample_rate": sample_rate}


def test_legacy_baked_design_migrates_to_learned_not_embedded():
    assert CabDesign.from_dict({"selected": True, "baked": True}).export_mode == EXPORT_MODE_LEARNED
    assert CabDesign.from_dict({"selected": True, "baked": False}).export_mode == EXPORT_MODE_NONE


def test_export_modes_roundtrip_and_keep_legacy_alias():
    for mode in (EXPORT_MODE_NONE, EXPORT_MODE_LEARNED, EXPORT_MODE_EMBEDDED):
        cab = CabDesign(selected=mode != EXPORT_MODE_NONE, export_mode=mode)
        restored = CabDesign.from_dict(cab.to_dict())
        assert restored.export_mode == mode
        assert restored.baked is (mode == EXPORT_MODE_LEARNED)


def test_embedded_requires_selected_cab():
    with pytest.raises(CabIrError):
        CabDesign(export_mode=EXPORT_MODE_EMBEDDED)


def test_preparation_policy_preserves_or_trims_leading_timing(tmp_path):
    import soundfile as sf
    path = tmp_path / "cab.wav"
    sf.write(path, np.array([0, 0, 1, .5], dtype=np.float32), 48000, subtype="FLOAT")
    trimmed = load_and_prepare_cab_ir(path, 48000, preparation_mode=PREPARATION_TRIM_INITIAL_SILENCE)
    preserved = load_and_prepare_cab_ir(path, 48000, preparation_mode=PREPARATION_PRESERVE_ORIGINAL_TIMING)
    assert trimmed.leading_samples_trimmed == 2 and len(trimmed.samples) == 2
    assert preserved.leading_samples_trimmed == 0 and len(preserved.samples) == 4


def test_frozen_ir_hash_mismatch_is_rejected(tmp_path):
    import soundfile as sf
    path = tmp_path / "cab.wav"
    sf.write(path, np.array([1.0], dtype=np.float32), 48000, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(path, 48000)
    cab = CabDesign(selected=True, ir_working_path=str(path), sha256=prepared.sha256)
    sf.write(path, np.array([.5], dtype=np.float32), 48000, subtype="FLOAT")
    with pytest.raises(CabIrError, match="hash"):
        get_frozen_prepared_cab_ir(cab, 48000)


def test_canonical_embedded_sequential_preserves_complete_head_and_tap_order(tmp_path):
    head = _a2_head()
    taps = np.array([1.0, -.25, .5], dtype=np.float32)
    package, record = build_embedded_sequential(head, taps, sample_rate=48000, final_scalar=.5)
    assert package["config"]["models"][0] is head["config"]["submodels"][0]["model"]
    linear = package["config"]["models"][1]
    assert linear["config"]["receptive_field"] == 3
    assert linear["weights"] == pytest.approx([.5, -.125, .25])
    assert record["prepared_ir_tap_count"] == 3
    head_path, out_path = tmp_path / "head.nam", tmp_path / "embedded.nam"
    head_path.write_text(json.dumps(head))
    package_embedded_sequential(head_path, out_path, taps, sample_rate=48000)
    assert json.loads(out_path.read_text())["architecture"] == "Sequential"


def test_embedded_packager_rejects_sample_rate_mismatch():
    with pytest.raises(SequentialNamError, match="sample_rate"):
        build_embedded_sequential(_a2_head(44100), np.ones(1), sample_rate=48000)


@pytest.mark.parametrize("scalar", [0.0, float("nan"), float("inf")])
def test_embedded_packager_rejects_invalid_final_scalar(scalar):
    with pytest.raises(SequentialNamError, match="scalar"):
        build_embedded_sequential(_a2_head(), np.ones(1), sample_rate=48000, final_scalar=scalar)


def test_embedded_packager_rejects_non_a2_head():
    with pytest.raises(SequentialNamError, match="SlimmableContainer"):
        build_embedded_sequential({"architecture": "WaveNet", "sample_rate": 48000}, np.ones(1), sample_rate=48000)


def test_embedded_packager_rejects_overflowing_scaled_taps():
    with pytest.raises(SequentialNamError, match="non-finite"):
        build_embedded_sequential(_a2_head(), np.array([np.finfo(np.float32).max]), sample_rate=48000,
                                  final_scalar=2.0)


def test_embedded_validator_requires_explicit_sequential_renderer(monkeypatch):
    monkeypatch.delenv("NAM_RENDER_SEQUENTIAL_EXE", raising=False)
    with pytest.raises(NamRenderError, match="NAM_RENDER_SEQUENTIAL_EXE"):
        find_sequential_nam_render_exe()


def test_sequential_renderer_path_resolution_is_suffix_agnostic(monkeypatch, tmp_path):
    executable = tmp_path / "renderer with spaces"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NAM_RENDER_SEQUENTIAL_EXE", "renderer with spaces")
    assert find_sequential_nam_render_exe() == executable.resolve()


def test_sequential_renderer_rejects_non_executable_and_accepts_windows_name(monkeypatch, tmp_path):
    executable = tmp_path / "nam_render.exe"
    executable.write_text("placeholder")
    monkeypatch.setenv("NAM_RENDER_SEQUENTIAL_EXE", str(executable))
    with pytest.raises(NamRenderError, match="not an executable"):
        find_sequential_nam_render_exe()
    executable.chmod(0o755)
    assert find_sequential_nam_render_exe().name == "nam_render.exe"


def test_sequential_renderer_expands_home(monkeypatch, tmp_path):
    home = tmp_path / "home"; home.mkdir()
    executable = home / "nam_render"; executable.write_text("x"); executable.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NAM_RENDER_SEQUENTIAL_EXE", "~/nam_render")
    assert find_sequential_nam_render_exe() == executable.resolve()


def test_local_and_kaggle_completion_share_canonical_embedded_package(tmp_path):
    import soundfile as sf
    head = _a2_head(); head_path = tmp_path / "trained-a2.nam"; head_path.write_text(json.dumps(head))
    ir_path = tmp_path / "source.wav"; sf.write(ir_path, np.array([1., .25], dtype=np.float32), 48000, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(ir_path, 48000)
    cab = CabDesign(selected=True, ir_working_path=str(ir_path), sha256=prepared.sha256)
    local = package_embedded_artifacts(head_path, tmp_path / "local", cab, sample_rate=48000, final_scalar=.8)
    kaggle = package_embedded_artifacts(head_path, tmp_path / "kaggle", cab, sample_rate=48000, final_scalar=.8)
    assert json.loads(Path(local["sequential_nam_path"]).read_text()) == json.loads(Path(kaggle["sequential_nam_path"]).read_text())
    assert local["linear_weights_sha256"] == kaggle["linear_weights_sha256"]
