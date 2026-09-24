import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.core.cab_ir import (
    CabDesign, CabIrError, EXPORT_MODE_EMBEDDED, EXPORT_MODE_LEARNED,
    EXPORT_MODE_NONE, PREPARATION_PRESERVE_ORIGINAL_TIMING,
    PREPARATION_TRIM_INITIAL_SILENCE, get_frozen_prepared_cab_ir,
    load_and_prepare_cab_ir,
)
from hybrid.training.sequential_nam import SequentialNamError, build_embedded_sequential, package_embedded_sequential, package_embedded_artifacts
from hybrid.core.render import NamRenderError, find_sequential_nam_render_exe
from hybrid.training.embedded_completion import complete_embedded_artifact


@pytest.fixture(autouse=True)
def experimental_architectures_on(monkeypatch, tmp_path):
    """These tests exercise the embedded (Sequential) package itself, which is
    only produced with experimental architectures enabled."""
    monkeypatch.setenv("NAM_MIXER_ENV_FILE", str(tmp_path / "test.env"))
    monkeypatch.setenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES", "true")


def _a2_head(sample_rate=48000):
    def wave(label):
        return {"version": "0.7.0", "architecture": "WaveNet", "config": {"label": label}, "weights": [1], "sample_rate": sample_rate}
    return {"version": "0.7.0", "architecture": "SlimmableContainer", "config": {"submodels": [
        {"max_value": .5, "model": wave("lite")}, {"max_value": 1.0, "model": wave("full")}
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
    full = head["config"]["submodels"][1]["model"]
    full["metadata"] = {"name": "Modern Boutique", "input_level_dbu": -12.0, "gain": .3548}
    taps = np.array([1.0, -.25, .5], dtype=np.float32)
    package, record = build_embedded_sequential(
        head, taps, sample_rate=48000, final_scalar=.5,
        cabinet_name="Modern Boutique 4x12", loudness=-12.3,
    )
    assert package["config"]["models"][0] is head["config"]["submodels"][1]["model"]
    assert package["config"]["models"][0]["config"]["label"] == "full"
    metadata = package["metadata"]
    assert metadata["gear_type"] == "amp_cab"
    assert metadata["modeled_by"] == "NAM Mixer"
    assert metadata["name"] == "Modern Boutique + Modern Boutique 4x12 [Embedded Cab · Full]"
    assert metadata["input_level_dbu"] == -12.0
    assert metadata["output_level_dbu"] is None
    assert metadata["loudness"] == -12.3
    assert metadata["gain"] == .3548
    assert set(metadata["date"]) == {"year", "month", "day", "hour", "minute", "second"}
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


def test_embedded_validator_uses_default_renderer_when_not_overridden(monkeypatch):
    monkeypatch.delenv("NAM_RENDER_SEQUENTIAL_EXE", raising=False)
    monkeypatch.setattr("hybrid.core.render.find_nam_render_exe", lambda: Path("/tmp/nam_render"))
    assert find_sequential_nam_render_exe() == Path("/tmp/nam_render")


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
    local_model = json.loads(Path(local["sequential_nam_path"]).read_text())
    kaggle_model = json.loads(Path(kaggle["sequential_nam_path"]).read_text())
    # Packaging timestamps legitimately differ while all audio/model content
    # must remain identical between the local and Kaggle paths.
    local_model["metadata"].pop("date")
    kaggle_model["metadata"].pop("date")
    assert local_model == kaggle_model
    assert local["linear_weights_sha256"] == kaggle["linear_weights_sha256"]


def test_embedded_completion_missing_renderer_preserves_head_and_reports_failed(monkeypatch, tmp_path):
    import soundfile as sf
    head = _a2_head(); head_path = tmp_path / "trained-a2.nam"; head_path.write_text(json.dumps(head))
    ir_path = tmp_path / "source.wav"; sf.write(ir_path, np.array([1., .25], dtype=np.float32), 48000, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(ir_path, 48000)
    manifest = {"cab": CabDesign(selected=True, ir_working_path=str(ir_path), sha256=prepared.sha256,
                                  export_mode=EXPORT_MODE_EMBEDDED).to_dict()}
    monkeypatch.delenv("NAM_RENDER_SEQUENTIAL_EXE", raising=False)
    result = complete_embedded_artifact(manifest, head_path, tmp_path, sample_rate=48000,
                                        final_scalar=1.0, validation_input=ir_path)
    assert result["state"] == "failed"
    assert head_path.is_file()


def _embedded_manifest(tmp_path):
    import soundfile as sf
    ir_path = tmp_path / "source.wav"; sf.write(ir_path, np.array([1., .25], dtype=np.float32), 48000, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(ir_path, 48000)
    return {"cab": CabDesign(selected=True, ir_working_path=str(ir_path), sha256=prepared.sha256,
                             export_mode=EXPORT_MODE_EMBEDDED).to_dict()}, ir_path


def test_embedded_completion_reports_unexpected_errors_instead_of_raising(monkeypatch, tmp_path):
    """E.g. a Sequential child without 'layers' (KeyError) must not escape into
    the trainer, which has already produced a valid head."""
    manifest, _ = _embedded_manifest(tmp_path)
    head_path = tmp_path / "trained-a2.nam"; head_path.write_text(json.dumps(_a2_head()))

    def broken(*_a, **_k):
        raise KeyError("layers")

    monkeypatch.setattr("hybrid.training.embedded_completion.package_embedded_artifacts", broken)
    result = complete_embedded_artifact(manifest, head_path, tmp_path, sample_rate=48000, final_scalar=1.0)
    assert result["state"] == "failed" and "layers" in result["error"]
    assert head_path.is_file()


def test_embedded_completion_never_validates_when_nothing_is_compared(monkeypatch, tmp_path):
    manifest, ir_path = _embedded_manifest(tmp_path)   # a 2-sample validation input
    head_path = tmp_path / "trained-a2.nam"; head_path.write_text(json.dumps(_a2_head()))
    monkeypatch.setattr("hybrid.training.embedded_completion.package_embedded_artifacts",
                        lambda *a, **k: {"sequential_nam_path": str(head_path)})
    monkeypatch.setattr("hybrid.training.embedded_completion.sequential_renderer_record", lambda: {"ok": True})
    monkeypatch.setattr("hybrid.training.embedded_completion._sequential_warmup_samples", lambda path: 1000)
    monkeypatch.setattr("hybrid.core.render.render", lambda model, audio, sr, **kw: np.asarray(audio, dtype=np.float32))
    monkeypatch.setattr("hybrid.core.render.find_sequential_nam_render_exe", lambda: Path("/tmp/seq"))
    result = complete_embedded_artifact(manifest, head_path, tmp_path, sample_rate=48000, final_scalar=1.0,
                                        validation_input=ir_path)
    assert result["state"] == "failed" and "warm-up" in result["error"]
    assert "download_available" not in result


_REPO = Path(__file__).resolve().parent.parent
_REAL_A2 = _REPO / "docs" / "history" / "Continuous Gain" / "phase4e" / "models" / "jcm800_P4E_B_s0.nam"
_REAL_IR = _REPO / "assets" / "nam_models" / "V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav"


def _sequential_renderer_available() -> bool:
    from hybrid.core.render import NamRenderError, find_sequential_nam_render_exe
    try:
        find_sequential_nam_render_exe()
        return True
    except NamRenderError:
        return False


@pytest.mark.skipif(not (_REAL_A2.is_file() and _REAL_IR.is_file() and _sequential_renderer_available()),
                    reason="needs the built native/nam_render (Sequential-capable) and the committed A2/IR fixtures")
@pytest.mark.parametrize("final_scalar", [1.0, 10 ** (-4 / 20)])
def test_real_embedded_package_with_a_long_ir_validates_inside_the_tolerance(tmp_path, final_scalar):
    """A real A2 head + the real 24001-tap V30 IR through the real Sequential
    renderer: measured max error ~3e-7, well inside the 3e-6 check (a 0.01 dB
    scalar error alone is ~7e-4)."""
    import shutil
    import soundfile as sf

    head_path = tmp_path / "trained-a2.nam"
    shutil.copyfile(_REAL_A2, head_path)
    ir_path = tmp_path / "cab.wav"
    shutil.copyfile(_REAL_IR, ir_path)
    prepared = load_and_prepare_cab_ir(ir_path, 48000)
    manifest = {"cab": CabDesign(selected=True, ir_working_path=str(ir_path), sha256=prepared.sha256,
                                  export_mode=EXPORT_MODE_EMBEDDED).to_dict()}
    dry, sr = sf.read(_REPO / "assets" / "di" / "moderate_brit.wav", dtype="float32")
    validation_input = tmp_path / "input.wav"
    sf.write(validation_input, dry[: 3 * sr], sr, subtype="FLOAT")
    result = complete_embedded_artifact(manifest, head_path, tmp_path / "out", sample_rate=48000,
                                        final_scalar=final_scalar, validation_input=validation_input)
    assert result["state"] == "validated", result.get("error")
    assert result["package_max_abs_error"] < 1e-6


def test_embedded_package_is_named_from_the_base_name_not_the_suffixed_head(tmp_path):
    """The head download is labelled '[Amp Only]'/'[Full Rig]'; the package with
    the cabinet must not inherit that suffix."""
    import soundfile as sf
    head = _a2_head()
    head["metadata"] = {"name": "Studio [Amp Only]"}
    head_path = tmp_path / "trained-a2.nam"; head_path.write_text(json.dumps(head))
    ir_path = tmp_path / "v30.wav"; sf.write(ir_path, np.array([1., .25], dtype=np.float32), 48000, subtype="FLOAT")
    prepared = load_and_prepare_cab_ir(ir_path, 48000)
    cab = CabDesign(selected=True, ir_working_path=str(ir_path), sha256=prepared.sha256, original_filename="v30.wav")
    art = package_embedded_artifacts(head_path, tmp_path / "out", cab, sample_rate=48000, final_scalar=1.0, base_name="Studio")
    metadata = json.loads(Path(art["sequential_nam_path"]).read_text())["metadata"]
    assert metadata["name"] == "Studio + v30.wav [Embedded Cab · Full]"
    assert metadata["gear_type"] == "amp_cab"
    assert json.loads(head_path.read_text())["metadata"]["name"] == "Studio [Amp Only]"   # the head download is untouched


def test_embedded_completion_is_disabled_unless_experimental_architectures_are_on(monkeypatch, tmp_path):
    import soundfile as sf
    head_path = tmp_path / "trained-a2.nam"; head_path.write_text(json.dumps(_a2_head()))
    manifest, ir_path = _embedded_manifest(tmp_path)
    monkeypatch.delenv("NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES")
    result = complete_embedded_artifact(manifest, head_path, tmp_path / "out", sample_rate=48000,
                                        final_scalar=1.0, validation_input=ir_path)
    assert result["state"] == "disabled"
    assert "artifacts" not in result and not (tmp_path / "out").exists()
    assert head_path.is_file()
