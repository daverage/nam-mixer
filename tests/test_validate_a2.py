"""Unit tests for scripts/validate_a2.py -- docs/phase3.md sections 24-28.

Mocks rendering (native NAMCore isn't required for these), same pattern as
tests/test_validation.py.
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_a2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_a2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_a2"] = module
    spec.loader.exec_module(module)
    return module


validate_a2 = _load_module()


def test_parse_gain_arg_with_label():
    gain_db, label = validate_a2._parse_gain_arg("-7.0:vintage_single")
    assert gain_db == -7.0
    assert label == "vintage_single"


def test_parse_gain_arg_without_label_derives_one():
    gain_db, label = validate_a2._parse_gain_arg("4.5")
    assert gain_db == 4.5
    assert label == "4p5"


def _write_nam(path):
    path.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000.0}), encoding="utf-8")
    return path


def test_write_listening_files_applies_same_gain_to_all_and_leaves_quiet_audio_alone(tmp_path):
    sr = 48000
    reference = np.full(1000, 0.1, dtype=np.float32)
    full = np.full(1000, 0.1, dtype=np.float32)
    lite = np.full(1000, 0.1, dtype=np.float32)

    case_dir = tmp_path / "case"
    validate_a2._write_listening_files(case_dir, reference, full, lite, sr)

    with open(case_dir / "listening_gain.json") as f:
        gain_info = json.load(f)
    assert gain_info["shared_gain_db"] == 0.0  # quiet audio -> untouched

    ref_out, _ = sf.read(case_dir / "reference_hybrid.wav", dtype="float32")
    np.testing.assert_allclose(ref_out, reference, atol=1e-6)
    assert (case_dir / "sequence.wav").is_file()


def test_write_listening_files_applies_shared_attenuation_when_hot(tmp_path):
    sr = 48000
    hot = np.full(1000, 0.99, dtype=np.float32)  # ~ -0.09 dBFS, above -3 dBFS target
    quiet = np.full(1000, 0.1, dtype=np.float32)

    case_dir = tmp_path / "case"
    validate_a2._write_listening_files(case_dir, hot, quiet, None, sr)

    with open(case_dir / "listening_gain.json") as f:
        gain_info = json.load(f)
    assert gain_info["shared_gain_db"] < 0.0

    ref_out, _ = sf.read(case_dir / "reference_hybrid.wav", dtype="float32")
    quiet_out, _ = sf.read(case_dir / "a2_full.wav", dtype="float32")
    gain_ratio = 10.0 ** (gain_info["shared_gain_db"] / 20.0)
    np.testing.assert_allclose(ref_out, hot * gain_ratio, atol=1e-6)
    np.testing.assert_allclose(quiet_out, quiet * gain_ratio, atol=1e-6)
    assert not (case_dir / "a2_lite.wav").is_file()  # lite=None -> not written


def test_run_validation_writes_report_and_files(tmp_path, monkeypatch):
    amp_a = _write_nam(tmp_path / "a.nam")
    amp_b = _write_nam(tmp_path / "b.nam")
    from hybrid.design import HybridDesign
    design = HybridDesign(
        amp_a_path=str(amp_a), amp_b_path=str(amp_b),
        crossover_dbfs=-20.0, transition_width_db=8.0, effective_b_trim_db=0.0,
    )

    di_path = tmp_path / "di.wav"
    rng = np.random.default_rng(0)
    sf.write(di_path, (0.2 * rng.uniform(-1, 1, 4800)).astype(np.float32), 48000, subtype="FLOAT")

    def fake_render_reference_hybrid(design, dry, sr):
        class R:
            hybrid = np.asarray(dry, dtype=np.float32)
        return R()

    def fake_render_trained_a2(nam_path, dry, sr, slim=None):
        return np.asarray(dry, dtype=np.float32)

    monkeypatch.setattr(validate_a2, "render_reference_hybrid", fake_render_reference_hybrid)
    monkeypatch.setattr(validate_a2, "render_trained_a2", fake_render_trained_a2)

    out_dir = tmp_path / "validation"
    report = validate_a2.run_validation(design, tmp_path / "trained.nam", [di_path], [(0.0, "paf")], out_dir)

    assert (out_dir / "validation_report.json").is_file()
    assert len(report["results"]) == 1
    assert report["results"][0]["full_vs_reference"]["raw_esr"] == pytest.approx(0.0, abs=1e-9)
    assert (out_dir / "di" / "paf" / "sequence.wav").is_file()
