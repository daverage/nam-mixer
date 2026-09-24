"""Before/after regression snapshots for the three design-mode A2 bundle
generators (Hybrid, Parallel Blend, Character), so a refactor of their shared
code (output gain, embedded-cab scalar, manifest skeleton) provably changes
neither the training targets nor the manifests.

The golden file was generated from the code BEFORE the #10 consolidation.
Regenerate it only for an intentional, separately documented change:
    NAM_UPDATE_BUNDLE_SNAPSHOTS=1 python -m pytest tests/test_bundle_generator_snapshots.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import hybrid.modes.blend_training_target as blend_training_target
import hybrid.modes.character_training_target as character_training_target
import hybrid.modes.training_target as training_target
from hybrid.core.cab_ir import CabDesign, load_and_prepare_cab_ir
from hybrid.modes.character_blend import CharacterBlendDesign
from hybrid.modes.design import HybridDesign
from hybrid.modes.fixed_blend import BlendDesign

GOLDEN = Path(__file__).parent / "golden" / "bundle_generator_snapshots.json"
VOLATILE_KEYS = {"git_commit"}


@pytest.fixture(autouse=True)
def deterministic_render(monkeypatch):
    def fake_render(model, audio, sample_rate, **_kwargs):
        # A little per-model colour so Amp A and Amp B differ deterministically.
        gain = 0.9 if "b.nam" in str(model.path) else 1.1
        return np.tanh(np.asarray(audio, dtype=np.float32) * gain).astype(np.float32)
    for module in (training_target, blend_training_target, character_training_target):
        monkeypatch.setattr(module, "render", fake_render)
    monkeypatch.setattr(training_target, "_md5_file", lambda path: training_target.OFFICIAL_V3_INPUT_MD5)
    # The detected version depends on whether neural-amp-modeler is installed
    # (.venv vs .venv-a2), not on the generators.
    monkeypatch.setattr(training_target, "_detect_nam_input_version", lambda path: "<pinned by test>")


def _write_nam(path):
    path.write_text(json.dumps({"architecture": "WaveNet", "sample_rate": 48000.0, "input_level_dbu": 12.0,
                                "output_level_dbu": 0.0, "metadata": {"gear_type": "amp", "tone_type": "clean"}}), encoding="utf-8")
    return path


def _inputs(tmp_path):
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.uniform(-1.0, 1.0, 48000)).astype(np.float32)
    training_input = tmp_path / "input.wav"
    sf.write(training_input, audio, 48000, subtype="FLOAT")
    ir = tmp_path / "cab.wav"
    sf.write(ir, np.array([1.0, 0.5, -0.25, 0.125], dtype=np.float32), 48000, subtype="FLOAT")
    return _write_nam(tmp_path / "a.nam"), _write_nam(tmp_path / "b.nam"), training_input, ir


def _cab(ir, mode):
    if mode is None:
        return None
    prepared = load_and_prepare_cab_ir(ir, 48000)
    return CabDesign(selected=True, ir_working_path=str(ir), original_filename="cab.wav", display_name="Test 4x12",
                     sha256=prepared.sha256, export_mode=mode)


OUTPUT_GAIN = {"manual0": dict(output_gain_mode="manual", manual_output_gain_db=0.0),
               "manual+3": dict(output_gain_mode="manual", manual_output_gain_db=3.0),
               "auto": dict(output_gain_mode="auto")}
CASES = [(mode, gain, cab) for mode in ("hybrid", "blend", "character")
         for gain, cab in (("manual0", None), ("manual+3", None), ("auto", None), ("auto", "learned"), ("auto", "embedded"))]


def _generate(mode, gain, cab_mode, tmp_path):
    amp_a, amp_b, training_input, ir = _inputs(tmp_path)
    common = dict(amp_a_path=str(amp_a), amp_b_path=str(amp_b), calibration_mode="raw", cab=_cab(ir, cab_mode), **OUTPUT_GAIN[gain])
    out = tmp_path / "out"
    if mode == "hybrid":
        design = HybridDesign(crossover_dbfs=-20.0, transition_width_db=8.0, auto_trim_db=-1.0, manual_b_trim_db=0.5,
                              effective_b_trim_db=-0.5, instrument_type="guitar", design_reference_profile_id="p90",
                              design_reference_profile_gain_db=1.0, **common)
        return training_target.generate_training_bundle(design, training_input, out)
    if mode == "blend":
        design = BlendDesign(mix_b=0.35, auto_trim_db=-1.0, manual_b_trim_db=0.5, effective_b_trim_db=-0.5, **common)
        return blend_training_target.generate_blend_training_bundle(design, training_input, out)
    design = CharacterBlendDesign(tone_mix_b=0.6, feel_mix_b=0.4, drive_mix_b=0.5, **common)
    return character_training_target.generate_character_training_bundle(design, training_input, out)


def _normalise(value, tmp, file_hashes):
    if isinstance(value, dict):
        return {k: _normalise(v, tmp, file_hashes) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_normalise(v, tmp, file_hashes) for v in value]
    if isinstance(value, str):
        return file_hashes.get(value, value.replace(str(tmp), "<TMP>"))
    if isinstance(value, float):
        return round(value, 9)
    return value


def _audio_digest(path):
    # libsndfile stamps float WAVs with a time-dependent PEAK chunk, so the
    # file bytes differ run to run; the decoded samples and format do not.
    info = sf.info(path)
    samples, _ = sf.read(path, dtype="float32", always_2d=True)
    return f"{info.samplerate}:{info.subtype}:{samples.shape}:{hashlib.sha256(samples.tobytes()).hexdigest()}"


def _snapshot(bundle, tmp_path):
    out = tmp_path / "out"
    wavs = sorted(p for p in out.iterdir() if p.suffix == ".wav")
    # Manifest hashes of WAV files are per-run too: replace each with the
    # file's decoded-sample digest (stable, and identical for identical audio).
    file_hashes = {hashlib.sha256(p.read_bytes()).hexdigest(): f"<wav {_audio_digest(p)}>"
                   for p in sorted(tmp_path.rglob("*.wav"))}
    manifest = json.loads((out / "training_manifest.json").read_text(encoding="utf-8"))
    snapshot = {"manifest": _normalise(manifest, tmp_path, file_hashes), "files": {p.name: _audio_digest(p) for p in wavs}}
    return json.loads(json.dumps(snapshot))


@pytest.mark.parametrize("mode, gain, cab_mode", CASES, ids=[f"{m}-{g}-{c or 'nocab'}" for m, g, c in CASES])
def test_bundle_generators_match_their_snapshots(mode, gain, cab_mode, tmp_path):
    snapshot = _snapshot(_generate(mode, gain, cab_mode, tmp_path), tmp_path)
    key = f"{mode}-{gain}-{cab_mode or 'nocab'}"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.is_file() else {}
    if os.environ.get("NAM_UPDATE_BUNDLE_SNAPSHOTS") == "1":
        golden[key] = snapshot
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("snapshot written")
    assert key in golden, f"no snapshot for {key}; generate it from the pre-refactor code"
    assert snapshot == golden[key]
