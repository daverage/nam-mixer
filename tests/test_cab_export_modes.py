import json

import numpy as np
import pytest

from hybrid.cab_ir import (
    CabDesign, CabIrError, EXPORT_MODE_EMBEDDED, EXPORT_MODE_LEARNED,
    EXPORT_MODE_NONE, PREPARATION_PRESERVE_ORIGINAL_TIMING,
    PREPARATION_TRIM_INITIAL_SILENCE, get_frozen_prepared_cab_ir,
    load_and_prepare_cab_ir,
)
from hybrid.sequential_nam import SequentialNamError, build_embedded_sequential, package_embedded_sequential


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
    head = {"version": "0.7.0", "architecture": "LSTM", "config": {"x": 1}, "weights": [1], "sample_rate": 48000}
    taps = np.array([1.0, -.25, .5], dtype=np.float32)
    package, record = build_embedded_sequential(head, taps, sample_rate=48000, final_scalar=.5)
    assert package["config"]["models"][0] is head
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
        build_embedded_sequential({"sample_rate": 44100}, np.ones(1), sample_rate=48000)
