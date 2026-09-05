import json

import numpy as np

from hybrid.design import HybridDesign, freeze_design
from hybrid.pipeline import HybridResult, RenderedPair


def _pair():
    n = 1000
    return RenderedPair(
        dry=np.zeros(n, dtype=np.float32),
        profiled_dry=np.zeros(n, dtype=np.float32),
        amp_a=np.zeros(n, dtype=np.float32),
        amp_b=np.zeros(n, dtype=np.float32),
        envelope_db=np.full(n, -20.0),
        source_envelope_db=np.full(n, -20.0),
        sample_rate=48000,
        instrument_type="guitar",
        input_profile_id="p90",
        input_profile_gain_db=1.0,
        calibration_mode="raw",
        calibration_applied=False,
        amp_a_calibration_gain_db=0.0,
        amp_b_calibration_gain_db=0.0,
    )


def _result():
    n = 1000
    return HybridResult(
        hybrid=np.zeros(n, dtype=np.float32),
        blend_curve=np.zeros(n),
        envelope_db=np.full(n, -20.0),
        auto_trim_db=-2.5,
        manual_trim_db=0.7,
        effective_b_trim_db=-1.8,
        alignment_offset_samples=0,
        level_match=None,
    )


def test_freeze_design_captures_auditioned_values():
    design = freeze_design(
        _pair(), _result(),
        amp_a_path="assets/nam_models/JCM800.nam",
        amp_b_path="assets/nam_models/Fender.nam",
        crossover_dbfs=-22.0,
        transition_width_db=8.0,
        alignment_enabled=False,
        design_di_file="moderate_brit.wav",
    )
    assert design.auto_trim_db == -2.5
    assert design.manual_b_trim_db == 0.7
    assert design.effective_b_trim_db == -1.8
    assert design.design_reference_profile_id == "p90"
    assert design.design_reference_profile_gain_db == 1.0
    assert design.crossover_dbfs == -22.0
    assert design.transition_width_db == 8.0
    assert design.alignment_enabled is False
    assert design.envelope_max_history_ms > 0


def test_freeze_design_never_captures_test_gain_db():
    """test_gain_db (hybrid/pipeline.py's stress-test-only additional gain)
    must never leak into the frozen design -- unlike
    design_reference_profile_gain_db, it isn't an instrument identity and
    has no field on HybridDesign at all."""
    pair = _pair()
    pair.test_gain_db = 15.0  # a large, obviously-distinctive value
    design = freeze_design(
        pair, _result(),
        amp_a_path="assets/nam_models/JCM800.nam",
        amp_b_path="assets/nam_models/Fender.nam",
        crossover_dbfs=-22.0,
        transition_width_db=8.0,
        alignment_enabled=False,
    )
    assert not hasattr(design, "test_gain_db")
    assert "test_gain" not in design.to_dict()
    # And it must not have silently folded into the profile gain either.
    assert design.design_reference_profile_gain_db == pair.input_profile_gain_db


def test_hybrid_design_json_roundtrip(tmp_path):
    design = freeze_design(
        _pair(), _result(),
        amp_a_path="a.nam", amp_b_path="b.nam",
        crossover_dbfs=-22.0, transition_width_db=8.0, alignment_enabled=False,
    )
    path = tmp_path / "design.json"
    design.write_json(path)

    reloaded = HybridDesign.read_json(path)
    assert reloaded == design

    with open(path) as f:
        raw = json.load(f)
    assert raw["effective_b_trim_db"] == -1.8


def test_hybrid_design_is_frozen():
    design = HybridDesign(amp_a_path="a.nam", amp_b_path="b.nam", crossover_dbfs=-22.0)
    try:
        design.crossover_dbfs = -10.0
        assert False, "HybridDesign should be immutable"
    except Exception:
        pass
