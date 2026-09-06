"""Tests for hybrid/receptive_field.py -- docs/phase3.md section 5.

The pure math (receptive-field formula) is tested directly against a
constructed config dict, independent of whether neural-amp-modeler is
actually installed. The "real installed config" path is exercised only when
the training environment is present (skipped otherwise, matching
docs/phase3.md section 37's "have a separate integration/manual test path for
real A2 training" for anything that needs the actual package).
"""
from __future__ import annotations

import pytest

import hybrid.receptive_field as receptive_field
from hybrid.receptive_field import (
    A2ReceptiveField,
    ReceptiveFieldUnavailable,
    _layer_array_receptive_field,
    _model_receptive_field,
    _net_receptive_field,
    assert_envelope_history_fits,
    cab_fir_serial_history_samples,
    combine_required_history,
    compute_a2_receptive_field,
    compute_source_nam_receptive_field,
)

try:
    import nam  # noqa: F401
    _NAM_INSTALLED = True
except ImportError:
    _NAM_INSTALLED = False


def test_layer_array_receptive_field_single_dilation():
    cfg = {"kernel_sizes": [3], "dilations": [1]}
    assert _layer_array_receptive_field(cfg) == 1 + (3 - 1) * 1


def test_layer_array_receptive_field_dilated_stack():
    cfg = {"kernel_sizes": [3, 3, 3, 3], "dilations": [1, 2, 4, 8]}
    expected = 1 + sum((3 - 1) * d for d in [1, 2, 4, 8])
    assert _layer_array_receptive_field(cfg) == expected


def test_layer_array_receptive_field_per_layer_kernel_sizes():
    """The real installed 0.13.x packed config uses a PER-LAYER kernel_sizes
    list (not one shared kernel size) -- see hybrid/receptive_field.py's
    module docstring."""
    cfg = {"kernel_sizes": [6, 15, 6], "dilations": [1, 3, 7]}
    expected = 1 + (6 - 1) * 1 + (15 - 1) * 3 + (6 - 1) * 7
    assert _layer_array_receptive_field(cfg) == expected


def test_net_receptive_field_stacks_layer_arrays_in_series():
    net_cfg = {
        "layers_configs": [
            {"kernel_sizes": [3, 3], "dilations": [1, 2]},
            {"kernel_sizes": [3, 3], "dilations": [1, 2]},
        ]
    }
    single = _layer_array_receptive_field(net_cfg["layers_configs"][0])
    # Two identical stages in series: total = single + (single - 1)
    assert _net_receptive_field(net_cfg) == single + (single - 1)


def test_receptive_field_unavailable_without_training_env():
    if _NAM_INSTALLED:
        pytest.skip("neural-amp-modeler is installed in this environment; unavailable-path not exercised")
    with pytest.raises(ReceptiveFieldUnavailable):
        compute_a2_receptive_field()
    with pytest.raises(ReceptiveFieldUnavailable):
        assert_envelope_history_fits(4000, 48000)


@pytest.mark.skipif(not _NAM_INSTALLED, reason="requires the neural-amp-modeler training package")
def test_real_a2_receptive_field_fits_bounded_envelope_history():
    from hybrid.envelope import bounded_envelope_max_history_samples
    rf = compute_a2_receptive_field()
    history = bounded_envelope_max_history_samples(48000)
    assert history < rf.receptive_field_samples


def test_model_receptive_field_plain_wavenet():
    model = {"architecture": "WaveNet", "config": {"layers": [{"kernel_sizes": [3, 3], "dilations": [1, 2]}]}}
    expected = _layer_array_receptive_field(model["config"]["layers"][0])
    assert _model_receptive_field(model) == expected


def test_model_receptive_field_slimmable_container_takes_max_across_submodels():
    """Real captured .nam schema (verified against assets/nam_models/*.nam):
    config.submodels[i] = {"max_value": ..., "model": {"architecture": "WaveNet", "config": {"layers": [...]}}}."""
    small = {"kernel_sizes": [3], "dilations": [1]}
    big = {"kernel_sizes": [6, 6], "dilations": [1, 3]}
    model = {
        "architecture": "SlimmableContainer",
        "config": {
            "submodels": [
                {"max_value": 0.5, "model": {"architecture": "WaveNet", "config": {"layers": [small]}}},
                {"max_value": 1.0, "model": {"architecture": "WaveNet", "config": {"layers": [big]}}},
            ]
        },
    }
    assert _model_receptive_field(model) == _layer_array_receptive_field(big)


def test_model_receptive_field_unknown_architecture_raises():
    with pytest.raises(ReceptiveFieldUnavailable):
        _model_receptive_field({"architecture": "SomeFutureThing", "config": {}})


def test_compute_source_nam_receptive_field_accepts_nam_model_object():
    class _FakeNamModel:
        raw = {"architecture": "WaveNet", "config": {"layers": [{"kernel_sizes": [3], "dilations": [1]}]}}

    assert compute_source_nam_receptive_field(_FakeNamModel()) == 1 + (3 - 1) * 1


@pytest.mark.skipif(not _NAM_INSTALLED, reason="requires the neural-amp-modeler training package")
def test_real_source_nam_captures_receptive_field(tmp_path):
    """Sanity check against the actual bundled amp captures, if present --
    proves compute_source_nam_receptive_field parses the REAL on-disk schema,
    not just a hand-constructed test dict."""
    from pathlib import Path

    from hybrid.nam_loader import load_nam

    candidates = list(Path("assets/nam_models").glob("*.nam"))
    if not candidates:
        pytest.skip("no real .nam captures available in assets/nam_models/")
    model = load_nam(candidates[0])
    rf = compute_source_nam_receptive_field(model)
    assert rf > 0


def _fake_a2_rf(samples: int) -> A2ReceptiveField:
    return A2ReceptiveField(receptive_field_samples=samples, submodel_names=["fake"], config_path="<fake>", raw_config={})


def test_assert_envelope_history_fits_permits_exact_fit(monkeypatch):
    """docs/phase3.md review: Amp A/B/envelope run in PARALLEL and the final
    blend is memoryless, so required == available is a legitimate exact fit,
    not a failure -- only required > available should raise."""
    monkeypatch.setattr(receptive_field, "compute_a2_receptive_field", lambda: _fake_a2_rf(1000))
    rf = assert_envelope_history_fits(1000, 48000)  # required == available
    assert rf.receptive_field_samples == 1000


def test_assert_envelope_history_fits_rejects_when_required_exceeds_available(monkeypatch):
    monkeypatch.setattr(receptive_field, "compute_a2_receptive_field", lambda: _fake_a2_rf(1000))
    with pytest.raises(ValueError):
        assert_envelope_history_fits(1001, 48000)


def test_assert_envelope_history_fits_accepts_comfortable_margin(monkeypatch):
    monkeypatch.setattr(receptive_field, "compute_a2_receptive_field", lambda: _fake_a2_rf(1000))
    rf = assert_envelope_history_fits(500, 48000)
    assert rf.receptive_field_samples == 1000


# -- docs/blend-mode.md "RECEPTIVE FIELD" -----------------------------------

def test_cab_fir_serial_history_samples_is_length_minus_one():
    assert cab_fir_serial_history_samples(1) == 0
    assert cab_fir_serial_history_samples(500) == 499
    assert cab_fir_serial_history_samples(0) == 0


def test_combine_required_history_hybrid_takes_max_of_three_parallel_branches():
    record = combine_required_history("hybrid", amp_a_samples=100, amp_b_samples=300, envelope_samples=200)
    assert record["base_required_samples"] == 300
    assert record["total_required_samples"] == 300
    assert record["branch_samples"] == {"amp_a": 100, "amp_b": 300, "envelope": 200}


def test_combine_required_history_blend_ignores_envelope_branch():
    record = combine_required_history("blend", amp_a_samples=100, amp_b_samples=300, envelope_samples=99999)
    assert "envelope" not in record["branch_samples"]
    assert record["base_required_samples"] == 300
    assert record["total_required_samples"] == 300


def test_combine_required_history_blend_requires_no_envelope_argument():
    record = combine_required_history("blend", amp_a_samples=100, amp_b_samples=50, envelope_samples=None)
    assert record["base_required_samples"] == 100


def test_combine_required_history_hybrid_requires_envelope_samples():
    with pytest.raises(ValueError):
        combine_required_history("hybrid", amp_a_samples=100, amp_b_samples=50, envelope_samples=None)


def test_combine_required_history_adds_baked_cab_serially_not_as_parallel_max():
    record = combine_required_history("blend", amp_a_samples=100, amp_b_samples=300, envelope_samples=None, cab_fir_samples=50)
    assert record["base_required_samples"] == 300
    assert record["cab_fir_serial_samples"] == 50
    assert record["total_required_samples"] == 350


def test_combine_required_history_rejects_unknown_mode():
    with pytest.raises(ValueError):
        combine_required_history("bogus", amp_a_samples=1, amp_b_samples=1, envelope_samples=None)
