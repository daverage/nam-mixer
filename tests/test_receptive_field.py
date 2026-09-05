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

from hybrid.receptive_field import (
    ReceptiveFieldUnavailable,
    _layer_array_receptive_field,
    _net_receptive_field,
    assert_envelope_history_fits,
    compute_a2_receptive_field,
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
