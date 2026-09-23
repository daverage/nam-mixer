"""NAM input calibration -- matching what the official NAM plugin does when
two different `.nam` captures were trained/calibrated against different
analogue input levels.

Feeding identical raw digital samples into two differently-calibrated models
does not necessarily represent feeding the same physical guitar voltage into
both amps. If a model records its calibration input level (`input_level_dbu`
in the `.nam` JSON), the official plugin compensates for it with:

    model_input_adjustment_db = reference_input_level_dbu - model_input_level_dbu

This module implements that formula and the Auto/Raw mode selection logic
described in docs/INPUT_PROFILE_RESEARCH.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# The official NAM plugin's default virtual/reference input calibration.
# This is NOT a pickup output value -- it's the analogue level assumed to
# correspond to digital full scale.
DEFAULT_REFERENCE_INPUT_LEVEL_DBU = 12.0

_UNAVAILABLE_WARNING = (
    "Input calibration unavailable for this pair: one or both source NAMs do "
    "not contain input_level_dbu. Using raw digital level for both models."
)


def input_calibration_gain_db(reference_input_level_dbu: float, model_input_level_dbu: float) -> float:
    """The official NAM plugin's per-model input compensation formula."""
    return reference_input_level_dbu - model_input_level_dbu


@dataclass
class CalibrationResult:
    mode: str  # "auto" or "raw"
    applied: bool
    reference_input_level_dbu: float
    amp_a_model_input_level_dbu: Optional[float]
    amp_b_model_input_level_dbu: Optional[float]
    amp_a_gain_db: float
    amp_b_gain_db: float
    warning: Optional[str]


def resolve_calibration(
    mode: str,
    reference_input_level_dbu: float,
    amp_a_input_level_dbu: Optional[float],
    amp_b_input_level_dbu: Optional[float],
) -> CalibrationResult:
    """Decide whether/how to apply per-model input calibration.

    - `mode="raw"`: never compensate, regardless of available metadata.
    - `mode="auto"`: compensate only when BOTH models report a calibrated
      `input_level_dbu`. If only one (or neither) does, fall back to raw for
      both rather than silently calibrating just one model -- that would
      apply a real physical assumption to one amp and not the other, which
      is worse than assuming neither is calibrated.
    """
    if mode not in ("auto", "raw"):
        raise ValueError(f"unknown calibration mode: {mode!r} (expected 'auto' or 'raw')")

    if mode == "raw":
        return CalibrationResult(
            mode="raw", applied=False, reference_input_level_dbu=reference_input_level_dbu,
            amp_a_model_input_level_dbu=amp_a_input_level_dbu, amp_b_model_input_level_dbu=amp_b_input_level_dbu,
            amp_a_gain_db=0.0, amp_b_gain_db=0.0, warning=None,
        )

    both_calibrated = amp_a_input_level_dbu is not None and amp_b_input_level_dbu is not None
    if not both_calibrated:
        warning = _UNAVAILABLE_WARNING if (amp_a_input_level_dbu is not None or amp_b_input_level_dbu is not None) else None
        return CalibrationResult(
            mode="auto", applied=False, reference_input_level_dbu=reference_input_level_dbu,
            amp_a_model_input_level_dbu=amp_a_input_level_dbu, amp_b_model_input_level_dbu=amp_b_input_level_dbu,
            amp_a_gain_db=0.0, amp_b_gain_db=0.0, warning=warning,
        )

    amp_a_gain_db = input_calibration_gain_db(reference_input_level_dbu, amp_a_input_level_dbu)
    amp_b_gain_db = input_calibration_gain_db(reference_input_level_dbu, amp_b_input_level_dbu)
    return CalibrationResult(
        mode="auto", applied=True, reference_input_level_dbu=reference_input_level_dbu,
        amp_a_model_input_level_dbu=amp_a_input_level_dbu, amp_b_model_input_level_dbu=amp_b_input_level_dbu,
        amp_a_gain_db=amp_a_gain_db, amp_b_gain_db=amp_b_gain_db, warning=None,
    )
