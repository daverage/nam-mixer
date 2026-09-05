"""Held-out validation: compare the LIVE reference hybrid (two source NAMs +
a locked HybridDesign) against a trained A2 export, on material the A2 never
trained on -- docs/phase3.md sections 24-28.

Deliberately independent of the training environment: rendering both the
reference hybrid and the trained A2 export uses `hybrid.render.render()`
(the native NAMCore tool), never torch/neural-amp-modeler, so this module
(and scripts/validate_a2.py) runs fine in the normal Flask/runtime
environment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import align_to_reference
from .blend import CrossoverConfig, blend
from .calibration import resolve_calibration
from .design import HybridDesign
from .envelope import BoundedEnvelopeConfig, bounded_causal_envelope_db
from .nam_loader import load_nam
from .render import render


@dataclass
class ReferenceHybridResult:
    hybrid: np.ndarray
    amp_a: np.ndarray
    amp_b: np.ndarray
    envelope_db: np.ndarray
    alignment_offset_samples: int


def render_reference_hybrid(design: HybridDesign, dry: np.ndarray, sample_rate: int) -> ReferenceHybridResult:
    """Render the LIVE two-NAM reference hybrid for held-out validation,
    reusing the frozen `design` exactly as auditioned -- same crossover,
    transition, fixed B trim, calibration rule, and envelope config as
    `hybrid.pipeline.render_pair`/`build_hybrid`, alignment OFF unless the
    design says otherwise. `dry` may already have a real input-profile gain
    applied by the caller (docs/phase3.md section 25 -- unlike target
    generation, validation DOES apply real profile gains, as actual audio,
    never the deprecated envelope-only `dry_gain_db`).
    """
    dry = np.asarray(dry, dtype=np.float32)
    amp_a = load_nam(design.amp_a_path)
    amp_b = load_nam(design.amp_b_path)

    calib = resolve_calibration(
        design.calibration_mode, design.reference_input_level_dbu,
        amp_a.input_level_dbu, amp_b.input_level_dbu,
    )
    amp_a_input = (dry * (10.0 ** (calib.amp_a_gain_db / 20.0))).astype(np.float32)
    amp_b_input = (dry * (10.0 ** (calib.amp_b_gain_db / 20.0))).astype(np.float32)

    amp_a_render = render(amp_a, amp_a_input, sample_rate)
    amp_b_render = render(amp_b, amp_b_input, sample_rate)

    envelope_config = BoundedEnvelopeConfig(
        rms_window_ms=design.envelope_rms_window_ms,
        attack_avg_ms=design.envelope_attack_avg_ms,
        release_window_ms=design.envelope_release_window_ms,
        release_range_db=design.envelope_release_range_db,
    )
    envelope_db = bounded_causal_envelope_db(dry, sample_rate, envelope_config)

    amp_b_aligned, offset = align_to_reference(amp_a_render, amp_b_render, enabled=design.alignment_enabled)

    config = CrossoverConfig(
        crossover_dbfs=design.crossover_dbfs,
        transition_width_db=design.transition_width_db,
        amp_b_trim_db=design.effective_b_trim_db,
    )
    hybrid_audio, _t = blend(envelope_db, amp_a_render, amp_b_aligned, config)

    return ReferenceHybridResult(
        hybrid=hybrid_audio, amp_a=amp_a_render, amp_b=amp_b_aligned,
        envelope_db=envelope_db, alignment_offset_samples=offset,
    )


def render_trained_a2(nam_path, dry: np.ndarray, sample_rate: int, slim: float | None = None) -> np.ndarray:
    """Thin wrapper for rendering `dry` through a trained/exported A2 -- see
    `hybrid.render.render`'s `slim` kwarg (None/0.0 = Full, 1.0 = Lite)."""
    model = load_nam(nam_path)
    return render(model, np.asarray(dry, dtype=np.float32), sample_rate, slim=slim)


def compute_esr_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """raw ESR, gain-normalized ESR, RMS difference, peak difference between
    two equal-purpose renders `a` (candidate) and `b` (reference) -- same
    metrics scripts/train_a2.py's compare_to_target uses, factored out here
    so held-out validation doesn't duplicate the math.
    """
    n = min(len(a), len(b))
    x = np.asarray(a[:n], dtype=np.float64)
    y = np.asarray(b[:n], dtype=np.float64)

    err = x - y
    esr = float(np.sum(err**2) / max(np.sum(y**2), 1e-12))

    x_rms = np.sqrt(np.mean(x**2))
    y_rms = np.sqrt(np.mean(y**2))
    if y_rms > 1e-12:
        x_normalized = x * (y_rms / max(x_rms, 1e-12))
        gain_normalized_esr = float(np.sum((x_normalized - y) ** 2) / max(np.sum(y**2), 1e-12))
    else:
        gain_normalized_esr = float("nan")

    return {
        "raw_esr": esr,
        "gain_normalized_esr": gain_normalized_esr,
        "rms_difference": float(abs(x_rms - y_rms)),
        "peak_difference": float(abs(np.max(np.abs(x)) - np.max(np.abs(y)))),
    }
