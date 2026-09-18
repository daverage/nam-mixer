"""Fully automated benchmark: how much of a real amplifier's physical Gain
sweep can be reproduced using ORDINARY NAM-style input gain (dB, applied to
the dry signal BEFORE inference -- exactly the "Input Gain" control in a
real NAM player) on fixed, untrained NAM captures?

See docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md for the full writeup.
No model training of any kind happens here -- this is pure signal
processing + real NAM inference (`hybrid.render.render`), reusing
`hybrid.continuous_gain`/`hybrid.audio_metrics`/`hybrid.validation` for the
discrete-interpolation baseline and metrics, exactly as established in the
earlier reports.

Three experiments, all against the same 19 real ground-truth captures
(Gain 1.0 to 10.0 in 0.5 steps) on the dense "Marshall JCM800 2203 -
updated" dataset:

  A. One fixed NAM (tested independently at anchors G1, G5, G10) + a
     virtual input-gain search (coarse then fine) for every target gain.
  B. Derived from A: for every target, which of the 3 anchors (with its
     own best input gain) gets closest?
  C. The existing dense G1-G10 nearest-neighbour interpolation baseline,
     evaluated on the half-step targets only (the genuinely-real withheld
     positions), for direct comparison.

Signal path for A/B (matches a real NAM player exactly):

    DI --(* 10**(input_gain_db/20))--> fixed NAM --> output

Never gain-adjusted after rendering -- see `render_with_input_gain`.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.audio_metrics import rms_dbfs, spectral_magnitude_correlation
from hybrid.continuous_gain import GainCapture, GainCaptureSet, interpolate_output, run_ground_truth_harness
from hybrid.nam_loader import NamModel, load_nam
from hybrid.render import render
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
AMP_DIR = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps/Marshall JCM800 2203 - updated")
EVAL_DI_FILE = _REPO_ROOT / "assets/di/moderate_brit.wav"
EVAL_SECONDS = 6.0  # same held-out convention as docs/CONTINUOUS_GAIN_CONDITIONAL_MODEL.md
ANCHOR_GAINS = (1.0, 5.0, 10.0)
ALL_GAINS = [round(1.0 + 0.5 * i, 1) for i in range(19)]  # 1.0, 1.5, ..., 10.0
HALF_STEP_GAINS = [g for g in ALL_GAINS if g % 1 != 0]  # 1.5, 2.5, ..., 9.5 -- the genuinely-real withheld positions


def gain_path(gain: float) -> Path:
    if gain == 10.0:
        return AMP_DIR / "jcm800-high-ga10-11.4dBu.nam"
    return AMP_DIR / f"jcm800-high-g{gain:.1f}-11.4dBu.nam"


def render_with_input_gain(model: NamModel, dry: np.ndarray, sample_rate: int, input_gain_db: float) -> np.ndarray:
    """The exact NAM-player signal path: gain is applied to the DRY signal
    BEFORE inference, never to the rendered output afterward. This is the
    one function every part of this benchmark routes through, so there is
    a single place to verify this -- see `sanity_check_pre_inference_gain`.
    """
    driven = dry * (10.0 ** (input_gain_db / 20.0))
    return render(model, driven.astype(np.float32), sample_rate)


@dataclass
class GainSearchResult:
    anchor_gain: float
    target_gain: float
    best_input_gain_db: float
    raw_esr: float
    gain_normalized_esr: float
    level_delta_db: float
    spectral_correlation: float
    hit_search_boundary: bool


def search_best_input_gain(
    model: NamModel, dry: np.ndarray, sample_rate: int, target: np.ndarray,
    *, coarse_range_db: tuple[float, float] = (-24.0, 24.0), coarse_step_db: float = 2.0,
    fine_half_range_db: float = 2.0, fine_step_db: float = 0.2,
) -> GainSearchResult:
    """Coarse-then-fine search for the input gain (dB) that MINIMIZES RAW
    (actual, unmatched) ESR against `target` -- i.e. what a real NAM-player
    user would optimize for by ear/meter, since raw output is what they
    actually hear. The level-matched (gain-normalized) ESR at that same
    optimum is reported alongside as the diagnostic that isolates whether
    remaining error is level or shape (see module docstring / doc Section
    on "actual vs level-matched").
    """
    def evaluate(gain_db: float) -> dict:
        candidate = render_with_input_gain(model, dry, sample_rate, gain_db)
        n = min(len(candidate), len(target))
        esr = compute_esr_metrics(candidate[:n], target[:n])
        return {
            "gain_db": gain_db,
            "raw_esr": esr["raw_esr"],
            "gain_normalized_esr": esr["gain_normalized_esr"],
            "level_delta_db": abs(rms_dbfs(candidate[:n]) - rms_dbfs(target[:n])),
            "spectral_correlation": spectral_magnitude_correlation(candidate[:n], target[:n]),
        }

    lo, hi = coarse_range_db
    coarse_candidates = np.arange(lo, hi + 1e-9, coarse_step_db)
    coarse_results = [evaluate(g) for g in coarse_candidates]
    coarse_best = min(coarse_results, key=lambda r: r["raw_esr"])
    hit_boundary = coarse_best["gain_db"] in (lo, hi)

    fine_lo = max(lo, coarse_best["gain_db"] - fine_half_range_db)
    fine_hi = min(hi, coarse_best["gain_db"] + fine_half_range_db)
    fine_candidates = np.arange(fine_lo, fine_hi + 1e-9, fine_step_db)
    fine_results = [evaluate(g) for g in fine_candidates]
    best = min(coarse_results + fine_results, key=lambda r: r["raw_esr"])

    return GainSearchResult(
        anchor_gain=float("nan"), target_gain=float("nan"),  # filled in by caller
        best_input_gain_db=best["gain_db"], raw_esr=best["raw_esr"],
        gain_normalized_esr=best["gain_normalized_esr"], level_delta_db=best["level_delta_db"],
        spectral_correlation=best["spectral_correlation"], hit_search_boundary=hit_boundary,
    )


def sanity_check_pre_inference_gain(model: NamModel, dry: np.ndarray, sample_rate: int, ground_truth: np.ndarray) -> None:
    """1) 0 dB input gain must reproduce the original capture's render
    essentially exactly. 2) Applying gain BEFORE inference must differ from
    applying the same gain AFTER inference (post-output) -- proves the gain
    is really being injected into the signal path the doc requires, not
    silently collapsing to a post-hoc rescale."""
    zero_db = render_with_input_gain(model, dry, sample_rate, 0.0)
    n = min(len(zero_db), len(ground_truth))
    max_diff = float(np.max(np.abs(zero_db[:n] - ground_truth[:n])))
    assert max_diff < 1e-5, f"0 dB input gain did not reproduce the original capture (max diff {max_diff})"

    test_gain_db = 6.0
    pre_inference = render_with_input_gain(model, dry, sample_rate, test_gain_db)
    post_output = ground_truth * (10.0 ** (test_gain_db / 20.0))
    n2 = min(len(pre_inference), len(post_output))
    diff = float(np.max(np.abs(pre_inference[:n2] - post_output[:n2])))
    assert diff > 1e-4, "Pre-inference gain is indistinguishable from post-output gain -- nonlinearity is not engaging"
    print(f"Sanity check passed: 0dB reproduces original (max diff {max_diff:.2e}); "
          f"pre-inference vs post-output gain differ by {diff:.4f} at +{test_gain_db}dB (nonlinearity confirmed engaged)")


def main() -> None:
    dry_full, sr = sf.read(EVAL_DI_FILE)
    dry = dry_full.astype(np.float32)[-int(EVAL_SECONDS * sr):]

    print(f"Loading {len(ALL_GAINS)} real captures and rendering ground truth...")
    models: dict[float, NamModel] = {g: load_nam(gain_path(g)) for g in ALL_GAINS}
    ground_truth: dict[float, np.ndarray] = {g: render(models[g], dry, sr) for g in ALL_GAINS}

    print("\n=== Sanity checks ===")
    for anchor in ANCHOR_GAINS:
        sanity_check_pre_inference_gain(models[anchor], dry, sr, ground_truth[anchor])

    print("\n=== Experiment A: one fixed NAM + virtual input gain, per anchor ===")
    experiment_a: dict[float, list[GainSearchResult]] = {}
    for anchor in ANCHOR_GAINS:
        print(f"\n-- Anchor G{anchor:g} --")
        results = []
        for target in ALL_GAINS:
            result = search_best_input_gain(models[anchor], dry, sr, ground_truth[target])
            result.anchor_gain, result.target_gain = anchor, target
            results.append(result)
            boundary_flag = " [HIT SEARCH BOUNDARY]" if result.hit_search_boundary else ""
            print(f"  target G{target:>4}: input_gain={result.best_input_gain_db:+6.2f}dB  "
                  f"raw_esr={result.raw_esr:.4f}  level_matched_esr={result.gain_normalized_esr:.4f}  "
                  f"level_Δ={result.level_delta_db:.2f}dB  spec_corr={result.spectral_correlation:.4f}{boundary_flag}")
        experiment_a[anchor] = results

    print("\n=== Experiment B: best of G1/G5/G10 + virtual gain, per target ===")
    experiment_b = []
    for i, target in enumerate(ALL_GAINS):
        candidates = [experiment_a[anchor][i] for anchor in ANCHOR_GAINS]
        best = min(candidates, key=lambda r: r.raw_esr)
        experiment_b.append(best)
        print(f"  target G{target:>4}: BEST anchor=G{best.anchor_gain:g}  input_gain={best.best_input_gain_db:+6.2f}dB  "
              f"raw_esr={best.raw_esr:.4f}  level_matched_esr={best.gain_normalized_esr:.4f}  spec_corr={best.spectral_correlation:.4f}")

    print("\n=== Experiment C: dense G1-G10 nearest-neighbour interpolation, half-step targets ===")
    integer_gains = [float(g) for g in range(1, 11)]
    training_captures = [GainCapture(model=models[g], control_position=g, label=f"G{g:g}") for g in integer_gains]
    capture_set = GainCaptureSet(training_captures)
    harness = run_ground_truth_harness(capture_set, dry, sr, calibration_mode="raw")

    experiment_c = []
    for target in HALF_STEP_GAINS:
        frac = (target - 1.0) / (10.0 - 1.0)
        reconstructed = interpolate_output(harness, capture_set, frac)
        real = ground_truth[target]
        n = min(len(reconstructed), len(real))
        esr = compute_esr_metrics(reconstructed[:n], real[:n])
        level_delta = abs(rms_dbfs(reconstructed[:n]) - rms_dbfs(real[:n]))
        spec_corr = spectral_magnitude_correlation(reconstructed[:n], real[:n])
        row = {"target_gain": target, "raw_esr": esr["raw_esr"], "gain_normalized_esr": esr["gain_normalized_esr"],
               "level_delta_db": level_delta, "spectral_correlation": spec_corr}
        experiment_c.append(row)
        print(f"  target G{target:>4}: raw_esr={row['raw_esr']:.4f}  level_matched_esr={row['gain_normalized_esr']:.4f}  "
              f"level_Δ={row['level_delta_db']:.2f}dB  spec_corr={row['spectral_correlation']:.4f}")

    # -- Representative audio for listening comparison --
    audio_dir = _REPO_ROOT / "work/continuous_gain_virtual_gain"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for target in (3.5, 6.5, 9.5):
        idx = ALL_GAINS.index(target)
        real = ground_truth[target]
        sf.write(audio_dir / f"g{target}_real.wav", real, sr)
        one_nam = experiment_b[idx]  # best single anchor result IS also usable for one-anchor rows; also dump G5-anchor explicitly
        g5_result = experiment_a[5.0][idx]
        g5_recon = render_with_input_gain(models[5.0], dry, sr, g5_result.best_input_gain_db)
        sf.write(audio_dir / f"g{target}_anchorG5_virtualgain.wav", g5_recon[:len(real)], sr)
        best_anchor_recon = render_with_input_gain(models[one_nam.anchor_gain], dry, sr, one_nam.best_input_gain_db)
        sf.write(audio_dir / f"g{target}_bestof3anchors_virtualgain.wav", best_anchor_recon[:len(real)], sr)
        if target in HALF_STEP_GAINS:
            frac = (target - 1.0) / 9.0
            dense_recon = interpolate_output(harness, capture_set, frac)
            sf.write(audio_dir / f"g{target}_denseinterp.wav", dense_recon[:len(real)], sr)
    print(f"\nWrote representative audio comparisons to {audio_dir}")

    # -- Save full report --
    report = {
        "eval_di_file": str(EVAL_DI_FILE), "eval_seconds": EVAL_SECONDS, "sample_rate": sr,
        "anchor_gains": list(ANCHOR_GAINS), "all_gains": ALL_GAINS, "half_step_gains": HALF_STEP_GAINS,
        "experiment_a": {str(anchor): [asdict(r) for r in results] for anchor, results in experiment_a.items()},
        "experiment_b": [asdict(r) for r in experiment_b],
        "experiment_c": experiment_c,
    }
    out_path = _REPO_ROOT / "work/continuous_gain_virtual_gain_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote full report to {out_path}")


if __name__ == "__main__":
    main()
