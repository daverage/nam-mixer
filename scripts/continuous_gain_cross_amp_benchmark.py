"""Cross-amp generalization benchmark for the Continuous Gain virtual-input-
gain approach. See docs/CONTINUOUS_GAIN_CROSS_AMP_GENERALIZATION.md.

Extends scripts/continuous_gain_virtual_gain_benchmark.py (the JCM800-only
single/best-of-3-fixed-anchor study) into a fully automated, per-amp
exhaustive anchor search across three amplifiers with different behaviour:

  * Marshall JCM800 2203 (updated) -- dense Gain sweep, 19 captures,
    1.0-10.0 in 0.5 steps. Known-good regression case.
  * Fender Super-Sonic 60W (Bassman channel, T5/B5 fixed) -- Volume sweep,
    10 captures, 1-10 integer steps.
  * Fender 57 Custom Twin (Channel 1) -- Volume sweep, 10 captures, 1-10
    integer steps.

No model training of any kind. Signal path for every reconstruction:

    DI --(* 10**(input_gain_db / 20))--> fixed NAM --> output

Three DIs are used (moderate_brit "standard", clean_smooth "clean",
high_metalcore "metalcore") so anchor-selection conclusions are not an
artifact of one input clip -- see run_ground_truth_harness usage per DI.

Experiments (see module docstring sections for each):
  A. Exhaustive single-anchor search: every real capture tried as anchor
     against every other capture as target.
  B. Best 2-anchor system, derived combinatorially from A (no extra
     rendering: for a fixed pair, the per-target best is just
     min(result_anchor1, result_anchor2), already computed in A).
  C. Best 3-anchor system, derived the same way.
  D. Dense discrete-capture interpolation baseline (hybrid.continuous_gain),
     leave-one-out over interior captures (the only "genuine intermediate"
     positions available once endpoints are excluded).

Rendering cost lives entirely in Experiment A (the only step that calls
`render()`); B/C/D are pure combinatorics/interpolation over A's results
plus D's own interpolation renders.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.audio_metrics import rms_dbfs, spectral_magnitude_correlation
from hybrid.continuous_gain import GainCapture, GainCaptureSet, interpolate_output, run_ground_truth_harness
from hybrid.nam_loader import NamModel, load_nam
from hybrid.render import render
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AMPS_ROOT = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps")

EVAL_SECONDS = 6.0
DI_FILES = {
    "standard": _REPO_ROOT / "assets/di/moderate_brit.wav",
    "clean": _REPO_ROOT / "assets/di/clean_smooth.wav",
    "metalcore": _REPO_ROOT / "assets/di/high_metalcore.wav",
}

COARSE_RANGE_DB = (-24.0, 24.0)
COARSE_STEP_DB = 2.0
FINE_HALF_RANGE_DB = 2.0
FINE_STEP_DB = 0.2

# Amps excluded from the primary pass per the task brief (Peavey 5150's
# saturated-amp validation metric issue is unresolved).


@dataclass
class AmpDataset:
    name: str
    control_label: str  # "Gain" or "Volume"
    captures: dict[float, Path]  # control_position -> .nam path


def _jcm800_captures() -> dict[float, Path]:
    d = _AMPS_ROOT / "Marshall JCM800 2203 - updated"
    out: dict[float, Path] = {}
    for i in range(19):
        g = round(1.0 + 0.5 * i, 1)
        name = "jcm800-high-ga10-11.4dBu.nam" if g == 10.0 else f"jcm800-high-g{g:.1f}-11.4dBu.nam"
        out[g] = d / name
    return out


def _supersonic_captures() -> dict[float, Path]:
    d = _AMPS_ROOT / "[500 Epochs] Fender Super-Sonic 60W Head mk.1 - Flat EQ - Complete Pack"
    return {float(v): d / f"Super-Sonic Bassman Ch T5 B5 V{v}.nam" for v in range(1, 11)}


def _twin_captures() -> dict[float, Path]:
    d = _AMPS_ROOT / "FENDER 57 CUSTOM TWIN (MULTI GAIN)"
    return {float(v): d / f"57 CUSTOM TWIN -  CH 1 - VOL {v}.nam" for v in range(1, 11)}


AMPS: list[AmpDataset] = [
    AmpDataset("Marshall JCM800 2203 (High, updated)", "Gain", _jcm800_captures()),
    AmpDataset("Fender Super-Sonic 60W (Bassman ch, T5/B5)", "Volume", _supersonic_captures()),
    AmpDataset("Fender 57 Custom Twin (Ch 1)", "Volume", _twin_captures()),
]


def render_with_input_gain(model: NamModel, dry: np.ndarray, sample_rate: int, input_gain_db: float) -> np.ndarray:
    driven = dry * (10.0 ** (input_gain_db / 20.0))
    return render(model, driven.astype(np.float32), sample_rate)


@dataclass
class SearchResult:
    anchor: float
    target: float
    best_input_gain_db: float
    raw_esr: float
    gain_normalized_esr: float
    level_delta_db: float
    spectral_correlation: float
    hit_search_boundary: bool


def search_best_input_gain(model: NamModel, dry: np.ndarray, sample_rate: int, target: np.ndarray) -> dict:
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

    lo, hi = COARSE_RANGE_DB
    coarse = [evaluate(g) for g in np.arange(lo, hi + 1e-9, COARSE_STEP_DB)]
    coarse_best = min(coarse, key=lambda r: r["raw_esr"])
    hit_boundary = coarse_best["gain_db"] in (lo, hi)

    fine_lo = max(lo, coarse_best["gain_db"] - FINE_HALF_RANGE_DB)
    fine_hi = min(hi, coarse_best["gain_db"] + FINE_HALF_RANGE_DB)
    fine = [evaluate(g) for g in np.arange(fine_lo, fine_hi + 1e-9, FINE_STEP_DB)]
    best = min(coarse + fine, key=lambda r: r["raw_esr"])
    best["hit_search_boundary"] = hit_boundary
    return best


def detect_latency_anomalies(models: dict[float, NamModel], positions: list[float], dry: np.ndarray, sr: int,
                              ground_truth: dict[float, np.ndarray]) -> dict[float, dict]:
    """Cross-correlate each capture's render against its immediate neighbours
    to flag a genuine timing defect (the JCM800 Gain-8.5 finding), rather
    than assume every dataset is clean. Returns {position: diagnostic}."""
    flags: dict[float, dict] = {}
    for i, p in enumerate(positions):
        neighbours = []
        if i > 0:
            neighbours.append(positions[i - 1])
        if i < len(positions) - 1:
            neighbours.append(positions[i + 1])
        if not neighbours:
            continue
        lags = []
        for nb in neighbours:
            a = ground_truth[p]
            b = ground_truth[nb]
            n = min(len(a), len(b))
            corr = np.correlate(a[:n] - a[:n].mean(), b[:n] - b[:n].mean(), mode="full")
            lag = int(np.argmax(corr) - (n - 1))
            peak = float(np.max(corr) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))
            lags.append({"neighbour": nb, "lag_samples": lag, "aligned_correlation": peak})
        worst_lag = max(abs(l["lag_samples"]) for l in lags)
        flags[p] = {"neighbour_lags": lags, "suspect": worst_lag > 20}
    return flags


def run_amp(amp: AmpDataset, di_name: str, dry_full: np.ndarray, sr: int) -> dict:
    dry = dry_full.astype(np.float32)[-int(EVAL_SECONDS * sr):]
    positions = sorted(amp.captures)
    print(f"\n### {amp.name} -- DI={di_name} -- {len(positions)} captures ###")

    t0 = time.time()
    models = {p: load_nam(amp.captures[p]) for p in positions}
    ground_truth = {p: render(models[p], dry, sr) for p in positions}
    print(f"  Loaded + rendered {len(positions)} ground-truth captures in {time.time() - t0:.1f}s")

    anomalies = detect_latency_anomalies(models, positions, dry, sr, ground_truth)
    excluded = {p for p, d in anomalies.items() if d["suspect"]}
    if excluded:
        print(f"  Latency-anomaly flagged and EXCLUDED from aggregates: {sorted(excluded)}")
    else:
        print("  No latency anomalies detected among neighbouring captures.")

    included = [p for p in positions if p not in excluded]

    # Sanity: 0 dB reproduces original bit-for-bit, for one representative anchor.
    rep = included[len(included) // 2]
    zero_db = render_with_input_gain(models[rep], dry, sr, 0.0)
    n = min(len(zero_db), len(ground_truth[rep]))
    max_diff = float(np.max(np.abs(zero_db[:n] - ground_truth[rep][:n])))
    assert max_diff < 1e-5, f"0dB sanity check failed for {amp.name} anchor {rep}: {max_diff}"

    print(f"  Experiment A: exhaustive {len(positions)}x{len(positions)} anchor search...")
    matrix: dict[float, dict[float, dict]] = {}
    t0 = time.time()
    n_done = 0
    for anchor in positions:
        matrix[anchor] = {}
        for target in positions:
            if anchor == target:
                matrix[anchor][target] = {
                    "gain_db": 0.0, "raw_esr": 0.0, "gain_normalized_esr": 0.0,
                    "level_delta_db": 0.0, "spectral_correlation": 1.0, "hit_search_boundary": False,
                }
                continue
            matrix[anchor][target] = search_best_input_gain(models[anchor], dry, sr, ground_truth[target])
            n_done += 1
    print(f"  Experiment A done: {n_done} anchor/target searches in {time.time() - t0:.1f}s")

    def mean_worst(vals: list[float]) -> tuple[float, float]:
        return (float(np.mean(vals)), float(np.max(vals)))

    # -- Experiment A summary per anchor (excluding self-match & anomalies) --
    exp_a_summary = {}
    for anchor in positions:
        errs = [matrix[anchor][t]["raw_esr"] for t in included if t != anchor]
        boundary_hits = sum(1 for t in included if t != anchor and matrix[anchor][t]["hit_search_boundary"])
        mean_e, worst_e = mean_worst(errs) if errs else (float("nan"), float("nan"))
        exp_a_summary[anchor] = {"mean_raw_esr": mean_e, "worst_raw_esr": worst_e, "boundary_hits": boundary_hits}
    best_single = min(exp_a_summary, key=lambda a: exp_a_summary[a]["mean_raw_esr"])

    # -- Experiment B: best pair, derived combinatorially --
    print("  Experiment B: searching all anchor pairs...")
    best_pair, best_pair_stats = None, None
    for pair in combinations(positions, 2):
        errs = []
        for t in included:
            if t in pair:
                errs.append(0.0)
                continue
            errs.append(min(matrix[a][t]["raw_esr"] for a in pair))
        mean_e, worst_e = mean_worst(errs)
        if best_pair_stats is None or mean_e < best_pair_stats["mean_raw_esr"]:
            best_pair, best_pair_stats = pair, {"mean_raw_esr": mean_e, "worst_raw_esr": worst_e}
    print(f"  Best pair: {best_pair} mean_raw_esr={best_pair_stats['mean_raw_esr']:.4f}")

    # -- Experiment C: best triple, derived combinatorially --
    print("  Experiment C: searching all anchor triples...")
    best_triple, best_triple_stats = None, None
    for triple in combinations(positions, 3):
        errs = []
        for t in included:
            if t in triple:
                errs.append(0.0)
                continue
            errs.append(min(matrix[a][t]["raw_esr"] for a in triple))
        mean_e, worst_e = mean_worst(errs)
        if best_triple_stats is None or mean_e < best_triple_stats["mean_raw_esr"]:
            best_triple, best_triple_stats = triple, {"mean_raw_esr": mean_e, "worst_raw_esr": worst_e}
    print(f"  Best triple: {best_triple} mean_raw_esr={best_triple_stats['mean_raw_esr']:.4f}")

    # -- Experiment D: dense discrete interpolation, leave-one-out over interior captures --
    print("  Experiment D: dense discrete interpolation (leave-one-out, interior captures)...")
    interior = included[1:-1]
    exp_d = []
    lo_pos, hi_pos = positions[0], positions[-1]
    for held_out in interior:
        train_positions = [p for p in included if p != held_out]
        train_captures = [GainCapture(model=models[p], control_position=p, label=f"pos_{p:g}") for p in train_positions]
        capture_set = GainCaptureSet(train_captures)
        harness = run_ground_truth_harness(capture_set, dry, sr, calibration_mode="raw")
        frac = (held_out - lo_pos) / (hi_pos - lo_pos)
        reconstructed = interpolate_output(harness, capture_set, frac)
        real = ground_truth[held_out]
        n = min(len(reconstructed), len(real))
        esr = compute_esr_metrics(reconstructed[:n], real[:n])
        exp_d.append({
            "held_out": held_out, "raw_esr": esr["raw_esr"], "gain_normalized_esr": esr["gain_normalized_esr"],
            "level_delta_db": abs(rms_dbfs(reconstructed[:n]) - rms_dbfs(real[:n])),
            "spectral_correlation": spectral_magnitude_correlation(reconstructed[:n], real[:n]),
        })
    dense_mean, dense_worst = mean_worst([r["raw_esr"] for r in exp_d]) if exp_d else (float("nan"), float("nan"))
    print(f"  Dense interpolation (leave-one-out): mean_raw_esr={dense_mean:.4f} worst={dense_worst:.4f}")

    return {
        "amp": amp.name, "control_label": amp.control_label, "di": di_name,
        "positions": positions, "excluded_latency_anomalies": sorted(excluded),
        "latency_anomaly_diagnostics": {str(k): v for k, v in anomalies.items()},
        "matrix": {str(a): {str(t): v for t, v in row.items()} for a, row in matrix.items()},
        "experiment_a_summary": {str(k): v for k, v in exp_a_summary.items()},
        "best_single_anchor": {"position": best_single, **exp_a_summary[best_single]},
        "best_pair": {"positions": list(best_pair), **best_pair_stats},
        "best_triple": {"positions": list(best_triple), **best_triple_stats},
        "dense_interpolation_leave_one_out": {"mean_raw_esr": dense_mean, "worst_raw_esr": dense_worst, "detail": exp_d},
    }


def run_stability_check(amp: AmpDataset, di_name: str, dry_full: np.ndarray, sr: int,
                         selected_anchors: list[float]) -> dict:
    """Cheaper cross-DI check: re-run the search ONLY for the anchors already
    selected as best-1/2/3 on the primary ("standard") DI, against every
    target, on a different DI. Answers "does anchor selection / mapping
    remain stable across DI material" without re-running the full NxN
    exhaustive search for every DI (which would be N times more expensive
    for no additional information about anchor STABILITY specifically)."""
    dry = dry_full.astype(np.float32)[-int(EVAL_SECONDS * sr):]
    positions = sorted(amp.captures)
    print(f"\n### {amp.name} -- DI stability check on '{di_name}' -- anchors {selected_anchors} ###")
    models = {p: load_nam(amp.captures[p]) for p in set(selected_anchors) | set(positions)}
    ground_truth = {p: render(models[p], dry, sr) for p in positions}

    per_anchor: dict[float, dict[float, dict]] = {}
    t0 = time.time()
    for anchor in selected_anchors:
        per_anchor[anchor] = {}
        for target in positions:
            if anchor == target:
                per_anchor[anchor][target] = {"gain_db": 0.0, "raw_esr": 0.0}
                continue
            per_anchor[anchor][target] = search_best_input_gain(models[anchor], dry, sr, ground_truth[target])
    print(f"  Stability check done in {time.time() - t0:.1f}s")

    best_of_selected = []
    for target in positions:
        best = min(selected_anchors, key=lambda a: per_anchor[a][target]["raw_esr"])
        best_of_selected.append(per_anchor[best][target]["raw_esr"])

    return {
        "amp": amp.name, "di": di_name, "selected_anchors": selected_anchors,
        "per_anchor": {str(a): {str(t): v for t, v in row.items()} for a, row in per_anchor.items()},
        "mean_raw_esr_using_selected_anchors": float(np.mean(best_of_selected)),
        "worst_raw_esr_using_selected_anchors": float(np.max(best_of_selected)),
    }


def main() -> None:
    all_results = []
    stability_results = []
    dry_cache: dict[str, tuple[np.ndarray, int]] = {}
    for di_name, di_path in DI_FILES.items():
        dry_cache[di_name] = sf.read(di_path)

    primary_di = "standard"
    dry_full, sr = dry_cache[primary_di]
    for amp in AMPS:
        result = run_amp(amp, primary_di, dry_full, sr)
        all_results.append(result)

        selected = sorted(set([result["best_single_anchor"]["position"], *result["best_pair"]["positions"],
                                *result["best_triple"]["positions"]]))
        for di_name in DI_FILES:
            if di_name == primary_di:
                continue
            other_dry, other_sr = dry_cache[di_name]
            stability_results.append(run_stability_check(amp, di_name, other_dry, other_sr, selected))

    out_path = _REPO_ROOT / "work/continuous_gain_cross_amp_benchmark.json"
    out_path.write_text(json.dumps({"primary_di": primary_di, "experiments": all_results,
                                     "di_stability_checks": stability_results}, indent=2))
    print(f"\nWrote full report to {out_path}")


if __name__ == "__main__":
    main()
