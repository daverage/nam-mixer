"""Mid-gain single NAM (Gain 5 + virtual input gain) vs. the production
multi-anchor Continuous Gain profile -- controlled, train/validation-separated
benchmark across four amp datasets.

See docs/CONTINUOUS_GAIN_MID_GAIN_VS_PROFILE.md for the full writeup, the
methodology judgment calls, and the results this script produces.

NOTHING is trained here and no production module is modified. Everything
routes through existing code:

  * `hybrid.render.render` (native NAMCore inference) via
    `hybrid.continuous_gain_profile.render_with_input_gain`
  * `hybrid.continuous_gain_profile.search_virtual_input_gain` -- the
    PRODUCTION profiling search (active-mask, coarse->fine, raw ESR)
  * `hybrid.continuous_gain_profile.build_profile` /
    `ContinuousGainRuntime`  -- the PRODUCTION multi-anchor profile builder
    and runtime engine (method B). Not an oracle/best-of search.
  * `hybrid.continuous_gain.run_ground_truth_harness` / `interpolate_output`
    -- the dense discrete-capture interpolation baseline (method C)
  * `hybrid.validation.compute_esr_metrics`,
    `hybrid.audio_metrics.rms_dbfs` / `spectral_magnitude_correlation`
  * `scripts.continuous_gain_cross_amp_benchmark`'s `AmpDataset` shape and
    neighbour-lag anomaly idea (refined here so a single defective capture
    does not poison its healthy neighbours -- see `isolate_defective_captures`)

Methods compared at every physical position:

  A_direct      single fixed Gain-5 NAM + a CONVENTIONAL linear gain law
                (one constant dB per knob step, least-squares fit through
                the anchor, training positions only)
  A_calibrated  single fixed Gain-5 NAM + the production profiling
                algorithm restricted to ONE anchor: measured per-position
                optimal input gain, monotonic-PCHIP interpolated
  B_profile     `build_profile(max_anchors=3)` + `ContinuousGainRuntime`
  C_dense       dense discrete-capture interpolation over the training
                captures

Train/validation separation:
  * JCM800: fit on integer G1..G10, validate on genuinely withheld
    half-step captures G1.5..G9.5.
  * Every other amp: leave-one-position-out over the integer captures. The
    held-out position never enters mapping fitting, anchor selection, or the
    interpolation capture set. Gain 5 always remains available as the fixed
    anchor for the single-anchor experiments (it is the one capture those
    methods are defined to own); at held-out position 5 the single-anchor
    methods are flagged `anchor_self_match` and excluded from aggregates.

Cost: the only expensive step is the per-amp NxN virtual-gain search grid,
which is computed ONCE per amp/DI and memoized to disk (resume-safe), then
reused by every leave-one-out fold and every method.

Usage:
    python scripts/continuous_gain_mid_vs_profile_benchmark.py
    python scripts/continuous_gain_mid_vs_profile_benchmark.py --amps jcm800
    python scripts/continuous_gain_mid_vs_profile_benchmark.py --skip-continuity
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid.audio_metrics import rms_dbfs, spectral_magnitude_correlation
from hybrid.continuous_gain import GainCapture, GainCaptureSet, interpolate_output, run_ground_truth_harness
from hybrid.continuous_gain_profile import (
    ContinuousGainProfile,
    ContinuousGainRuntime,
    GainSearchResult,
    TrainingTarget,
    _active_mask,
    _monotonic_interpolate,
    build_profile,
    render_with_input_gain,
)
from hybrid import continuous_gain_profile as cgp
from hybrid.nam_loader import NamModel, load_nam
from hybrid.render import render
from hybrid.validation import compute_esr_metrics

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AMPS_ROOT = Path("/Users/andrzejmarczewski/Documents/Amp Stuff/NAM/Amps")
OUT_DIR = _REPO_ROOT / "work/continuous_gain_mid_vs_profile"

EVAL_SECONDS = 6.0
ANCHOR_POSITION = 5.0
PRIMARY_DI = "standard"
DI_FILES = {
    "standard": _REPO_ROOT / "assets/di/moderate_brit.wav",
    "clean": _REPO_ROOT / "assets/di/clean_smooth.wav",
    "metalcore": _REPO_ROOT / "assets/di/high_metalcore.wav",
}

# Neighbour-lag magnitude (samples) above which a capture pair is considered
# to have a genuine timing offset rather than ordinary tonal difference --
# same threshold as scripts/continuous_gain_cross_amp_benchmark.py.
LATENCY_LAG_THRESHOLD = 20


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

@dataclass
class AmpSpec:
    key: str
    name: str
    control_label: str
    captures: dict[float, Path]
    train_positions: list[float]
    validation_positions: list[float]  # empty => leave-one-position-out
    notes: str = ""

    def label(self, position: float) -> str:
        return f"p{position:g}"


def _jcm800() -> AmpSpec:
    d = _AMPS_ROOT / "Marshall JCM800 2203 - updated"
    captures: dict[float, Path] = {}
    for i in range(19):
        g = round(1.0 + 0.5 * i, 1)
        name = "jcm800-high-ga10-11.4dBu.nam" if g == 10.0 else f"jcm800-high-g{g:.1f}-11.4dBu.nam"
        captures[g] = d / name
    train = [float(v) for v in range(1, 11)]
    validation = [p for p in sorted(captures) if p not in train]
    return AmpSpec("jcm800", "Marshall JCM800 2203 (High channel, updated)", "Gain",
                   captures, train, validation,
                   notes="Dense sweep with genuine half-step captures: strict train/validation split.")


def _supersonic() -> AmpSpec:
    d = _AMPS_ROOT / "[500 Epochs] Fender Super-Sonic 60W Head mk.1 - Flat EQ - Complete Pack"
    captures = {float(v): d / f"Super-Sonic Bassman Ch T5 B5 V{v}.nam" for v in range(1, 11)}
    return AmpSpec("supersonic", "Fender Super-Sonic 60W (Bassman ch, T5/B5)", "Volume",
                   captures, [float(v) for v in range(1, 11)], [],
                   notes="Integer captures only: leave-one-position-out.")


def _twin() -> AmpSpec:
    d = _AMPS_ROOT / "FENDER 57 CUSTOM TWIN (MULTI GAIN)"
    captures = {float(v): d / f"57 CUSTOM TWIN -  CH 1 - VOL {v}.nam" for v in range(1, 11)}
    return AmpSpec("twin", "Fender 57 Custom Twin (Ch 1)", "Volume",
                   captures, [float(v) for v in range(1, 11)], [],
                   notes="Integer captures only: leave-one-position-out.")


def _peavey() -> AmpSpec:
    d = _AMPS_ROOT / "Peavy 5150 (Head Only)"
    captures = {float(v): d / f"AMP HEAD - 5150 Gain {v}.nam" for v in range(1, 11)}
    return AmpSpec("peavey5150", "Peavey 5150 (head only, stock)", "Gain",
                   captures, [float(v) for v in range(1, 11)], [],
                   notes=("Heavily saturated preamp amp: raw ESR is known to be unreliable here "
                          "(docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md). Metrics reported but flagged."))


AMPS = {a.key: a for a in (_jcm800(), _supersonic(), _twin(), _peavey())}


# --------------------------------------------------------------------------
# Disk-memoized virtual-gain search (resume-safe)
# --------------------------------------------------------------------------

class SearchCache:
    """Memoizes `search_virtual_input_gain` results across leave-one-out folds
    and across runs. The search is deterministic (fixed grid, deterministic
    native renderer), so a cache hit is exactly the value the search would
    have recomputed -- this only removes redundant rendering, never changes
    a result."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.is_file():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.data = {}
        self.hits = 0
        self.misses = 0
        self._scope: Optional[str] = None
        self._path_to_label: dict[str, str] = {}
        self._target_to_label: dict[int, str] = {}
        self._real = cgp.search_virtual_input_gain
        self._dirty = 0

    def install(self, scope: str, path_to_label: dict[str, str], target_to_label: dict[int, str]) -> None:
        self._scope = scope
        self._path_to_label = path_to_label
        self._target_to_label = target_to_label
        cgp.search_virtual_input_gain = self._wrapper

    def uninstall(self) -> None:
        cgp.search_virtual_input_gain = self._real
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        self._dirty = 0

    def _wrapper(self, anchor_model, dry, sample_rate, target_output, mask, **kwargs) -> GainSearchResult:
        anchor_label = self._path_to_label.get(str(anchor_model.path))
        target_label = self._target_to_label.get(id(target_output))
        if anchor_label is None or target_label is None:
            # Unknown provenance -- never guess, just do the real search.
            return self._real(anchor_model, dry, sample_rate, target_output, mask, **kwargs)
        key = f"{self._scope}|{anchor_label}|{target_label}"
        cached = self.data.get(key)
        if cached is not None:
            self.hits += 1
            return GainSearchResult(**cached)
        self.misses += 1
        result = self._real(anchor_model, dry, sample_rate, target_output, mask, **kwargs)
        self.data[key] = {
            "input_gain_db": result.input_gain_db, "raw_esr": result.raw_esr,
            "hit_lower_bound": result.hit_lower_bound, "hit_upper_bound": result.hit_upper_bound,
        }
        self._dirty += 1
        if self._dirty >= 25:
            self.flush()
        return result

    def lookup(self, scope: str, anchor_label: str, target_label: str) -> Optional[dict]:
        return self.data.get(f"{scope}|{anchor_label}|{target_label}")


# --------------------------------------------------------------------------
# Metrics / helpers
# --------------------------------------------------------------------------

def metrics(candidate: np.ndarray, real: np.ndarray) -> dict:
    n = min(len(candidate), len(real))
    c, r = candidate[:n], real[:n]
    esr = compute_esr_metrics(c, r)
    return {
        "raw_esr": esr["raw_esr"],
        "level_matched_esr": esr["gain_normalized_esr"],
        "level_delta_db": float(rms_dbfs(c) - rms_dbfs(r)),
        "spectral_correlation": float(spectral_magnitude_correlation(c, r)),
    }


def load_di(name: str) -> tuple[np.ndarray, int]:
    dry, sr = sf.read(DI_FILES[name], dtype="float32", always_2d=False)
    if dry.ndim > 1:
        dry = dry.mean(axis=1)
    return dry[-int(EVAL_SECONDS * sr):].astype(np.float32), sr


def isolate_defective_captures(positions: list[float], gt: dict[float, np.ndarray]) -> dict:
    """Neighbour cross-correlation timing check, refined so that ONE defective
    capture does not also condemn its healthy neighbours.

    A position is reported `defective` only when (a) its lag against every
    available neighbour exceeds the threshold, and (b) those neighbours align
    cleanly with EACH OTHER when the suspect capture is skipped. That second
    condition is what separates the JCM800 Gain-8.5 file defect from its
    healthy Gain-8.0/Gain-9.0 neighbours (which the coarser pairwise check in
    scripts/continuous_gain_cross_amp_benchmark.py also flagged)."""
    def lag_between(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
        n = min(len(a), len(b))
        x, y = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        corr = np.correlate(x, y, mode="full")
        lag = int(np.argmax(corr) - (n - 1))
        peak = float(np.max(corr) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        return lag, peak

    report: dict[float, dict] = {}
    for i, p in enumerate(positions):
        neighbours = [positions[j] for j in (i - 1, i + 1) if 0 <= j < len(positions)]
        if not neighbours:
            continue
        lags = []
        for nb in neighbours:
            lag, peak = lag_between(gt[p], gt[nb])
            lags.append({"neighbour": nb, "lag_samples": lag, "aligned_correlation": peak})
        all_offset = all(abs(entry["lag_samples"]) > LATENCY_LAG_THRESHOLD for entry in lags)
        neighbours_agree = None
        if all_offset and len(neighbours) == 2:
            skip_lag, skip_peak = lag_between(gt[neighbours[0]], gt[neighbours[1]])
            neighbours_agree = {"lag_samples": skip_lag, "aligned_correlation": skip_peak}
        defective = bool(all_offset and (neighbours_agree is None
                                         or abs(neighbours_agree["lag_samples"]) <= LATENCY_LAG_THRESHOLD))
        report[p] = {"neighbour_lags": lags, "neighbours_skipping_this_capture": neighbours_agree,
                     "defective": defective}
    return report


def fit_direct_slope(points: list[tuple[float, float]], anchor_position: float) -> float:
    """Best single dB-per-knob-step constant for a CONVENTIONAL linear gain
    law through the anchor: gain_db(p) = slope * (p - anchor). Least squares
    with no intercept, because the anchor at its own position needs 0 dB by
    construction."""
    xs = np.array([p - anchor_position for p, _ in points if p != anchor_position], dtype=np.float64)
    ys = np.array([g for p, g in points if p != anchor_position], dtype=np.float64)
    if xs.size == 0 or np.sum(xs * xs) == 0:
        return 0.0
    return float(np.sum(xs * ys) / np.sum(xs * xs))


# --------------------------------------------------------------------------
# Per-amp benchmark
# --------------------------------------------------------------------------

def render_ground_truth(spec: AmpSpec, models: dict[float, NamModel], dry: np.ndarray, sr: int) -> dict[float, np.ndarray]:
    return {p: render(models[p], dry, sr) for p in sorted(spec.captures)}


def build_search_grid(spec: AmpSpec, models: dict[float, NamModel], gt: dict[float, np.ndarray],
                      dry: np.ndarray, sr: int, mask: np.ndarray, cache: SearchCache, scope: str) -> dict:
    """Full training x training virtual-gain grid, memoized. Only TRAINING
    positions participate -- a withheld capture's real output must never
    enter any fit."""
    grid: dict[tuple[float, float], dict] = {}
    for anchor in spec.train_positions:
        for target in spec.train_positions:
            if anchor == target:
                grid[(anchor, target)] = {"input_gain_db": 0.0, "raw_esr": 0.0,
                                          "hit_lower_bound": False, "hit_upper_bound": False}
                continue
            key = cache.lookup(scope, spec.label(anchor), spec.label(target))
            if key is None:
                result = cache._real(models[anchor], dry, sr, gt[target], mask)
                key = {"input_gain_db": result.input_gain_db, "raw_esr": result.raw_esr,
                       "hit_lower_bound": result.hit_lower_bound, "hit_upper_bound": result.hit_upper_bound}
                cache.data[f"{scope}|{spec.label(anchor)}|{spec.label(target)}"] = key
                cache.misses += 1
                cache._dirty += 1
                if cache._dirty >= 25:
                    cache.flush()
            else:
                cache.hits += 1
            grid[(anchor, target)] = key
    cache.flush()
    return grid


def build_profile_for_fold(spec: AmpSpec, models: dict[float, NamModel], gt: dict[float, np.ndarray],
                           dry: np.ndarray, sr: int, fit_positions: list[float],
                           cache: SearchCache, scope: str) -> ContinuousGainProfile:
    targets, nam_paths, target_to_label = [], {}, {}
    for p in fit_positions:
        label = spec.label(p)
        nam_paths[label] = str(spec.captures[p])
        output = gt[p]
        target_to_label[id(output)] = label
        targets.append(TrainingTarget(label=label, physical_position=p, model=models[p], output=output))
    path_to_label = {str(spec.captures[p]): spec.label(p) for p in fit_positions}
    cache.install(scope, path_to_label, target_to_label)
    try:
        profile = build_profile(targets, dry, sr, control_name=spec.control_label,
                                max_anchors=3, nam_paths=nam_paths)
    finally:
        cache.uninstall()
    return profile


def dense_reconstruction(spec: AmpSpec, models: dict[float, NamModel], dry: np.ndarray, sr: int,
                         fit_positions: list[float], position: float) -> tuple[np.ndarray, bool]:
    captures = [GainCapture(model=models[p], control_position=p, label=spec.label(p)) for p in fit_positions]
    capture_set = GainCaptureSet(captures)
    harness = run_ground_truth_harness(capture_set, dry, sr, calibration_mode="raw")
    lo, hi = min(fit_positions), max(fit_positions)
    frac = (position - lo) / (hi - lo)
    extrapolated = not (lo <= position <= hi)
    return interpolate_output(harness, capture_set, frac), extrapolated


def evaluate_amp(spec: AmpSpec, cache: SearchCache, di_names: list[str],
                 write_wavs: bool = True) -> dict:
    print(f"\n=== {spec.name} ===")
    missing = [str(p) for p in spec.captures.values() if not p.is_file()]
    if missing:
        print(f"  SKIP: missing capture files: {missing[:3]}")
        return {"amp": spec.key, "skipped": True, "missing": missing}

    dis = {name: load_di(name) for name in di_names}
    dry, sr = dis[PRIMARY_DI]
    mask = _active_mask(dry, sr, cgp.DEFAULT_BOUNDED_ENVELOPE_CONFIG)

    t0 = time.time()
    models = {p: load_nam(path) for p, path in spec.captures.items()}
    gt = {name: render_ground_truth(spec, models, d, s) for name, (d, s) in dis.items()}
    print(f"  Ground truth: {len(models)} captures x {len(dis)} DIs in {time.time() - t0:.1f}s")

    anomalies = isolate_defective_captures(sorted(spec.captures), gt[PRIMARY_DI])
    defective = sorted(p for p, r in anomalies.items() if r["defective"])
    print(f"  Timing-defect isolation: {defective if defective else 'none'}")

    scope = f"{spec.key}|{PRIMARY_DI}"
    t0 = time.time()
    grid = build_search_grid(spec, models, gt[PRIMARY_DI], dry, sr, mask, cache, scope)
    print(f"  Search grid ({len(spec.train_positions)}x{len(spec.train_positions)}) "
          f"in {time.time() - t0:.1f}s (cache hits {cache.hits}, misses {cache.misses})")

    anchor = ANCHOR_POSITION
    assert anchor in spec.train_positions, f"{spec.key}: no Gain-5 anchor capture"

    lopo = not spec.validation_positions
    eval_positions = spec.validation_positions if not lopo else list(spec.train_positions)

    rows: list[dict] = []
    fold_profiles: dict[float, dict] = {}
    t0 = time.time()
    for position in eval_positions:
        fit_positions = [p for p in spec.train_positions if p != position] if lopo else list(spec.train_positions)
        measured = [(p, grid[(anchor, p)]["input_gain_db"]) for p in fit_positions]
        pinned = sum(1 for p in fit_positions
                     if grid[(anchor, p)]["hit_lower_bound"] or grid[(anchor, p)]["hit_upper_bound"])
        slope = fit_direct_slope(measured, anchor)
        mapped_positions = [p for p, _ in measured]
        extrapolating = not (min(mapped_positions) <= position <= max(mapped_positions))

        gain_direct = slope * (position - anchor)
        gain_calibrated = (measured[0][1] if len(measured) == 1
                           else float(_monotonic_interpolate([p for p, _ in measured],
                                                             [g for _, g in measured], position)))

        profile = build_profile_for_fold(spec, models, gt[PRIMARY_DI], dry, sr, fit_positions, cache, scope)
        anchor_models = {a.label: models[a.physical_position] for a in profile.anchors}
        runtime = ContinuousGainRuntime(profile, anchor_models)
        region = runtime._region_for(runtime._clamp_position(position))
        fold_profiles[position] = {
            "anchors": [a.label for a in profile.anchors],
            "anchor_positions": [a.physical_position for a in profile.anchors],
            "quality": profile.validation.quality,
            "worst_raw_esr_build_time": profile.validation.worst_raw_esr,
            "region_used": region.anchor_label,
            "region_input_gain_db": region.input_gain_db(position),
        }

        per_di: dict[str, dict] = {}
        for di_name, (d, s) in dis.items():
            real = gt[di_name][position]
            recon = {
                "A_direct": render_with_input_gain(models[anchor], d, s, gain_direct),
                "A_calibrated": render_with_input_gain(models[anchor], d, s, gain_calibrated),
                "B_profile": runtime.render_at(position, d, s),
            }
            dense, dense_extrap = dense_reconstruction(spec, models, d, s, fit_positions, position)
            recon["C_dense"] = dense
            per_di[di_name] = {m: metrics(v, real) for m, v in recon.items()}
            if write_wavs and di_name == PRIMARY_DI:
                maybe_write_wavs(spec, position, real, recon)

        rows.append({
            "position": position,
            "withheld": True if lopo else True,
            "extrapolated_mapping": extrapolating or dense_extrap,
            "anchor_self_match": position == anchor,
            "timing_defect": bool(anomalies.get(position, {}).get("defective")),
            "gains": {"A_direct_db": gain_direct, "A_calibrated_db": gain_calibrated,
                      "B_profile_db": fold_profiles[position]["region_input_gain_db"],
                      "C_dense_db": None},
            "direct_slope_db_per_step": slope,
            "boundary_pinned_fit_points": pinned,
            "profile": fold_profiles[position],
            "metrics": per_di,
        })
        print(f"    pos {position:>4g}  A_dir {per_di[PRIMARY_DI]['A_direct']['raw_esr']:.4f}  "
              f"A_cal {per_di[PRIMARY_DI]['A_calibrated']['raw_esr']:.4f}  "
              f"B {per_di[PRIMARY_DI]['B_profile']['raw_esr']:.4f}  "
              f"C {per_di[PRIMARY_DI]['C_dense']['raw_esr']:.4f}"
              f"{'  [defect]' if anomalies.get(position, {}).get('defective') else ''}")
    print(f"  Evaluated {len(rows)} positions in {time.time() - t0:.1f}s")

    # For JCM800 the integer positions are the FIT set; report them too, marked
    # as in-sample so the train-vs-withheld contrast can be made honestly.
    train_rows: list[dict] = []
    if not lopo:
        profile = build_profile_for_fold(spec, models, gt[PRIMARY_DI], dry, sr, spec.train_positions, cache, scope)
        anchor_models = {a.label: models[a.physical_position] for a in profile.anchors}
        runtime = ContinuousGainRuntime(profile, anchor_models)
        measured = [(p, grid[(anchor, p)]["input_gain_db"]) for p in spec.train_positions]
        slope = fit_direct_slope(measured, anchor)
        for position in spec.train_positions:
            gain_direct = slope * (position - anchor)
            gain_cal = float(_monotonic_interpolate([p for p, _ in measured], [g for _, g in measured], position))
            real = gt[PRIMARY_DI][position]
            recon = {
                "A_direct": render_with_input_gain(models[anchor], dry, sr, gain_direct),
                "A_calibrated": render_with_input_gain(models[anchor], dry, sr, gain_cal),
                "B_profile": runtime.render_at(position, dry, sr),
            }
            train_rows.append({
                "position": position, "withheld": False,
                "anchor_self_match": position == anchor,
                "gains": {"A_direct_db": gain_direct, "A_calibrated_db": gain_cal},
                "metrics": {PRIMARY_DI: {m: metrics(v, real) for m, v in recon.items()}},
            })
        fold_profiles["_full_train_profile"] = {
            "anchors": [a.label for a in profile.anchors],
            "anchor_positions": [a.physical_position for a in profile.anchors],
            "quality": profile.validation.quality,
            "worst_raw_esr_build_time": profile.validation.worst_raw_esr,
            "profile_json": profile.to_dict(),
        }

    return {
        "amp": spec.key, "name": spec.name, "control_label": spec.control_label,
        "notes": spec.notes, "split": "half_step_validation" if not lopo else "leave_one_position_out",
        "train_positions": spec.train_positions, "validation_positions": spec.validation_positions,
        "timing_defects": defective,
        "timing_diagnostics": {str(k): v for k, v in anomalies.items()},
        "measured_anchor_mapping": {str(p): grid[(anchor, p)] for p in spec.train_positions},
        "withheld_rows": rows, "train_rows": train_rows,
        "fold_profiles": {str(k): v for k, v in fold_profiles.items()},
    }


def maybe_write_wavs(spec: AmpSpec, position: float, real: np.ndarray, recon: dict[str, np.ndarray]) -> None:
    """Write listening material for representative positions (and, for the
    5150, level-matched versions, since its metrics are known unreliable)."""
    representative = {"jcm800": {1.5, 5.5, 9.5}, "supersonic": {2.0, 6.0, 9.0},
                      "twin": {2.0, 6.0, 9.0}, "peavey5150": {2.0, 4.0, 6.0, 8.0, 10.0}}
    if position not in representative.get(spec.key, set()):
        return
    out = OUT_DIR / "audio" / spec.key
    out.mkdir(parents=True, exist_ok=True)
    sr = 48000
    sf.write(out / f"p{position:g}_real.wav", real, sr, subtype="FLOAT")
    for name, audio in recon.items():
        sf.write(out / f"p{position:g}_{name}.wav", audio, sr, subtype="FLOAT")
        if spec.key == "peavey5150":
            n = min(len(audio), len(real))
            trim = rms_dbfs(real[:n]) - rms_dbfs(audio[:n])
            matched = audio[:n] * (10.0 ** (trim / 20.0))
            sf.write(out / f"p{position:g}_{name}_levelmatched.wav", matched, sr, subtype="FLOAT")


# --------------------------------------------------------------------------
# Runtime continuity / CPU / memory
# --------------------------------------------------------------------------

def continuity_experiment(spec: AmpSpec, cache: SearchCache) -> dict:
    """Automated 1 -> 10 -> 1 knob sweep, gradual and rapid, comparing the
    single G5 anchor against the multi-anchor profile at runtime.

    A continuous knob cannot be swept through an offline block renderer
    sample-by-sample, so the sweep is assembled the only honest offline way:
    render the whole clip at a fine grid of knob positions, then, per sample,
    linearly blend the two bracketing grid renders at the weight the knob's
    instantaneous position implies. Discontinuities from anchor switching
    survive this assembly intact (they live inside the rendered outputs and
    their per-sample weighting), which is exactly what is being measured."""
    print(f"\n=== Continuity / runtime: {spec.name} ===")
    dry, sr = load_di(PRIMARY_DI)
    mask = _active_mask(dry, sr, cgp.DEFAULT_BOUNDED_ENVELOPE_CONFIG)
    models = {p: load_nam(path) for p, path in spec.captures.items()}
    gt = render_ground_truth(spec, models, dry, sr)
    scope = f"{spec.key}|{PRIMARY_DI}"
    grid = build_search_grid(spec, models, gt, dry, sr, mask, cache, scope)

    anchor = ANCHOR_POSITION
    measured = [(p, grid[(anchor, p)]["input_gain_db"]) for p in spec.train_positions]
    xs, ys = [p for p, _ in measured], [g for _, g in measured]
    profile = build_profile_for_fold(spec, models, gt, dry, sr, spec.train_positions, cache, scope)
    anchor_models = {a.label: models[a.physical_position] for a in profile.anchors}
    runtime = ContinuousGainRuntime(profile, anchor_models)

    positions = [round(1.0 + 0.25 * i, 2) for i in range(37)]

    def timed(fn) -> tuple[np.ndarray, float, float, int]:
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.perf_counter()
        audio = fn()
        wall = time.perf_counter() - t0
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        return audio, wall, cpu, r1.ru_maxrss

    banks: dict[str, dict[float, np.ndarray]] = {"G5_only": {}, "multi_anchor": {}}
    cost: dict[str, dict] = {}
    for method in banks:
        walls, cpus, maxrss = [], [], 0
        for p in positions:
            if method == "G5_only":
                gain = float(_monotonic_interpolate(xs, ys, p))
                audio, w, c, rss = timed(lambda: render_with_input_gain(models[anchor], dry, sr, gain))
            else:
                audio, w, c, rss = timed(lambda: runtime.render_at(p, dry, sr))
            banks[method][p] = audio
            walls.append(w)
            cpus.append(c)
            maxrss = max(maxrss, rss)
        cost[method] = {
            "positions_rendered": len(positions),
            "mean_wall_s_per_position": float(np.mean(walls)),
            "max_wall_s_per_position": float(np.max(walls)),
            "total_child_cpu_s": float(np.sum(cpus)),
            "peak_child_rss_bytes": int(maxrss),
            "nam_models_resident": 1 if method == "G5_only" else len(anchor_models),
            "nam_model_bytes_resident": (spec.captures[anchor].stat().st_size if method == "G5_only"
                                         else sum(spec.captures[a.physical_position].stat().st_size
                                                  for a in profile.anchors)),
        }

    def sweep(method: str, traversals: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(dry)
        t = np.arange(n) / n
        # triangle wave: 1 -> 10 -> 1, repeated `traversals` times
        phase = (t * traversals) % 1.0
        knob = 1.0 + 9.0 * (1.0 - np.abs(2.0 * phase - 1.0))
        out = np.zeros(n, dtype=np.float64)
        arr = np.array(positions)
        idx = np.clip(np.searchsorted(arr, knob) - 1, 0, len(arr) - 2)
        lo = arr[idx]
        hi = arr[idx + 1]
        w = (knob - lo) / (hi - lo)
        for i in range(len(arr) - 1):
            sel = idx == i
            if not np.any(sel):
                continue
            a = banks[method][positions[i]][:n]
            b = banks[method][positions[i + 1]][:n]
            out[sel] = a[sel] * (1.0 - w[sel]) + b[sel] * w[sel]
        return out.astype(np.float32), knob

    def frame_analysis(audio: np.ndarray, frame: int = 480) -> dict:
        nf = len(audio) // frame
        frames = audio[: nf * frame].reshape(nf, frame)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-12))
        jumps = np.abs(np.diff(db))
        diffs = np.abs(np.diff(audio.astype(np.float64)))
        return {
            "max_frame_level_jump_db": float(np.max(jumps)) if jumps.size else 0.0,
            "mean_frame_level_jump_db": float(np.mean(jumps)) if jumps.size else 0.0,
            "frames_over_3db_jump": int(np.sum(jumps > 3.0)),
            "max_sample_delta": float(np.max(diffs)) if diffs.size else 0.0,
            "samples_over_0p25_delta": int(np.sum(diffs > 0.25)),
            "peak": float(np.max(np.abs(audio))),
            "rms_dbfs": float(rms_dbfs(audio)),
        }

    sweeps: dict[str, dict] = {}
    audio_dir = OUT_DIR / "audio" / f"{spec.key}_continuity"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for mode, traversals in (("gradual", 1.0), ("rapid", 4.0)):
        sweeps[mode] = {}
        for method in banks:
            audio, knob = sweep(method, traversals)
            sf.write(audio_dir / f"{mode}_{method}.wav", audio, sr, subtype="FLOAT")
            sweeps[mode][method] = frame_analysis(audio)
        a = sweeps[mode]["G5_only"]
        b = sweeps[mode]["multi_anchor"]
        sweeps[mode]["delta_multi_minus_g5"] = {
            k: b[k] - a[k] for k in ("max_frame_level_jump_db", "mean_frame_level_jump_db", "max_sample_delta")
        }

    # Static level/phase continuity across the position grid (anchor transitions)
    static = {}
    for method in banks:
        levels = [float(rms_dbfs(banks[method][p][: len(mask)][mask[: len(banks[method][p])]])) for p in positions]
        d = np.diff(levels)
        static[method] = {
            "levels_dbfs": levels,
            "max_step_db": float(np.max(np.abs(d))),
            "monotonic_increasing": bool(np.all(d > -0.05)),
            "negative_steps": [[positions[i], float(d[i])] for i in range(len(d)) if d[i] < -0.05],
        }
    boundaries = sorted({a.range_low for a in profile.anchors} | {a.range_high for a in profile.anchors})
    static["anchor_region_boundaries"] = boundaries
    static["anchor_regions"] = [[a.label, a.physical_position, a.range_low, a.range_high] for a in profile.anchors]

    # Phase-cancellation probe: inside each transition band, does the
    # crossfaded output dip below BOTH contributing anchors' own levels?
    cancellation = []
    for a in profile.anchors:
        for boundary in (a.range_low, a.range_high):
            near = [p for p in positions if abs(p - boundary) <= 0.5]
            for p in near:
                combined = banks["multi_anchor"][p]
                n = min(len(combined), len(mask))
                combined_db = rms_dbfs(combined[:n][mask[:n]])
                parts = []
                for other in profile.anchors:
                    reg = other.region()
                    part = render_with_input_gain(models[other.physical_position], dry, sr,
                                                  reg.input_gain_db(p))
                    parts.append(float(rms_dbfs(part[:n][mask[:n]])))
                cancellation.append({"position": p, "boundary": boundary,
                                     "combined_dbfs": float(combined_db),
                                     "contributing_anchor_dbfs": parts,
                                     "dips_below_all_contributors": bool(combined_db < min(parts) - 0.2)})
    seen = set()
    unique_cancellation = []
    for entry in cancellation:
        key = (entry["position"], entry["boundary"])
        if key in seen:
            continue
        seen.add(key)
        unique_cancellation.append(entry)

    return {
        "amp": spec.key, "grid_positions": positions,
        "profile_anchors": [a.label for a in profile.anchors],
        "runtime_cost": cost, "sweeps": sweeps, "static_level_continuity": static,
        "phase_cancellation_probe": unique_cancellation,
        "audio_dir": str(audio_dir),
    }


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amps", nargs="*", default=list(AMPS))
    parser.add_argument("--dis", nargs="*", default=list(DI_FILES))
    parser.add_argument("--skip-continuity", action="store_true")
    parser.add_argument("--continuity-amp", default="jcm800")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "results.json")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = SearchCache(OUT_DIR / "search_cache.json")

    started = time.time()
    results = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "eval_seconds": EVAL_SECONDS, "primary_di": PRIMARY_DI,
               "anchor_position": ANCHOR_POSITION, "amps": [], "continuity": None}

    for key in args.amps:
        spec = AMPS[key]
        t0 = time.time()
        res = evaluate_amp(spec, cache, args.dis)
        res["wall_clock_s"] = time.time() - t0
        results["amps"].append(res)
        args.out.write_text(json.dumps(results, indent=2))

    if not args.skip_continuity:
        t0 = time.time()
        results["continuity"] = continuity_experiment(AMPS[args.continuity_amp], cache)
        results["continuity"]["wall_clock_s"] = time.time() - t0

    results["total_wall_clock_s"] = time.time() - started
    results["search_cache"] = {"hits": cache.hits, "misses": cache.misses, "entries": len(cache.data)}
    cache.flush()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out} (total {results['total_wall_clock_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
