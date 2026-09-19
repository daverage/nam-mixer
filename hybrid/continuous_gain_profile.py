"""Continuous Gain PRODUCTION profile: offline builder + runtime engine.

See docs/CONTINUOUS_GAIN_PRODUCTION.md for the full design writeup and
docs/CONTINUOUS_GAIN_UI.md's "Turn the current Continuous Gain research into
a production-ready implementation" section for the requirements this module
implements.

This is deliberately a SEPARATE module from `hybrid.continuous_gain`
(Phase 1/2 research harness -- ground truth rendering, discrete
nearest-neighbour interpolation, leave-one-out validation) rather than a
rewrite of it: the production path reuses that module's `GainCapture`/
`GainCaptureSet`/`run_ground_truth_harness`/`detect_capture_anomalies`/
`interpolate_output` for its "discrete interpolation" fallback strategy (see
`ContinuousGainProfile.strategy`), and only adds what those don't cover --
virtual (pre-model) input-gain anchors, automatic anchor selection, a fitted
nonlinear physical-knob -> input-gain mapping, anchor regions/transitions,
and a serializable profile.

Two costs, matching the project's existing "expensive render / cheap reblend"
split (see hybrid/pipeline.py and CLAUDE.md "Design modes"):

- **Profile building** (`build_profile`) is EXPENSIVE: it renders every
  training capture many times over a virtual-gain search grid.  This only
  happens once, offline, when a Continuous Gain profile is created.
- **Runtime evaluation** (`ContinuousGainRuntime.render_at`) is CHEAP: one
  (or, inside a transition band, two) NAM render(s) at an already-known input
  gain, no search, no training capture other than the selected anchor(s).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio_metrics import rms_dbfs
from .coverage import ACTIVE_SIGNAL_THRESHOLD_DBFS, active_signal_mask
from .envelope import DEFAULT_BOUNDED_ENVELOPE_CONFIG, BoundedEnvelopeConfig, bounded_causal_envelope_db
from .nam_loader import NamModel
from .render import render
from .validation import compute_esr_metrics

PROFILE_SCHEMA_VERSION = 1

# Quality thresholds referenced against the measured research distributions
# in docs/CONTINUOUS_GAIN_PHASE3.md (Lo channel: raw ESR 0.003-0.03 with tight
# bracketing; Hi channel knee: 0.09-0.27 when badly bracketed, <0.01 once
# densely bracketed) and docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md's
# virtual-gain anchor results -- NOT arbitrary round numbers.
QUALITY_VALIDATED_MAX_ESR = 0.015
QUALITY_ACCEPTABLE_MAX_ESR = 0.06


class ProfileQuality:
    VALIDATED = "validated"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


def classify_quality(worst_raw_esr: float) -> str:
    if not np.isfinite(worst_raw_esr):
        return ProfileQuality.POOR
    if worst_raw_esr <= QUALITY_VALIDATED_MAX_ESR:
        return ProfileQuality.VALIDATED
    if worst_raw_esr <= QUALITY_ACCEPTABLE_MAX_ESR:
        return ProfileQuality.ACCEPTABLE
    return ProfileQuality.POOR


def render_with_input_gain(
    model: NamModel, dry: np.ndarray, sample_rate: int, input_gain_db: float,
) -> np.ndarray:
    """Apply ordinary pre-model input gain (dB) to `dry`, then render --
    exactly the signal path of a real NAM player's "Input Gain" knob:
    `DI --(* 10**(input_gain_db/20))--> NAM --> output`. Shared by the
    virtual-gain search below and by
    `scripts/continuous_gain_virtual_gain_benchmark.py` so the research
    benchmark and the production builder can never silently diverge on what
    "virtual input gain" means.
    """
    dry = np.asarray(dry, dtype=np.float32)
    scaled = dry * np.float32(10.0 ** (input_gain_db / 20.0)) if input_gain_db else dry
    return render(model, scaled.astype(np.float32), sample_rate)


def _active_mask(dry: np.ndarray, sample_rate: int, envelope_config: BoundedEnvelopeConfig) -> np.ndarray:
    envelope_db = bounded_causal_envelope_db(np.asarray(dry, dtype=np.float32), sample_rate, envelope_config)
    return active_signal_mask(envelope_db, ACTIVE_SIGNAL_THRESHOLD_DBFS)


@dataclass
class GainSearchResult:
    """One virtual-gain search outcome: the best input gain (dB) found for
    `anchor_model` to reproduce `target_output`, plus the resulting error and
    whether the search bottomed out against its own search bounds (doc:
    "detect search-boundary failures")."""

    input_gain_db: float
    raw_esr: float
    hit_lower_bound: bool
    hit_upper_bound: bool


def search_virtual_input_gain(
    anchor_model: NamModel,
    dry: np.ndarray,
    sample_rate: int,
    target_output: np.ndarray,
    mask: np.ndarray,
    *,
    coarse_range_db: tuple[float, float] = (-24.0, 24.0),
    coarse_step_db: float = 3.0,
    fine_step_db: float = 0.5,
    finest_step_db: float = 0.1,
) -> GainSearchResult:
    """Coarse-then-fine search (doc requirement 3: "use coarse then fine
    optimisation") for the input gain that makes `anchor_model` best
    reproduce `target_output` over the DI's ACTIVE material (reusing the
    same active-signal convention as `hybrid.coverage`/
    `hybrid.fixed_blend.compute_active_trim`, not overall/silence-diluted
    loudness).

    Level and shape are optimised together (raw ESR, not gain-normalized) --
    unlike `hybrid.continuous_gain.interpolate_output_level_matched`, this
    has no oracle: the search only ever compares AGAINST the real target
    capture it is trying to match, which is exactly what a profile build
    (offline, with the real captures in hand) is allowed to do.
    """
    n = min(len(target_output), len(mask))
    target = target_output[:n]
    active_mask = mask[:n]

    def esr_at(gain_db: float) -> float:
        candidate = render_with_input_gain(anchor_model, dry, sample_rate, gain_db)
        m = min(len(candidate), n)
        metrics = compute_esr_metrics(candidate[:m][active_mask[:m]], target[active_mask[:m]])
        return metrics["raw_esr"]

    lo, hi = coarse_range_db
    grid = np.arange(lo, hi + 1e-9, coarse_step_db)
    scores = {float(g): esr_at(float(g)) for g in grid}
    best_db = min(scores, key=scores.get)
    hit_lower = best_db <= lo + 1e-9
    hit_upper = best_db >= hi - 1e-9

    for step in (fine_step_db, finest_step_db):
        window = np.arange(best_db - coarse_step_db, best_db + coarse_step_db + 1e-9, step)
        window = np.clip(window, lo, hi)
        for g in window:
            g = float(g)
            if g not in scores:
                scores[g] = esr_at(g)
        best_db = min(scores, key=scores.get)
        coarse_step_db = step

    return GainSearchResult(
        input_gain_db=best_db,
        raw_esr=scores[best_db],
        hit_lower_bound=hit_lower,
        hit_upper_bound=hit_upper,
    )


@dataclass
class TrainingTarget:
    """One real training capture prepared as a virtual-gain-search target."""

    label: str
    physical_position: float
    model: NamModel
    output: np.ndarray


@dataclass
class AnchorCandidateResult:
    """Per-target search outcome when `anchor_label` is evaluated as (one of)
    the profile's anchors."""

    anchor_label: str
    target_label: str
    input_gain_db: float
    raw_esr: float
    hit_lower_bound: bool
    hit_upper_bound: bool


def _search_grid(
    anchors: list[TrainingTarget], targets: list[TrainingTarget], dry: np.ndarray, sample_rate: int, mask: np.ndarray,
) -> dict[tuple[str, str], AnchorCandidateResult]:
    """Search every (anchor, target) pair once. Cached/shared across every
    candidate anchor SET so combinatorics over which subset of anchors to use
    doesn't re-render audio that's already been measured -- see the doc's
    "avoid exhaustive combinatorial searches if they become expensive"."""
    grid: dict[tuple[str, str], AnchorCandidateResult] = {}
    for anchor in anchors:
        for target in targets:
            if anchor.label == target.label:
                grid[(anchor.label, target.label)] = AnchorCandidateResult(
                    anchor_label=anchor.label, target_label=target.label,
                    input_gain_db=0.0, raw_esr=0.0, hit_lower_bound=False, hit_upper_bound=False,
                )
                continue
            result = search_virtual_input_gain(anchor.model, dry, sample_rate, target.output, mask)
            grid[(anchor.label, target.label)] = AnchorCandidateResult(
                anchor_label=anchor.label, target_label=target.label,
                input_gain_db=result.input_gain_db, raw_esr=result.raw_esr,
                hit_lower_bound=result.hit_lower_bound, hit_upper_bound=result.hit_upper_bound,
            )
    return grid


def _best_anchor_for_target(
    anchor_set: tuple[str, ...], target_label: str, grid: dict[tuple[str, str], AnchorCandidateResult],
) -> AnchorCandidateResult:
    candidates = [grid[(a, target_label)] for a in anchor_set]
    return min(candidates, key=lambda c: c.raw_esr)


def _candidate_anchor_sets(labels: list[str], size: int, max_combinations: int = 60) -> list[tuple[str, ...]]:
    """Candidate anchor sets of a given size. Exhaustive (doc: "the exact
    exhaustive version has been validated on small capture sets") whenever
    the combination count stays bounded; otherwise falls back to a spread of
    evenly-indexed subsets across the sorted label order so build time stays
    predictable on a 10-capture set (`C(10,3)=120`, still small -- the cap
    exists for future-proofing rather than because 10 captures needs it)."""
    all_combos = list(itertools.combinations(labels, size))
    if len(all_combos) <= max_combinations:
        return all_combos
    n = len(labels)
    step = max(1, n // max_combinations)
    return list(itertools.combinations(labels[::step] or labels, size))[:max_combinations]


@dataclass
class AnchorRegion:
    """One anchor's ownership of a physical-Gain sub-range, plus its fitted
    mapping curve -- doc requirement 5 ("Anchor regions") and 4 ("Derive the
    continuous control mapping")."""

    anchor_label: str
    physical_position: float
    range_low: float
    range_high: float
    # Sorted (physical_position, input_gain_db) measured points this
    # anchor's region mapping was fit from -- kept so the production mapping
    # can be validated against the measurements it was derived from (doc
    # requirement 4: "Preserve the measured points").
    mapping_points: list[tuple[float, float]]

    def input_gain_db(self, physical_position: float) -> float:
        xs = [p for p, _ in self.mapping_points]
        ys = [g for _, g in self.mapping_points]
        if len(xs) == 1:
            return ys[0]
        return float(_monotonic_interpolate(xs, ys, physical_position))


def _monotonic_interpolate(xs: list[float], ys: list[float], x: float) -> float:
    """Monotonic PCHIP fit (doc requirement 4: "Prefer a stable interpolation
    such as monotonic PCHIP / monotonic spline... Do not allow the fitted
    curve to introduce large overshoot"). Falls back to linear for exactly
    two points (PCHIP degrades to the same result there anyway) and clamps
    extrapolation outside the measured range rather than letting a spline
    overshoot beyond it.
    """
    order = np.argsort(xs)
    xs_sorted = np.asarray(xs, dtype=np.float64)[order]
    ys_sorted = np.asarray(ys, dtype=np.float64)[order]
    if x <= xs_sorted[0]:
        return float(ys_sorted[0])
    if x >= xs_sorted[-1]:
        return float(ys_sorted[-1])
    from scipy.interpolate import PchipInterpolator

    curve = PchipInterpolator(xs_sorted, ys_sorted, extrapolate=False)
    value = curve(x)
    return float(value)


@dataclass
class ContinuousGainAnchor:
    """One selected anchor, as stored in a `ContinuousGainProfile` -- schema
    field group per doc requirement 5 ("Store: anchor NAM, anchor physical
    Gain position, valid physical Gain range, mapping curve, confidence/
    validation metrics")."""

    label: str
    nam_path: str
    physical_position: float
    range_low: float
    range_high: float
    mapping_points: list[tuple[float, float]]
    worst_raw_esr_in_region: float

    def region(self) -> AnchorRegion:
        return AnchorRegion(
            anchor_label=self.label, physical_position=self.physical_position,
            range_low=self.range_low, range_high=self.range_high,
            mapping_points=self.mapping_points,
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label, "nam_path": self.nam_path,
            "physical_position": self.physical_position,
            "range": [self.range_low, self.range_high],
            "mapping": [[p, g] for p, g in self.mapping_points],
            "worst_raw_esr_in_region": self.worst_raw_esr_in_region,
        }

    @staticmethod
    def from_dict(d: dict) -> "ContinuousGainAnchor":
        lo, hi = d["range"]
        return ContinuousGainAnchor(
            label=d["label"], nam_path=d["nam_path"], physical_position=d["physical_position"],
            range_low=lo, range_high=hi,
            mapping_points=[(p, g) for p, g in d["mapping"]],
            worst_raw_esr_in_region=d["worst_raw_esr_in_region"],
        )


@dataclass
class ProfileValidationReport:
    per_target_raw_esr: dict[str, float]
    worst_raw_esr: float
    quality: str
    problem_regions: list[str]
    recommended_capture_position: Optional[float]

    def to_dict(self) -> dict:
        return {
            "per_target_raw_esr": self.per_target_raw_esr,
            "worst_raw_esr": self.worst_raw_esr,
            "quality": self.quality,
            "problem_regions": self.problem_regions,
            "recommended_capture_position": self.recommended_capture_position,
        }

    @staticmethod
    def from_dict(d: dict) -> "ProfileValidationReport":
        return ProfileValidationReport(**d)


# Transition crossfade width in NORMALIZED (0..1 fraction of the anchor
# ownership range) units around a region boundary -- doc requirement 6:
# "short crossfade between the outgoing and incoming anchor outputs". Kept
# small since research shows anchor regions are chosen for good local
# accuracy; a wide crossfade would blend in a range where one of the two
# anchors is already known to fit poorly.
DEFAULT_TRANSITION_WIDTH = 0.08


@dataclass
class ContinuousGainProfile:
    """Serializable Continuous Gain profile -- doc requirement 8 ("Profile
    format"). `strategy` selects the runtime path (doc requirement 15:
    "Continuous Gain profiles should be able to use virtual-gain anchor mode
    or fall back to discrete interpolation")."""

    version: int
    control_name: str
    control_min: float
    control_max: float
    anchors: list[ContinuousGainAnchor]
    validation: ProfileValidationReport
    strategy: str = "virtual_gain_anchor"  # or "discrete_interpolation"
    transition_width: float = DEFAULT_TRANSITION_WIDTH

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "control": {"name": self.control_name, "min": self.control_min, "max": self.control_max},
            "strategy": self.strategy,
            "transition_width": self.transition_width,
            "anchors": [a.to_dict() for a in sorted(self.anchors, key=lambda a: a.physical_position)],
            "validation": self.validation.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "ContinuousGainProfile":
        if d.get("version") != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Continuous Gain profile schema version {d.get('version')!r} "
                f"(expected {PROFILE_SCHEMA_VERSION})."
            )
        control = d["control"]
        return ContinuousGainProfile(
            version=d["version"],
            control_name=control["name"], control_min=control["min"], control_max=control["max"],
            anchors=[ContinuousGainAnchor.from_dict(a) for a in d["anchors"]],
            validation=ProfileValidationReport.from_dict(d["validation"]),
            strategy=d.get("strategy", "virtual_gain_anchor"),
            transition_width=d.get("transition_width", DEFAULT_TRANSITION_WIDTH),
        )


def build_profile(
    training: list[TrainingTarget],
    dry: np.ndarray,
    sample_rate: int,
    *,
    control_name: str = "Gain",
    max_anchors: int = 3,
    envelope_config: BoundedEnvelopeConfig = DEFAULT_BOUNDED_ENVELOPE_CONFIG,
    nam_paths: Optional[dict[str, str]] = None,
) -> ContinuousGainProfile:
    """Offline profile builder -- doc requirements 3/4/5/9/10 in one pass:

    1. try anchor-set sizes 1, 2, ... up to `max_anchors`
    2. for each size, exhaustively (bounded) try candidate anchor sets
    3. for each candidate set, every training capture is reconstructed by
       whichever anchor in the set fits it best (own virtual input gain)
    4. accept the SMALLEST anchor-set size whose worst-case raw ESR across
       ALL training targets is at least "acceptable" (`classify_quality`)
    5. if no size up to `max_anchors` reaches "acceptable", still return the
       best `max_anchors`-anchor profile found, marked POOR, with the
       highest-error physical-Gain region flagged for an additional capture
       (doc requirement 9: "recommend an additional capture in that region")

    `training` must have at least 1 entry (a single anchor is a degenerate
    but valid profile); realistic use has >= 2. Positions are the REAL
    physical knob values recorded when each capture was taken -- never
    inferred from output level (doc requirement 2).
    """
    if not training:
        raise ValueError("build_profile requires at least one training capture.")
    training = sorted(training, key=lambda t: t.physical_position)
    labels = [t.label for t in training]
    by_label = {t.label: t for t in training}
    mask = _active_mask(dry, sample_rate, envelope_config)

    grid = _search_grid(training, training, dry, sample_rate, mask)

    best_profile: Optional[ContinuousGainProfile] = None
    best_quality_rank = -1
    quality_rank = {ProfileQuality.POOR: 0, ProfileQuality.ACCEPTABLE: 1, ProfileQuality.VALIDATED: 2}

    max_size = min(max_anchors, len(labels))
    for size in range(1, max_size + 1):
        best_for_size: Optional[tuple[float, tuple[str, ...], dict[str, AnchorCandidateResult]]] = None
        for anchor_set in _candidate_anchor_sets(labels, size):
            assignment = {t.label: _best_anchor_for_target(anchor_set, t.label, grid) for t in training}
            worst = max(r.raw_esr for r in assignment.values())
            if best_for_size is None or worst < best_for_size[0]:
                best_for_size = (worst, anchor_set, assignment)

        worst, anchor_set, assignment = best_for_size
        quality = classify_quality(worst)
        profile = _assemble_profile(
            anchor_set, assignment, training, by_label, control_name, quality, nam_paths or {},
        )
        if quality_rank[quality] > best_quality_rank:
            best_profile, best_quality_rank = profile, quality_rank[quality]
        if quality in (ProfileQuality.ACCEPTABLE, ProfileQuality.VALIDATED):
            return profile  # doc requirement 10: smallest set that passes wins immediately

    assert best_profile is not None
    return best_profile


def _assemble_profile(
    anchor_set: tuple[str, ...],
    assignment: dict[str, AnchorCandidateResult],
    training: list[TrainingTarget],
    by_label: dict[str, TrainingTarget],
    control_name: str,
    quality: str,
    nam_paths: dict[str, str],
) -> ContinuousGainProfile:
    # Region ownership: which contiguous physical-position span each anchor
    # covers, based on which anchor best reconstructs each target (doc
    # requirement 5: "based on measured reconstruction quality", not knob
    # midpoint).
    ordered_targets = sorted(training, key=lambda t: t.physical_position)
    owner_by_position = [(t.physical_position, assignment[t.label].anchor_label) for t in ordered_targets]

    anchors_by_label: dict[str, list[tuple[float, float]]] = {a: [] for a in anchor_set}
    worst_by_label: dict[str, float] = {a: 0.0 for a in anchor_set}
    for t in ordered_targets:
        result = assignment[t.label]
        anchors_by_label[result.anchor_label].append((t.physical_position, result.input_gain_db))
        worst_by_label[result.anchor_label] = max(worst_by_label[result.anchor_label], result.raw_esr)

    positions = [p for p, _ in owner_by_position]
    lo_bound, hi_bound = min(positions), max(positions)

    anchors: list[ContinuousGainAnchor] = []
    for label in anchor_set:
        points = sorted(anchors_by_label[label]) or [(by_label[label].physical_position, 0.0)]
        owned_positions = [p for p, owner in owner_by_position if owner == label]
        range_low = min(owned_positions) if owned_positions else by_label[label].physical_position
        range_high = max(owned_positions) if owned_positions else by_label[label].physical_position
        # Extend the outermost anchors to the full control range so runtime
        # lookup never falls outside every anchor's region.
        if range_low == min(p for p, _ in owner_by_position):
            range_low = lo_bound
        if range_high == max(p for p, _ in owner_by_position):
            range_high = hi_bound
        anchors.append(ContinuousGainAnchor(
            label=label, nam_path=nam_paths.get(label, ""), physical_position=by_label[label].physical_position,
            range_low=range_low, range_high=range_high, mapping_points=points,
            worst_raw_esr_in_region=worst_by_label[label],
        ))

    worst_overall = max(r.raw_esr for r in assignment.values())
    per_target = {label: assignment[label].raw_esr for label in assignment}
    problem_regions = []
    recommended: Optional[float] = None
    if quality == ProfileQuality.POOR:
        worst_target_label = max(assignment, key=lambda label: assignment[label].raw_esr)
        worst_position = by_label[worst_target_label].physical_position
        problem_regions.append(
            f"Physical Gain near {worst_position:g} reconstructs poorly "
            f"(raw ESR {assignment[worst_target_label].raw_esr:.4f}) with the current anchor set."
        )
        recommended = worst_position

    validation = ProfileValidationReport(
        per_target_raw_esr=per_target, worst_raw_esr=worst_overall, quality=quality,
        problem_regions=problem_regions, recommended_capture_position=recommended,
    )
    return ContinuousGainProfile(
        version=PROFILE_SCHEMA_VERSION, control_name=control_name,
        control_min=lo_bound, control_max=hi_bound,
        anchors=anchors, validation=validation,
    )


class ContinuousGainRuntime:
    """Runtime engine: physical Gain knob -> anchor selection -> mapping
    lookup -> pre-NAM input gain -> render -> optional transition crossfade
    (doc requirement 7). No searches, comparisons, or training here -- every
    anchor's NAM model must already be loaded by the caller (doc requirement
    13: "avoid keeping unnecessary NAM models loaded simultaneously outside
    transition regions" is the caller's responsibility, e.g. lazy-loading
    only the anchor(s) needed for the current knob position).
    """

    def __init__(self, profile: ContinuousGainProfile, anchor_models: dict[str, NamModel]):
        missing = [a.label for a in profile.anchors if a.label not in anchor_models]
        if missing:
            raise ValueError(f"Missing NAM model(s) for anchor(s): {missing}")
        self.profile = profile
        self.anchor_models = anchor_models
        self._regions = sorted(
            [a.region() for a in profile.anchors], key=lambda r: r.range_low,
        )

    def _clamp_position(self, position: float) -> float:
        return float(np.clip(position, self.profile.control_min, self.profile.control_max))

    def _region_for(self, position: float) -> AnchorRegion:
        for region in self._regions:
            if region.range_low <= position <= region.range_high:
                return region
        # Outside every region only happens via floating point edges after
        # clamping; fall back to the nearest region rather than raising.
        return min(self._regions, key=lambda r: min(abs(position - r.range_low), abs(position - r.range_high)))

    def _neighbor_region(self, region: AnchorRegion, position: float) -> Optional[AnchorRegion]:
        idx = self._regions.index(region)
        if position <= region.range_low and idx > 0:
            return self._regions[idx - 1]
        if position >= region.range_high and idx + 1 < len(self._regions):
            return self._regions[idx + 1]
        return None

    def render_at(self, physical_position: float, dry: np.ndarray, sample_rate: int) -> np.ndarray:
        """Render `dry` through the profile at `physical_position`, crossfading
        across an anchor boundary within `transition_width` of it (fraction
        of that region's own span) rather than clicking between anchors."""
        position = self._clamp_position(physical_position)
        region = self._region_for(position)
        span = max(region.range_high - region.range_low, 1e-9)
        half_width = self.profile.transition_width * span / 2.0

        primary_output = render_with_input_gain(
            self.anchor_models[region.anchor_label], dry, sample_rate, region.input_gain_db(position),
        )

        near_low = position - region.range_low <= half_width
        near_high = region.range_high - position <= half_width
        other_region = None
        if near_low:
            other_region = self._neighbor_region(region, region.range_low)
            boundary = region.range_low
            distance = position - boundary
        elif near_high:
            other_region = self._neighbor_region(region, region.range_high)
            boundary = region.range_high
            distance = boundary - position

        if other_region is None:
            return primary_output

        other_output = render_with_input_gain(
            self.anchor_models[other_region.anchor_label], dry, sample_rate, other_region.input_gain_db(position),
        )
        n = min(len(primary_output), len(other_output))
        # 0 at the boundary (other anchor's own capture point of view is
        # irrelevant here) -> 1 a half-width away (fully this region's own
        # anchor). Smoothstep avoids a linear-crossfade "comb" edge at the
        # boundary (doc requirement 6: test for audible comb filtering).
        t = float(np.clip(distance / half_width, 0.0, 1.0)) if half_width > 0 else 1.0
        weight_primary = t * t * (3.0 - 2.0 * t)
        return primary_output[:n] * weight_primary + other_output[:n] * (1.0 - weight_primary)
