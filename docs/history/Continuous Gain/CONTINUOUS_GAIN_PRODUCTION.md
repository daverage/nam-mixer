# Continuous Gain -- Production Architecture

This turns the Continuous Gain research (docs/CONTINUOUS_GAIN.md,
docs/CONTINUOUS_GAIN_PHASE3.md, docs/CONTINUOUS_GAIN_VIRTUAL_GAIN_BENCHMARK.md)
into a production foundation: `hybrid/continuous_gain_profile.py`.

## Principle

> Use the smallest number of real NAM anchors that can reproduce the
> measured physical Gain sweep within a validated quality threshold, using
> ordinary pre-model NAM input gain and a measured nonlinear control mapping.

This is deliberately NOT a rewrite of `hybrid/continuous_gain.py` (the
Phase 1/2 research harness). That module's `GainCapture`/`GainCaptureSet`/
`run_ground_truth_harness`/`detect_capture_anomalies`/`interpolate_output`
remain the "discrete interpolation" fallback strategy a
`ContinuousGainProfile` can select (`strategy="discrete_interpolation"`);
the new module adds the virtual-input-gain anchor path on top.

## Two costs

- **Profile building** (`build_profile`) is EXPENSIVE -- it renders every
  training capture through a virtual-gain search grid, against every
  candidate anchor. This happens once, offline, per capture set.
- **Runtime evaluation** (`ContinuousGainRuntime.render_at`) is CHEAP -- one
  render at an already-known input gain (two, briefly, inside an anchor
  transition band). No search, comparison, training, or extra NAM models
  beyond the anchors involved.

## Architecture

```
multiple real NAM captures (GainCapture-style: model + physical position)
        |
        v
build_profile()
  1. for anchor-set size = 1, 2, ... up to max_anchors:
       for every candidate anchor set of that size (exhaustive, bounded --
       see _candidate_anchor_sets):
         for every training capture (target):
           search_virtual_input_gain() against EVERY anchor in the set,
           keep whichever anchor fits best
         worst-case raw ESR across all targets -> classify_quality()
       keep the best (smallest raw-ESR) anchor set at this size
     return the first size whose quality is >= "acceptable"
     (else return the best `max_anchors`-anchor result found, marked "poor")
        |
        v
_assemble_profile(): region ownership (doc requirement 5) + per-anchor
monotonic PCHIP mapping (physical position -> input gain dB) fit only from
the anchor's OWN measured points
        |
        v
ContinuousGainProfile (schema v1, JSON-serializable)
        |
        v
ContinuousGainRuntime(profile, anchor_models)
  physical Gain knob -> region lookup -> mapping.input_gain_db(position)
    -> render_with_input_gain() -> optional smoothstep crossfade with the
       neighbouring anchor's region near a boundary
```

## Profile schema (v1)

```json
{
  "version": 1,
  "control": {"name": "Gain", "min": 1.0, "max": 10.0},
  "strategy": "virtual_gain_anchor",
  "transition_width": 0.08,
  "anchors": [
    {
      "label": "g5",
      "nam_path": ".../jcm800-high-g5.0-11.4dBu.nam",
      "physical_position": 5.0,
      "range": [3.5, 7.0],
      "mapping": [[3.5, -2.1], [4.0, -1.4], [5.0, 0.0], [6.0, 1.8], [7.0, 3.9]],
      "worst_raw_esr_in_region": 0.021
    }
  ],
  "validation": {
    "per_target_raw_esr": {"g1": 0.30, "g3": 0.04, "g5": 0.0, "g7": 0.03, "g10": 0.22},
    "worst_raw_esr": 0.30,
    "quality": "poor",
    "problem_regions": ["Physical Gain near 1 reconstructs poorly (raw ESR 0.3000) with the current anchor set."],
    "recommended_capture_position": 1.0
  }
}
```

`mapping` always includes the anchor's own physical position with 0 dB
implicitly reproduced (the anchor mapped to itself needs no virtual gain --
`_search_grid` short-circuits that pair to `input_gain_db=0.0, raw_esr=0.0`).
`ContinuousGainAnchor.region().input_gain_db()` clamps outside the measured
mapping points rather than extrapolating a spline past them (doc requirement
4: "Do not allow the fitted curve to introduce large overshoot").

## Quality thresholds

`QUALITY_VALIDATED_MAX_ESR = 0.015` and `QUALITY_ACCEPTABLE_MAX_ESR = 0.06`
are set from the measured research distributions in
docs/CONTINUOUS_GAIN_PHASE3.md (Lo-channel raw ESR 0.003-0.03 with
reasonable bracketing; Hi-channel drops from >0.05 to <0.01 once bracketed
tightly across the clean-to-breakup knee) -- not arbitrary round numbers.
Anything above 0.06 is "poor": the profile is still returned (never
withheld), but flagged, with the worst-reconstructing physical-Gain position
surfaced as `recommended_capture_position` so the UI can prompt for exactly
one additional real capture there (doc requirement 9).

## Anchor selection and regions

`build_profile` tries anchor-set sizes 1..`max_anchors` (default 3),
exhaustively for small capture counts (`_candidate_anchor_sets` caps
combinatorics at 60 candidates and falls back to an evenly-spread subset
above that, per the doc's "avoid exhaustive combinatorial searches if they
become expensive" once the exhaustive version is validated on small sets).
It stops at the smallest anchor-set size that reaches "acceptable" or
better (doc requirement 10: minimum-capture discovery), never forcing a
larger set than needed.

Region ownership is decided per training target by which anchor in the
winning set best reconstructed it (`_best_anchor_for_target`), NOT by knob
midpoint (doc requirement 5). The outermost anchors' regions are extended to
the full control range so every physical position within `[control_min,
control_max]` has an owning anchor.

## Runtime and transitions

`ContinuousGainRuntime.render_at` looks up the owning region, computes that
anchor's mapped input gain via the per-anchor PCHIP curve, and renders once.
Within `transition_width` (default 8% of the region's own span) of a region
boundary, it also renders through the NEIGHBOURING anchor (with ITS OWN
mapped gain at that position) and smoothstep-crossfades between the two --
doc requirement 6. Both anchors always receive their own correctly-mapped
gain during a transition; nothing is crossfaded at a mismatched level.
`test_runtime_transition_has_no_level_discontinuity` in
tests/test_continuous_gain_profile.py checks the RMS either side of a
boundary doesn't jump.

## Fallback compatibility

`ContinuousGainProfile.strategy` reserves `"discrete_interpolation"` as an
explicit alternate value pointing back at
`hybrid.continuous_gain.interpolate_output` for capture sets where the
virtual-gain search cannot reach an acceptable quality even at
`max_anchors` -- this module does not remove or alter that path (doc
requirement 15). Wiring `ContinuousGainRuntime` to actually dispatch on
`strategy` is left for the UI/runtime integration step described in
docs/CONTINUOUS_GAIN_UI.md, once a discrete-interpolation profile format is
finalized; today `strategy` is stored but only `"virtual_gain_anchor"` has a
runtime implementation.

## Tests

`tests/test_continuous_gain_profile.py` uses the same fake-render
convention as `tests/test_continuous_gain.py` (a `model.raw["gain"]`-driven
tanh saturation curve) -- no native tool or real captures required. It
covers:

- `render_with_input_gain` applies gain before rendering, not after
- `search_virtual_input_gain` finds ~0 dB / ~0 ESR when an anchor matches
  its own target exactly
- quality classification thresholds
- degenerate single-capture profiles
- anchor-set-size selection and full region coverage with no gaps
- a poor-quality profile flags a recommendation
- schema round-trip through `to_dict`/`from_dict`, and version rejection
- runtime clamping of out-of-range physical positions
- runtime raising on a missing anchor model
- transition crossfade continuity

## Real-data regression benchmark

`scripts/continuous_gain_profile_builder.py` builds a profile from the same
dense "Marshall JCM800 2203 - updated" integer-gain captures (G1-G10) used
by `scripts/continuous_gain_virtual_gain_benchmark.py`, and validates it
against the genuinely-real withheld half-step captures (G1.5, G2.5, ...,
G9.5) -- not part of the automated `pytest` suite (it shells out to the real
native `nam_render` tool against a user-local capture directory, the same
convention as `tests/test_render.py`'s auto-skip). Run it directly:

```bash
python scripts/continuous_gain_profile_builder.py --max-anchors 3
```

### Result (2026-09-19 run, JCM800 2203 "updated" dataset, High channel, 10
integer-gain training captures G1-G10, `moderate_brit.wav` DI, `max_anchors=3`)

```
Selected 3 anchor(s): g1, g6, g9
Build-time quality: acceptable (worst raw ESR 0.0528, at target g8)
  g1: range [1, 1]   (degenerate -- the only target it best reconstructs is itself)
  g6: range [2, 7]   worst ESR in region 0.0468
  g9: range [8, 10]  worst ESR in region 0.0528

Held-out half-step regression (raw ESR):
  G1.5  0.0951      G2.5  0.0397      G3.5  0.0191
  G4.5  0.0081      G5.5  0.0031      G6.5  0.0223
  G7.5  0.1146      G8.5  2.1100      G9.5  0.1489
mean 0.2845, worst 2.1100 (at G8.5)
```

Automatic anchor selection landed on a 3-anchor set close to (though not
identical to) the earlier ad hoc virtual-gain benchmark's G1/G5/G10 --
picking G6 instead of G5 and G9 instead of G10 as the mid/high anchors,
which the builder's own per-target search found fit this DI/dataset
combination marginally better. Mid-range held-out reconstruction (G2.5-G6.5)
is strong (raw ESR 0.003-0.04, "validated"-tier), consistent with the
research harness's Lo-channel-like results once anchors are reasonably
placed.

**G8.5's raw ESR of 2.11 is a genuine finding, not an implementation bug** --
verified directly: the reconstruction's RMS level matches the real capture
almost exactly (-20.03 dBFS reconstructed vs. -19.98 dBFS real), so the
error is purely waveform/harmonic shape, the same failure mode
docs/CONTINUOUS_GAIN_PHASE3.md's HF-correction ablation already identified
(a magnitude-only fix cannot recover it). This particular amp's response
between G8 and G9 changes character sharply enough that neither anchor's
own gain-shifted curve tracks it -- exactly the "extreme anchors can fail
outside their useful range" caveat the UI spec calls out. It argues for
either an additional real anchor near G8.5-G9 for this specific unit, or
(automatically) a smaller anchor-ownership range around G9 -- a candidate
follow-up refinement to `_assemble_profile`'s region-boundary logic, not
yet implemented.

The important qualitative result to check on every re-run:

1. How many anchors were selected (the doc's minimum-capture-discovery
   question).
2. The reported build-time `quality` (here: "acceptable").
3. Whether any held-out region spikes sharply above its neighbours the way
   G8.5 did here -- that is the "problem region" signal the UI should
   surface, even though the CURRENT builder only checks TRAINING-capture
   error at build time, not held-out half-steps (see "Remaining
   experimental" below).

## What was implemented

- Production architecture and profile schema (v1), versioned, serializable.
- Offline profile builder with automatic anchor selection (1..N, smallest
  passing set wins) and per-anchor nonlinear (PCHIP) control mapping.
- Region ownership by measured reconstruction quality, not knob midpoint.
- Runtime anchor/mapping engine with smoothstep anchor-transition
  crossfading, matched-gain on both sides of a transition.
- Fallback strategy field reserved for discrete interpolation (not yet
  wired to a runtime dispatch).
- Automated tests against a fake render() (no native tool required).
- Real-data regression benchmark script against the JCM800 dataset.

## Remaining experimental / not yet done

- `strategy="discrete_interpolation"` is stored but has no runtime dispatch
  yet -- `ContinuousGainRuntime` only implements the virtual-gain-anchor
  path. Wiring the fallback requires deciding how a discrete-interpolation
  profile's own capture set is stored alongside a `ContinuousGainProfile`.
- No UI wiring yet (docs/CONTINUOUS_GAIN_UI.md's tab/flow is a separate,
  larger deliverable: capture rows, gain-range visualisation, build
  progress, session integration). This module only provides the backend
  the UI would call into.
- Cross-amp generalization (Fender Super-Sonic, Fender 57 Twin) has not
  been run through THIS production builder yet -- only the earlier research
  harness (`scripts/continuous_gain_cross_amp_benchmark.py`) exercised
  multiple amp families, and that was Phase 2's discrete-interpolation
  baseline, not this module's virtual-gain-anchor path. This is the biggest
  open item before removing the Experimental label per the UI spec.
- CPU/memory/anchor-transition performance has not been measured; the
  runtime only renders through already-loaded models, so its cost is
  bounded by however many NAM renders the caller's UI triggers per knob
  move -- no explicit profiling has been done yet.
- Anomaly detection (latency-offset compensation, non-monotonic capture
  flags) is not yet re-run against the production anchor-selection path --
  `hybrid.continuous_gain.detect_capture_anomalies` exists and could be
  called on the same `TrainingTarget` outputs before `build_profile`, but
  this module does not call it automatically yet.
- Build-time quality is only measured at the TRAINING capture positions
  (the ones the builder has real audio for). The real-data regression
  benchmark above found a held-out HALF-STEP position (G8.5) can spike to
  raw ESR 2.11 even when every training capture nearby scored "acceptable"
  -- a real, verified reconstruction failure the training-only quality
  metric cannot see. `recommended_capture_position` is therefore currently
  blind to this class of failure; closing this gap (e.g. by shrinking an
  anchor's owned range faster near a training capture with high
  `worst_raw_esr_in_region`, or checking curvature between adjacent
  mapping points) is unimplemented follow-up work.

## What must be validated before removing the Experimental label

1. Cross-amp generalization: rerun the same regression benchmark on at
   least one additional amp family (Fender Super-Sonic, Fender 57 Twin) and
   confirm anchor selection still reaches "acceptable" or better within
   `max_anchors=3`.
2. Confirm the production anchor-selection/mapping-fit result matches the
   earlier ad hoc virtual-gain-benchmark research finding on the SAME
   dataset (same anchors chosen, comparable ESR) -- i.e. that formalizing
   the search into `build_profile` did not silently change the answer.
3. Decide and implement the discrete-interpolation fallback's actual
   trigger condition and runtime dispatch, so "poor" virtual-gain profiles
   have a real fallback rather than just a flagged strategy field.
4. UI integration per docs/CONTINUOUS_GAIN_UI.md, including surfacing
   `recommended_capture_position` as the "add a capture here" prompt the UI
   spec describes.
