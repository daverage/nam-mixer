# Continuous Gain Model -- Response-Coordinate Investigation

Follow-up to docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md,
docs/CONTINUOUS_GAIN_GENERALIZATION.md, and
docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md. This is diagnostic/geometry work,
not Phase 3 feature work -- no Hybrid/Character machinery is touched.

New code: `hybrid/response_coordinate.py` (the response axis, non-oracle
interpolation, and oracle diagnostic), plus `hybrid.audio_metrics.
envelope_error_db` / `multi_resolution_log_spectral_distance` /
`framed_spectral_correlation` for Part 7. Reused throughout:
`scripts/continuous_gain_response_coordinate.py` runs Parts 1-4/6 against
one real dataset.

## 1. The control-law problem

`hybrid.continuous_gain` currently treats a capture's KNOB position (Gain
1..10, Volume 1..10) as the interpolation coordinate directly, via
`GainCaptureSet.normalized_position`. Two distinct nonlinearities can hide
behind an apparently steep "knee" measured this way:

```text
knob position -> control/pot law -> electrical gain/drive -> amplifier
nonlinear response -> output
```

A knob's electrical/taper law is not guaranteed to be linear (log/audio
tapers, loading interactions with the surrounding circuit), and even a
perfectly linear electrical gain can drive an amplifier through a highly
nonlinear clean -> breakup -> saturation transition. Every finding so far
has been stated in terms of the OUTPUT'S response to knob position, which
conflates these two stages. This report does not attempt to separate them
electrically (no circuit measurements exist) -- it instead asks a narrower,
answerable question: how much of the observed interpolation error is
explained by knob position simply being the wrong INTERPOLATION coordinate,
regardless of which upstream stage causes the mismatch.

## 2. Adjacent-capture measurements (Part 1)

For each dataset, per-step level change (|Δ active RMS|), peak change, and
spectral distance (`1 - spectral_magnitude_correlation`, NOT raw ESR --
see docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md for why sample-domain
comparison between independently rendered saturated captures is misleading)
between neighbouring TRAINING captures, standard DI, 6s.

### JCM800 2203 1985, Hi channel

| Step | level Δ (dB) | peak Δ (dB) | spectral distance |
|---|---:|---:|---:|
| G2->G4 | 4.316 | 1.578 | 0.0701 |
| G4->G6 | 0.667 | 0.241 | 0.0773 |
| G6->G8 | 0.249 | 0.549 | 0.1110 |
| G8->G10 | 0.043 | 0.398 | 0.1017 |

### JCM800 2203 1985, Lo channel

| Step | level Δ (dB) | peak Δ (dB) | spectral distance |
|---|---:|---:|---:|
| G2->G4 | 5.803 | 5.323 | 0.0217 |
| G4->G6 | 2.190 | 1.703 | 0.0208 |
| G6->G8 | 2.896 | 1.809 | 0.0211 |
| G8->G10 | 1.233 | 0.578 | 0.0257 |

### Marshall JCM800 2203 -- updated, dense sweep (Gain 1.0-4.0 shown, the
knee region)

| Step | level Δ (dB) | peak Δ (dB) | spectral distance |
|---|---:|---:|---:|
| G1.0->G1.5 | 1.449 | 0.179 | 0.0426 |
| G1.5->G2.0 | 0.662 | 0.314 | 0.0404 |
| G2.0->G2.5 | 0.612 | 0.481 | 0.0406 |
| G2.5->G3.0 | 0.347 | 0.040 | 0.0364 |
| G3.0->G3.5 | 0.119 | 0.357 | 0.0297 |
| G3.5->G4.0 | 0.088 | 0.093 | 0.0278 |

### Fender Super-Sonic, Vibrolux channel (V1-V10)

| Step | level Δ (dB) | peak Δ (dB) | spectral distance |
|---|---:|---:|---:|
| V1->V2 | 2.462 | 2.812 | 0.0309 |
| V2->V3 | 4.248 | 3.225 | 0.0425 |
| V3->V4 | 1.910 | 1.720 | 0.0587 |
| V4->V5 | 0.739 | 0.192 | 0.0615 |
| V5->V6 | 0.454 | 0.999 | 0.0647 |
| V6->V7 | 0.527 | 0.315 | 0.0741 |
| V7->V8 | 0.373 | 0.153 | 0.0791 |
| V8->V9 | 0.224 | 0.321 | 0.0684 |
| V9->V10 | 0.041 | 0.170 | 0.0616 |

### Fender 57 Custom Twin, Channel 1 (Vol 1-10)

| Step | level Δ (dB) | peak Δ (dB) | spectral distance |
|---|---:|---:|---:|
| V1->V2 | 3.301 | 5.041 | 0.0621 |
| V2->V3 | 1.280 | 0.515 | 0.0868 |
| V3->V4 | 0.241 | 0.388 | 0.0838 |
| V4->V5 | 0.091 | 0.296 | 0.0666 |
| V5->V6 | 0.092 | 0.197 | 0.0643 |
| V6->V7 | 0.088 | 0.228 | 0.0968 |
| V7->V8 | 0.010 | 0.079 | 0.0905 |
| V8->V9 | 0.029 | 0.389 | 0.0664 |
| V9->V10 | 0.024 | 0.195 | 0.0232 |

**Observation:** on both JCM800 datasets, spectral distance does NOT simply
track level change -- e.g. Hi channel's G6->G8 and G8->G10 steps have
LARGER spectral distance (0.111, 0.102) than the "obvious" G2->G4 knee
(0.070), even though their level change is far smaller (0.25, 0.04 dB vs
4.3 dB). Level-based knee-finding (used throughout the earlier reports)
would never have surfaced this -- the amp's TONAL character keeps moving
noticeably even where its LOUDNESS has flattened out.

## 3. Knob coordinate vs. capture-derived response coordinate (Part 2)

Two response axes per dataset: RMS-only (matches the level-based knee-
finding used previously) and level+spectral (equal weights, not tuned).

### JCM800 Hi

| Knob | RMS-only r | level+spectral r |
|---:|---:|---:|
| 2 | 0.000 | 0.000 |
| 4 | 0.837 | 0.318 |
| 6 | 0.960 | 0.421 |
| 8 | 1.000 | 0.754 |
| 10 | 1.000 | 1.000 |

RMS-only compresses almost the ENTIRE response into the first step (G2->G4)
and treats G6->G10 as essentially flat (r goes from 0.96 to 1.00). The
spectral-aware axis disagrees sharply: it places barely a third of the
total response change in that first step, and a large additional jump
between G6 and G8 -- exactly the step the level-only view called "done."

### Fender Super-Sonic

| Knob | RMS-only r | level+spectral r |
|---:|---:|---:|
| 1 | 0.000 | 0.000 |
| 2 | 0.228 | 0.072 |
| 3 | 0.625 | 0.227 |
| 4 | 0.801 | 0.355 |
| 5 | 0.867 | 0.456 |
| 6 | 0.906 | 0.556 |
| 7 | 0.951 | 0.682 |
| 8 | 0.983 | 0.817 |
| 9 | 1.000 | 0.920 |
| 10 | 1.000 | 1.000 |

Here the two axes agree more closely in SHAPE (both front-loaded), but
RMS-only still says the sweep is "basically done" by V3 (r=0.625), while
level+spectral says under a quarter of the response has happened by then
(r=0.227) -- the amp keeps changing tonally well past where its loudness
curve flattens.

(Fender 57 Twin and JCM800 Lo show the same qualitative pattern -- level
saturates faster than the spectral-aware coordinate in every dataset
tested; full tables in the script output, omitted here for length.)

## 4. Oracle diagnostic: does knob-linear placement miss the true response
position? (Part 3, Test A -- DIAGNOSTIC ONLY)

For each interior hidden capture, `oracle_response_mismatch` uses the
capture's OWN real output (never available at inference time -- this is
never a production method) to measure where it truly sits, in response
space, between its two real neighbours, and compares that to the
knob-linear fraction a production system would have assumed.

| Dataset | Hidden | knob-linear fraction | true response fraction | mismatch | knob-linear raw ESR |
|---|---:|---:|---:|---:|---:|
| JCM800 Hi | 4 | 0.50 | 0.74 | 0.24 | 0.0986 |
| JCM800 Hi | 6 | 0.50 | 0.24 | 0.26 | 0.0558 |
| JCM800 Hi | 8 | 0.50 | 0.58 | 0.08 | 0.0491 |
| JCM800 Lo | 4 | 0.50 | 0.14 | 0.36 | 0.0169 |
| JCM800 Lo | 6 | 0.50 | 0.67 | 0.17 | 0.0235 |
| JCM800 Lo | 8 | 0.50 | 0.45 | 0.05 | 0.0078 |
| Super-Sonic | 2 | 0.50 | 0.29 | 0.21 | 0.0647 |
| Super-Sonic | 3 | 0.50 | 0.49 | 0.01 | 0.0503 |
| Super-Sonic | 4 | 0.50 | 0.44 | 0.06 | 0.0236 |
| Super-Sonic | 5-9 | 0.50 | 0.44-0.57 | 0.001-0.07 | 0.005-0.008 |
| 57 Twin | 2-9 | 0.50 | 0.39-0.66 | 0.01-0.14 | 0.002-0.048 |

Per-dataset correlation between mismatch and knob-linear raw ESR:

| Dataset | n | correlation |
|---|---:|---:|
| JCM800 Hi | 3 | +0.51 |
| JCM800 Lo | 3 | +0.48 |
| Dense Marshall (knee subset) | 5 | -0.65 (includes an edge-of-range artifact, see caveat below) |
| Fender Super-Sonic | 8 | +0.60 |
| Fender 57 Twin | 8 | -0.32 (also affected by an edge artifact at its last interior point) |

**Caveat:** `oracle_response_mismatch`'s normalization range comes from
that dataset's OWN adjacent-step measurements; when the hidden point is
near the edge of a small/truncated sweep, the reference range is thin and
the mismatch value is less reliable (both the dense-Marshall and 57-Twin
outlier points are their last interior position). This is a real limitation
of the diagnostic, not evidence the underlying question is unanswerable --
it means edge points should be weighted less, not that the correlation
sign flip is necessarily meaningful.

**Conclusion for Part 3 Test A:** the correlation is positive and
moderate on 3 of 5 datasets (both JCM800 channels, Super-Sonic), consistent
with the hypothesis that knob-linear placement mismatch contributes to
error, but the sample sizes (n=3-8 per dataset) are too small for
confidence, and 2 of 5 datasets show a negative correlation (partly
explained by an edge-of-range artifact). **This is supported but not
proven.**

## 5. Non-oracle response-coordinate interpolation (Part 3, Test B)

Same leave-one-out captures, but now `interpolate_output_response_coordinate`
estimates the hidden position's response coordinate via `PchipInterpolator`
fit ONLY on the remaining training captures' (knob position, response
coordinate) pairs -- no oracle access to the hidden capture's real output.

| Dataset | Hidden | knob-linear raw ESR | response-coordinate raw ESR | Better? |
|---|---:|---:|---:|---|
| JCM800 Hi | 4 | 0.0986 | 0.1480 | worse |
| JCM800 Hi | 6 | 0.0558 | 0.0578 | worse |
| JCM800 Hi | 8 | 0.0491 | 0.0630 | worse |
| JCM800 Lo | 4 | 0.0169 | 0.0042 | **4x better** |
| JCM800 Lo | 6 | 0.0235 | 0.0235 | same |
| JCM800 Lo | 8 | 0.0078 | 0.0394 | 5x worse |
| Super-Sonic | 2 | 0.0647 | 0.0432 | better |
| Super-Sonic | 3 | 0.0503 | 0.0498 | ~same |
| Super-Sonic | 4 | 0.0236 | 0.0206 | better |
| Super-Sonic | 5-9 | 0.005-0.008 | 0.005-0.008 | same or slightly better |
| 57 Twin | 2-9 (8 points) | -- | -- | better or equal at ALL 8 points, margins small |

**Conclusion for Part 3 Test B:**

- **JCM800 Hi**: response-coordinate interpolation is WORSE at all 3 tested
  points. Report this clearly, as instructed -- it does not help here.
- **JCM800 Lo**: mixed -- a large win at one point, a clear loss at another.
  Not a reliable improvement.
- **Fender Super-Sonic and Fender 57 Twin**: consistently better or equal at
  every interior point tested (8/8 on each), though margins on the 57 Twin
  are small. This is the one place Test B shows a clean, repeatable win.

There is no dataset where response-coordinate interpolation is
UNAMBIGUOUSLY better across the board, and one dataset (JCM800 Hi) where it
is unambiguously worse. **Response-coordinate interpolation, in this
simplest form, is not a general improvement over knob-linear
interpolation** -- see Part 6 for why it may still be useful for a
different purpose.

## 6. Capture placement vs. interpolation weighting (Part 4)

These are separate questions and the evidence differs between them:

- **Interpolation weighting** (Part 5 above): mixed-to-negative. Warping the
  blend fraction using the response axis does not reliably reduce
  reconstruction error.
- **Capture placement** (score comparison, Part 4/6): on JCM800 Hi and Lo,
  the level+spectral score identified a DIFFERENT top-priority step (Hi:
  G6->G8, spectral distance-driven) than RMS-only (Hi: G2->G4, level-driven)
  -- see Section 2's spectral-distance-doesn't-track-level observation.
  Whether this second signal is a genuinely better placement recommendation
  cannot be confirmed without adding a real capture at G6-G8 and measuring
  the effect (out of scope here -- flagged as the recommended next
  experiment). On the two Fender datasets, RMS-only and combined scores
  agreed on the top step (V1->V2 or V2->V3), so the extra spectral signal
  added no new information there.

**This matches the doc's anticipated "valid and potentially preferable"
outcome**: the response axis's clearest, most consistent value so far is as
an ADDITIONAL capture-placement signal (surfacing tonal changes that level
alone misses), not as a direct replacement for knob-linear interpolation
weights.

## 7. Reassessing the Super-Sonic "knee" (Part 5)

The earlier generalization report described the Super-Sonic's Volume 2->3
jump (+8.5 dB active RMS) as a sharp knee and speculated it reflects a
clean-headroom-clipping onset. Revisiting with the response coordinate:

- The RMS-only response axis places 62% of the total normalized response by
  V3 (r=0.625); the spectral-aware axis places only 23% there (r=0.227).
  **Level and tonal character are NOT moving together at the same rate**
  through this region -- level races ahead of spectral character, or
  spectral character continues changing well past where level saturates
  (Section 3's observation, most pronounced here).
- What the available NAM captures CAN tell us: the OUTPUT response (level
  and spectral character together) changes fastest between V2 and V3, and
  continues changing non-trivially through at least V7 (spectral distance
  stays 0.06-0.08 all the way to V7->V8, comparable to the V2->V3 step's
  0.0425).
- What they CANNOT tell us: whether this is caused by (a) the Volume pot's
  own taper being nonlinear, (b) the pot interacting with surrounding
  circuit loading, (c) the amplifier's own clean-to-breakup clipping
  transition, or (d) some combination. A `.nam` capture is an end-to-end
  audio measurement; it has no visibility into the potentiometer's own
  electrical law or where in the circuit the nonlinearity originates.
  Separating these would require either direct electrical measurement of
  the pot/circuit, or a far denser set of intermediate captures than exists
  for this dataset (the JCM800 "updated" 0.5-step sweep is the only dataset
  dense enough to even approach this, and it is a different amp).

**Revised claim: do not describe the Super-Sonic's V2->V3 jump as
"resolved" evidence of a specific circuit mechanism.** It is real,
large, and reproducible in the measured OUTPUT; its physical origin is
undetermined with the data available.

## 8. RMS-only vs. spectral-aware capture placement (Part 6)

| Dataset | RMS-only top step | level+spectral top step | Agree? |
|---|---|---|---|
| JCM800 Hi | G2->G4 | G6->G8 | **No** |
| JCM800 Lo | G2->G4 | G2->G4 | Yes |
| Dense Marshall (knee subset) | G1.0->G1.5 | G1.0->G1.5 | Yes |
| Super-Sonic | V2->V3 | V2->V3 | Yes |
| 57 Twin | V1->V2 | V1->V2 | Yes |

4 of 5 datasets agree; only JCM800 Hi disagrees, and it disagrees because
of the Section 2 finding (spectral character keeps moving in the "flat"
G6->G10 region while level does not). **RMS-only is not obviously wrong
as a default, but it can miss a real, measurable tonal-only knee that the
spectral-aware score catches** -- exactly the JCM800 Hi case. Given the
low cost of computing `spectral_magnitude_correlation` per step (already
implemented, already fast), there is no strong reason to drop the spectral
component, but there is also not yet direct evidence (a validated capture
added at G6-G8) that acting on it improves anything -- this is a
recommendation to KEEP tracking it and validate with a real capture next,
not to replace RMS-only today.

## 9. 5150 metric-discrimination experiment (Part 7)

Per docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md, raw ESR cannot be trusted on
the 5150. Before attempting ANY interpolation-quality conclusion there, this
tests whether ANY of a small set of candidate metrics can tell an
obviously-closer reconstruction (narrow bracket: G5/G7 -> G6, the natural
nearest neighbours) from an obviously-worse one (wide bracket: G1/G10 ->
G6) on the SAME stock 5150 data, across all three DI types.

| DI | Bracket | raw ESR | whole-signal spectral corr | framed spectral corr | multi-res log-spectral distance | envelope error (dB) |
|---|---|---:|---:|---:|---:|---:|
| standard | narrow | 1.539 | 0.967 | 0.969 | 6.77 | **2.01** |
| standard | wide | 1.450 | 0.969 | 0.971 | 6.55 | **4.48** |
| clean | narrow | 1.679 | 0.966 | 0.964 | 6.63 | **1.90** |
| clean | wide | 1.432 | 0.966 | 0.968 | 6.72 | **4.58** |
| metalcore | narrow | 1.766 | 0.967 | 0.966 | 6.85 | **1.80** |
| metalcore | wide | 1.554 | 0.965 | 0.967 | 6.57 | **4.33** |

**Result:** raw ESR, whole-signal spectral correlation, framed spectral
correlation, and multi-resolution log-spectral distance ALL FAIL the basic
discrimination test -- narrow is not consistently ranked better than wide
by any of them; several even rank the (objectively worse) wide bracket as
slightly better. **`envelope_error_db` is the one candidate that discriminates
cleanly and consistently**: narrow is roughly HALF the error of wide across
all three DI types (2.0 vs 4.5, 1.9 vs 4.6, 1.8 vs 4.3 dB) -- the direction
and magnitude a real quality difference should produce.

A harmonic/THD-style comparison was not implemented: THD is only
well-defined for a sinusoidal input, and the DI material here is broadband
music, so no defensible THD-style metric was available without a much
larger effort (a proxy would need its own validation, which is out of
scope for this pass).

**This unblocks 5150 interpolation-quality investigation, cautiously**:
`envelope_error_db` should be the PRIMARY metric for any future 5150 leave-
one-out work, with raw ESR and spectral correlation reported alongside for
context but not trusted alone. A single passing discrimination test on one
capture/DI combination is not full validation -- the next 5150 experiment
should confirm this holds across multiple hidden gains before drawing any
density/spacing conclusion for this amp.

## 10. Conclusions

- **Confirmed**: spectral character and active-RMS level do not always
  change together -- on both JCM800 datasets and the Super-Sonic, level
  saturates measurably faster than spectral distance. RMS-alone
  knee-finding can miss a real, measurable tonal transition (JCM800 Hi's
  G6->G8 step).
- **Confirmed**: none of raw ESR, whole-signal spectral correlation, framed
  spectral correlation, or multi-resolution log-spectral distance can
  discriminate an obviously-better from an obviously-worse reconstruction
  on the 5150's heavily saturated material; `envelope_error_db` can, at
  least in this one test.
- **Supported but not proven**: knob-linear interpolation's mismatch from
  the true (oracle) response position correlates positively with
  reconstruction error on 3 of 5 datasets (JCM800 Hi, JCM800 Lo,
  Super-Sonic), but sample sizes are small (n=3-8) and 2 datasets show a
  negative correlation, partly attributable to an edge-of-range artifact in
  the diagnostic.
- **Supported but not proven**: response-coordinate interpolation
  (non-oracle) helps consistently on both Fender datasets, but is neutral-
  to-harmful on both JCM800 datasets. There is no evidence yet that this
  generalizes as a universal replacement for knob-linear interpolation.
- **Unresolved**: whether the response axis is a reliably better
  CAPTURE-PLACEMENT signal than RMS alone -- it flagged a different (and
  plausible) priority region on JCM800 Hi, but this has not been validated
  by actually adding a capture there and re-measuring.
- **Unresolved, and explicitly out of scope for this report**: the physical
  origin (pot taper vs. circuit interaction vs. amplifier clipping) of any
  of the measured knees. The available NAM captures cannot distinguish
  these causes.
- **Unresolved**: whether preamp-driven vs. power/output-stage-driven amps
  have systematically different knee behaviour -- see decision gate Q7.

## 11. Product implications for NAM Mixer

- Do not build a response-coordinate INTERPOLATION mode yet -- the evidence
  is mixed, and on one dataset (JCM800 Hi) it made things worse. This would
  add complexity without a demonstrated general benefit.
- DO consider surfacing the spectral-aware capture-placement score
  alongside the existing RMS-delta-based one in any future capture-set
  analysis UI -- it is cheap to compute (already implemented) and caught a
  real region (JCM800 Hi's G6->G8) that level-only analysis missed. Treat
  its recommendations as a second opinion to validate, not a rule to apply
  automatically, until a real capture confirms it help
- For the 5150 and other heavily-saturated amps, plan to report
  `envelope_error_db` as the primary validation metric rather than raw ESR
  or spectral correlation, once a fuller discrimination test confirms it
  holds beyond this single example.
- Do not present any Super-Sonic-style knee to users as evidence of "this
  is a clipping-stage transition" or similar circuit-specific language --
  the data only supports "the amp's overall response changes fastest here,"
  not why.

## 12. Recommended next experiment

The smallest experiment that would resolve the largest remaining
uncertainty: **validate the capture-placement disagreement on JCM800 Hi**.
Obtain (or identify an existing) real capture between Gain 6 and Gain 8 on
that channel, add it to the training set, and re-run leave-one-out at
Gain 7 (and re-check Gain 6/Gain 8's own reconstruction quality). If this
confirms a real accuracy gain from a capture placed where the spectral-aware
score (not the RMS-only score) says to place it, that is strong, direct
evidence the spectral component is worth keeping in a capture-placement
recommendation. If it doesn't help, RMS-only remains the simpler, equally
good default and the spectral component can be deprioritized.

A secondary, cheaper follow-up: extend the Part 7 discrimination test to at
least 2 more hidden gains on the 5150 (e.g. Gain 4 and Gain 8) before
trusting `envelope_error_db` for any real 5150 density/spacing conclusion.

---

## Decision gate

1. **Are equal knob increments approximately equal response increments on
   any of the tested amps?** No. Every dataset tested (JCM800 Hi/Lo,
   Super-Sonic, 57 Twin, dense Marshall) shows a front-loaded or otherwise
   uneven response curve under EITHER response axis. Fixed knob spacing is
   not a safe assumption anywhere in this data.
2. **Does departure from knob linearity correlate with reconstruction
   error?** Supported but not proven -- positive correlation on 3 of 5
   datasets (JCM800 Hi/Lo, Super-Sonic), negative on 2 (partly an
   edge-of-range artifact), with small sample sizes throughout.
3. **Does response-coordinate interpolation outperform knob-linear
   interpolation without oracle information?** No, not in general.
   Consistent improvement on both Fender datasets; neutral-to-worse on both
   JCM800 datasets, including one dataset where it is clearly worse at
   every tested point.
4. **Is the response coordinate more useful for interpolation, capture
   placement, both, or neither?** Capture placement, tentatively -- it
   surfaced a real, plausible signal (JCM800 Hi's tonal-only knee at
   G6->G8) that RMS-alone missed, while its interpolation-weighting benefit
   is mixed. Not yet validated by an actual added capture.
5. **Does spectral information improve capture-placement prediction enough
   to justify complexity beyond RMS?** Partially -- 4 of 5 datasets showed
   no disagreement (spectral information added nothing new there), but the
   1 disagreement (JCM800 Hi) was a real, measurable, non-trivial signal.
   Given the negligible extra cost, keep computing it, but do not yet
   replace the simpler RMS-only heuristic as the primary signal.
6. **Can any new metric reliably distinguish narrow vs wide interpolation
   on the saturated 5150?** Yes -- `envelope_error_db`, in this one test.
   Raw ESR, whole-signal and framed spectral correlation, and multi-
   resolution log-spectral distance all failed to discriminate reliably.
7. **Is there now enough evidence to compare preamp-driven vs
   power/output-stage-driven knee behaviour, or is that topology question
   still blocked?** Still blocked. The 5150 (preamp-driven) remains
   unvalidated for any interpolation-quality question until
   `envelope_error_db` (or another metric) is confirmed across more than
   one hidden-gain test; no fair comparison to the JCM800/Fender
   (power-amp-adjacent) results can be made yet.
8. **What is the smallest next experiment that would resolve the largest
   remaining uncertainty?** Add a real capture between JCM800 Hi's Gain 6
   and Gain 8 and re-run leave-one-out there -- this directly tests whether
   the spectral-aware capture-placement signal (the one clear, actionable,
   novel finding in this report) predicts a real improvement, which is the
   single result most likely to change what gets built next.
