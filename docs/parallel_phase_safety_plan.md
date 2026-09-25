# Parallel Blend phase safety plan

## Product intent

Parallel Blend exists to keep two deliberately different amps audible at the
same time (for example, clean articulation underneath a distorted amp).  The
tool must therefore assess and improve the real parallel sum.  It must not
redirect the user to a different design mode merely because the sources are
different.

The existing timing diagnostic answers only one narrower question: whether a
single, repeatable whole-signal delay can be proven and removed.  It is not a
verdict on cancellation, polarity, masking, or whether both amps remain
audible in the selected mix.

## User-facing contract

Parallel Blend will report three independent facts:

1. **Parallel compatibility** - whether the actual weighted sum shows strong,
   repeatable destructive interaction in musically important bands.
2. **Amp presence** - whether both weighted amp contributions remain material
   at the selected ratio, or one is likely to be obscured.
3. **Fixed latency** - the existing conservative fixed-offset result, retained
   as advanced evidence and never presented as the overall blend verdict.

The wording must distinguish `safe`, `some coloration`, `strong cancellation`,
and `not enough evidence`.  "Safe" means no strong repeatable cancellation was
found; it never claims that two different amps are perfectly phase-aligned.

## Measurement design

Analyse the exact components used by Parallel Blend after timing choice,
automatic/manual Amp B trim, polarity, and mix weighting.  Use overlapping
windowed spectra over active playing and compare the energy of the real sum
against the independent component-power reference `P(A) + P(B)`.

Only score a frequency band when both components have meaningful energy there.
This avoids calling an ordinary EQ difference a phase failure.  Aggregate over
multiple frames so a transient-only notch is not reported as a persistent
problem.  Report the worst material band and its interaction loss in dB.

Presence is a separate weighted-level measurement.  It warns when one selected
component is so far below the other that it is unlikely to remain clearly
audible, without conflating masking with phase cancellation.

## Corrections and audition

- Offer **Original / Flip Amp B polarity** inside Parallel Blend.  Recommend a
  flip only when the measured original sum has strong cancellation and the
  inverted version materially improves the worst band without creating an
  equally severe replacement problem.
- Continue to offer the existing verified fixed-delay correction independently.
- Every chosen correction must affect normal preview, live audition, generated
  training targets, held-out validation, saved sessions, manifests, and export.
- Compare alternatives at the same downstream output-level policy so a louder
  option is not mistaken for a better one.
- Never silently apply a correction.

The Parallel panel presents these controls as one guided workflow: first play
the current mix in the shared comparison player, then audition Original and
flipped Amp B polarity in that same player, then confirm the preferred result
across two additional performances.  The verdict leads with whether the result
is good, needs listening, or needs action; band measurements and fixed-latency
evidence remain available under Technical evidence.

The preview-DI result is always labelled **First check** and is never presented
as the final safety verdict.  The full check replaces it with counts across all
usable performances.  One failing performance is described as
performance-dependent risk; only two or more failures are described as
repeatable.  When no polarity is consistently safer, the UI offers explicit
10% moves toward Amp A or Amp B, plays the choice, and asks for a re-check.

## Delivery stages

### Stage 1 - auditioned-performance safety (implemented with this plan)

- Add deterministic band-energy cancellation and weighted-presence analysis.
- Return it from `/api/mix_info` for the current ratio/trim/timing/polarity.
- Replace the Parallel panel's timing-first presentation with a compatibility
  verdict and plain-language action.
- Add instant Original / Flip Amp B polarity audition.
- Freeze polarity into the design, generated target, validation reference,
  manifest, and session.
- Move rejected per-region timing guesses behind advanced evidence and display
  `agreement not applicable` when no timing regions were usable.

### Stage 2 - independent-DI confirmation (implemented with this plan)

- Re-run compatibility and polarity comparison on demand across two independent
  bundled DIs in addition to the preview DI.
- Promote wording from "on this performance" to "confirmed across performances"
  only when all usable DIs agree; otherwise keep the choice available for
  audition but remove the recommendation.

### Stage 3 - phase-optimised delay audition

- Search a small bounded integer-delay range against the cancellation score,
  independently of the latency detector.
- Call it **parallel phase optimisation**, never latency correction.
- Recommend/freeze a candidate only when it improves the worst-case result
  across independent DIs and does not create a new severe band.
- Expose manual sample/ms audition only as an advanced control, with the same
  persistence and validation rules.

## Acceptance criteria

- Identical in-polarity sources report safe and both present.
- An equal-level polarity-reversed pair reports strong cancellation and
  recommends flipping Amp B; applying the flip restores the expected sum.
- Spectrally different, weakly coherent sources are not rejected merely for
  having low waveform correlation.
- Mix endpoints report the absent source honestly and do not produce a false
  phase warning.
- Changing mix, trim, timing, or polarity refreshes the verdict without NAM
  inference.
- Preview, live audition, frozen design, generated target, validation, session
  restore, and manifest all reproduce the selected polarity exactly.
- Old sessions and designs without polarity fields remain original-polarity and
  bit-for-bit compatible.
