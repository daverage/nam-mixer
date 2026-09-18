# Continuous Gain Model -- Raw-ESR Metric Breaks Down for Heavily-Saturated Amps

Follow-up to docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md and
docs/CONTINUOUS_GAIN_GENERALIZATION.md. Tests a master-volume, high-gain
preamp amp (Peavey 5150) to separate preamp-driven saturation from the
Fender/Marshall power-amp-adjacent knees studied so far -- and surfaces a
methodological limitation in Phase 2's validation approach itself, not just
another data point on the same scale.

## Source datasets

Peavey 5150 head, one channel, Gain 1-10, in two variants: stock and
"Boosted" (an SD1-style overdrive pushing the input, so the amp is already
saturated from Gain 1). A third dataset, Fender 57 Custom Twin (Channel 1,
Volume 1-10), was also run as a plain generalization check and behaves like
the previously-studied Fender Super-Sonic (front-loaded curve, elevated
error only near its own Volume 2-3 knee, otherwise consistent with the
existing report -- not detailed further here since it doesn't change any
conclusion).

## A genuine capture anomaly, caught by a new advisory check

Before running leave-one-out, the 5150 STOCK set's active-RMS progression
is non-monotonic: Gain 4 measures QUIETER than both Gain 3 and Gain 5
(-19.67 dBFS vs -18.16 and -17.30 dBFS) -- exactly the "unusual level
jumps"/"non-monotonic distortion changes" the original doc's "Capture
Validation" section anticipated. `hybrid.continuous_gain.
detect_capture_anomalies` (new, advisory-only per the doc's "do not reject
captures simply because an amplifier behaves unusually") now flags this
automatically from a capture set's own ground-truth harness output.

## Leave-one-out looks catastrophic -- uniformly, not near a knee

Raw ESR at every interior gain, standard DI, stock variant:

| Hidden | Neighbours | raw ESR | gain-norm ESR | spectral correlation |
|---:|---|---:|---:|---:|
| 2 | G1/G3 | 1.52 | 1.96 | 0.960 |
| 3 | G2/G4 | 0.79 | 1.05 | 0.968 |
| 4 | G3/G5 | 0.86 | 0.95 | 0.967 |
| 5 | G4/G6 | 1.38 | 1.93 | 0.974 |
| 6 | G5/G7 | 1.54 | 1.87 | 0.967 |
| 7 | G6/G8 | 1.33 | 1.65 | 0.967 |
| 8 | G7/G9 | 1.13 | 1.43 | 0.968 |
| 9 | G8/G10 | 1.30 | 1.73 | 0.971 |

Every JCM800/Fender dataset studied so far had raw ESR in the 0.003-0.10
range even at their worst knee. Here it is **10-200x higher everywhere**,
including gains far from the anomalous Gain 4 and far from either end of
the sweep. The boosted variant looks the same (raw ESR consistently
1.0-1.8) despite having no comparable capture anomaly and being already
saturated (so, if anything, a SIMPLER, flatter curve to interpolate).

## The metric, not the reconstruction, is what's failing here

Three lines of evidence, all pointing the same way:

1. **Sample-domain correlation between adjacent captures is near zero.**
   `corrcoef(G5_output, G7_output) = 0.30` for two captures one Gain step
   apart on the same amp/DI. The reconstruction-vs-real correlation for the
   withheld Gain 6 is even lower (0.06).
2. **Spectral-magnitude correlation for the exact same pair stays high**
   (`spectral_magnitude_correlation`, new in `hybrid.audio_metrics`): 0.967
   log-magnitude-spectrum correlation between the Gain-6 reconstruction and
   the real Gain-6 capture, despite 0.064 sample-domain correlation on the
   same two signals. This function's own unit test demonstrates the
   mechanism directly: a signal and a time-shifted copy of ITSELF show
   near-zero sample correlation but >0.5 spectral correlation, because
   shifting doesn't change a magnitude spectrum. Heavy nonlinear distortion
   is hypersensitive to microscopic input differences at individual
   clipping instants -- a tiny gain difference shifts WHERE a sample clips,
   sample-decorrelating the waveforms without the harmonic/spectral content
   actually differing much.
3. **Wide-vs-narrow spacing barely moves raw ESR here, unlike every other
   dataset.** On JCM800 and both Fender sets, widening the bracket around a
   target gain made raw ESR 4-20x worse (see
   CONTINUOUS_GAIN_PHASE2_RESULTS.md /
   CONTINUOUS_GAIN_GENERALIZATION.md). On the 5150 stock set, predicting
   Gain 6 from G5/G7 (narrow) gives raw ESR 1.54; from G1/G10 (as wide as
   possible) gives 1.45 -- **statistically indistinguishable, if anything
   slightly better wider**. If raw ESR were measuring real reconstruction
   quality, tighter brackets should help here too, the same way they did on
   every other dataset. That they don't is strong evidence the ESR values
   are dominated by clipping-instant decorrelation noise that swamps
   whatever real bracket-width signal exists underneath it.

## Conclusion

**No conclusion about capture density, spacing, or reconstruction quality
can be drawn for this amp from raw ESR.** The metric itself is unsuitable
for a heavily-saturated, preamp-driven high-gain amp in its current form --
this is a limitation of the Phase 2 validation methodology, not evidence
that the baseline (or density, or any other lever) works or fails on this
amp category. `spectral_magnitude_correlation` is a partial answer but is
uniformly high (0.96-0.97) across every gain/spacing combination tested
here, so it also doesn't currently discriminate a good reconstruction from
a bad one on this data -- it confirms the SHAPE survives, but has no
resolution left to say how well.

This matches the original doc's own caution: "do not treat a single ESR
value as proof of perceptual correctness... also generate audio comparisons
for listening tests." For heavily-driven amps specifically, that stops
being a supplementary nicety and becomes required -- ESR here is actively
misleading, not just noisy.

## What this changes going forward

- Do not extend the Phase 2/generalization conclusions (capture density,
  spacing sensitivity, DI robustness) to heavily-saturated/high-gain
  preamp amps until a metric that survives clipping-instant decorrelation
  is validated on this data. Candidates worth trying before any further
  5150-style dataset: a multi-resolution STFT/mel-spectrogram distance,
  an envelope-only comparison, harmonic-series/THD comparison, or an
  alignment step (e.g. per-frame phase correction) applied before any
  sample-domain metric.
- `detect_capture_anomalies` and `spectral_magnitude_correlation` are both
  now permanent, tested additions to `hybrid.continuous_gain`/
  `hybrid.audio_metrics` and are included in every future leave-one-out
  run's metrics automatically -- this failure mode will be visible (high
  spectral correlation alongside high raw ESR) rather than silently
  misread as "the baseline collapsed everywhere" the next time a heavily
  saturated amp is tested.
- The Peavey 5150's actual density/spacing/interpolation-quality behaviour
  remains an open question, not a negative result -- it simply cannot be
  answered with the current metric.
