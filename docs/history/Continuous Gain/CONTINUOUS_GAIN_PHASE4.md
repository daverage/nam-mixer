# Continuous Gain Phase 4: quality-aware feasibility and capture selection

Guide for the work after `CONTINUOUS_GAIN_v3_COMPARISON.md`. Code: `scripts/p4_*.py`; data: `work/p4/<amp>/` (gitignored);
reports: `docs/CONTINUOUS_GAIN_PHASE4_*.md`. The v3 models, settings and results are the baseline and are never overwritten.

## Objective

Take fixed-gain NAM captures of ONE amp and produce ONE standard `.nam` giving a convincing approximation of the amp's
gain range through ordinary input gain. Not a numerically exact knob mapping: a useful continuous progression keeping
gain/saturation, compression/dynamics, spectrum/EQ/tone, output-level progression (including real plateaus), and response
to different input levels. Must beat switching between fixed captures, with practical generation time. Do not assume 3
captures suffice, that 10 is better, or that input gain can reproduce every aspect of a physical gain control.

## Phases

- **4A Capture audit.** Reproducible per-capture QA: timing/latency, clipping, dropouts, noise/hum, level and metadata
  consistency, tonal/spectral/dynamic anomalies vs neighbours, repeatability where possible. Status VALID / CORRECTED /
  SUSPECT / INVALID with evidence, original and corrected measurements. Never invalidate a capture merely for breaking an
  assumed smooth or monotonic curve. Never replace a capture with an interpolated waveform presented as real. Originals
  preserved; corrected copies separate; unresolved anomalies quarantined. Specific cases: JCM800 G8.5 timing, Peavey 5150 G4
  output level, Peavey timing offsets, Super-Sonic Bassman and Orange noise.
- **4B Response profile.** Per amp, consistent source material and several calibrated input levels: output RMS/peak, EQ
  bands, HF/tilt, THD and harmonic distribution, crest/dynamics, input-output curve and compression, noise. Level, tone,
  saturation, compression kept as separate dimensions. Measured vs estimated visibly distinct; plateaus and non-monotonic
  behaviour preserved; uncertainty shown for missing/suspect positions. Plots.
- **4C Mapping feasibility, NO retraining.** Render existing v3 models across an input-gain sweep and compare with reliable
  real references. Is there one ordered input-gain progression that approximates tone, saturation, compression and level?
  Not an RMS-only match. One mapping for all DIs/levels (never per recording or per metric). Compare fixed 4 dB spacing with a
  measured mapping. Separate what mapping can fix, what needs different training captures, what the approach cannot do.
  Report trade-offs (better saturation but worse compression is not a general improvement).
- **4D Capture usefulness.** From validated profiles, compare candidate 3- and 5-capture subsets against the 10-capture
  baseline; choose by measured contribution to the tonal/nonlinear progression, not knob spacing; consider whether endpoints
  are essential. Not solely NAM evaluation errors. No large automated neural search.
- **4E Pilot.** JCM800 + Super-Sonic Vibrolux. Freeze reference data + QA decisions, profile method, training positions,
  mapping, protocol and acceptance criteria BEFORE training. Mapping-only baseline = existing v3 C3 with original and revised
  mapping. Then ONE new adaptively selected C3 per pilot amp, same architecture and comparable settings; evaluate with both
  mappings separately; record real training duration. Do not expand to C5/C10 automatically.

## Safeguards

Keep explicit: reference vs model output; measured vs fitted; training vs analysis-only captures; training error vs excluded
positions; mapping vs model improvement; metric vs audible improvement. Fitting profiles/selection/mapping on a capture or DI
makes it non-independent, so independent DIs (held-out) and untouched reference positions are reserved for final
validation. Single-seed v3 results: no claims from small differences without repeatability checks. No claim from mean ESR or
one THD number alone. Exported `.nam` must work in an ordinary player; anything needing extra DSP/metadata is reported separately.

## Deliverables

Capture-QA report (8 amps); response plots; mapping-feasibility report (fixed vs measured, no retraining); proposed 3/5
subsets with reasons; controlled JCM800/Vibrolux pilot; training-time/fidelity/regression summary; workflow recommendation.
Final report must answer: did correcting captures change our understanding of the gain progression; can better mapping make an
existing NAM behave more like the amp; do 3 selected captures suffice for the pilot amps; which audible mismatches remain
and are they mapping, capture selection or model limits; what is the smallest practical workflow. Do not build the production
pipeline or claim all eight amps work until the pilot shows what genuinely helps.
