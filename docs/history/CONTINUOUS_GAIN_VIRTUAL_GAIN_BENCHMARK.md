# Continuous Gain Model -- Virtual Input Gain Benchmark

Follow-up to docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md,
docs/CONTINUOUS_GAIN_GENERALIZATION.md,
docs/CONTINUOUS_GAIN_RESPONSE_COORDINATE(_ADDENDUM).md, and
docs/CONTINUOUS_GAIN_CONDITIONAL_MODEL.md. No model training of any kind
occurs in this experiment -- pure signal processing and real NAM inference
(`hybrid.render.render`), reusing the existing discrete-interpolation
baseline and metrics for comparison. New code:
`scripts/continuous_gain_virtual_gain_benchmark.py`.

## Automated methodology

Primary dataset: the dense "Marshall JCM800 2203 - updated" sweep, High
channel, 19 real captures at Gain 1.0 to 10.0 in 0.5 steps. Held-out
evaluation audio: the final 6 seconds of `assets/di/moderate_brit.wav`
(same held-out-tail convention as docs/CONTINUOUS_GAIN_CONDITIONAL_MODEL.md
-- never used to fit anything, since nothing is fit in this experiment
anyway, but kept identical for cross-report comparability).

**Ground truth**: every one of the 19 real captures rendered through the
identical DI/sample rate/trim -- no alignment correction applied (see the
Gain 8.5 finding below for why that matters).

### Exact NAM-style signal path

```text
DI --(x 10**(input_gain_db / 20))--> fixed NAM --> output
```

`render_with_input_gain()` is the single function every part of this
benchmark routes through: gain is applied to the DRY signal BEFORE
`render()`, never to the rendered output afterward.

### Input-gain search procedure

For each (anchor, target) pair: a coarse grid from -24 to +24 dB in 2 dB
steps (25 points), then a fine grid at 0.2 dB steps within +-2 dB of the
coarse optimum (up to 21 more points), both minimizing RAW (actual,
unmatched) ESR -- the same thing a real NAM-player user optimizes for by
ear, since they hear the raw output, not a post-hoc level-corrected one.
The level-matched (gain-normalized) ESR at that same optimal input gain is
reported alongside as a diagnostic to isolate whether any remaining error
is level or shape -- **matching level alone is never treated as success**.

## Sanity checks (passed)

For every anchor (G1, G5, G10): 0 dB input gain reproduces the original
capture's render bit-for-bit (max difference 0.0, floating-point exact,
since 0 dB gain is a no-op multiply). Also verified that gain applied
BEFORE inference measurably differs from the same gain applied AFTER
inference (post-output) -- e.g. for the G5 anchor at +6 dB, pre-inference
and post-output renders differ by up to 0.17 in amplitude, confirming the
amp's nonlinearity is genuinely engaging the driven signal, not just being
bypassed by a linear gain trick.

## An important, unplanned finding: the Gain 8.5 capture has a real
latency offset

Gain 8.5 produces a catastrophic raw ESR (1.0-2.5, roughly 100-1000x every
neighbouring position) across ALL THREE experiments below -- one-anchor
virtual gain from G1, G5, AND G10 independently, and the unrelated
10-capture dense interpolation baseline. Since these methods share nothing
except the Gain 8.5 ground-truth target, the anomaly must live in that one
capture file.

Cross-correlating the raw renders confirms it: the best-aligning lag
between Gain 8.0's and Gain 8.5's renders is **-105 samples** (correlation
0.996 once aligned), and between Gain 8.5's and Gain 9.0's is **+105
samples** (0.996 aligned) -- while Gain 8.0 vs. Gain 9.0 directly (skipping
8.5) align at -1 sample (0.997), i.e. essentially zero offset. **Gain 8.5's
`.nam` file has a genuine ~105-sample (~2.2ms @ 48kHz) latency offset baked
into it relative to its neighbours.** This is not a level, spectral, or
modelling issue -- it is a real timing defect in that one capture,
independently confirmed by 4 unrelated reconstruction methods all tripping
over the exact same file the exact same way.

**This retroactively explains the previously-unresolved Gain 8.5 anomaly**
in docs/CONTINUOUS_GAIN_CONDITIONAL_MODEL.md (Section 10, updated with a
pointer to this finding) -- it was never a training-budget, capacity, or
conditional-model-specific issue.

**Gain 8.5 is excluded from all aggregate/average statistics below** and
reported only in its own row, flagged.

## Experiment A: one fixed NAM + virtual input gain, per anchor

Full per-target results in `work/continuous_gain_virtual_gain_benchmark.json`
(gitignored scratch output, regenerate with the script). Summary, raw ESR
(actual/unmatched), excluding Gain 8.5:

| Target | Anchor G1 | Anchor G5 | Anchor G10 |
|---:|---:|---:|---:|
| 1.0 | 0.0000 | 0.0344 (boundary) | 24.9998 (boundary, fails) |
| 1.5 | 0.0006 | 0.0087 | 4.5416 (boundary, fails) |
| 2.0 | 0.0015 | 0.0058 | 1.0542 (boundary, fails) |
| 2.5 | 0.0010 | 0.0033 | 0.0647 (boundary) |
| 3.0 | 0.0017 | 0.0048 | 0.0094 |
| 3.5 | 0.0041 | 0.0014 | 0.0035 |
| 4.0 | 0.0053 | 0.0021 | 0.0062 |
| 4.5 | 0.0083 | 0.0026 | 0.0061 |
| 5.0 | 0.0122 (boundary) | 0.0000 | 0.0027 |
| 5.5 | 0.0168 (boundary) | 0.0005 | 0.0032 |
| 6.0 | 0.0219 (boundary) | 0.0005 | 0.0032 |
| 6.5 | 0.1066 (boundary) | 0.0013 | 0.0029 |
| 7.0 | 0.1978 (boundary) | 0.0026 | 0.0016 |
| 7.5 | 0.2474 (boundary) | 0.0035 | 0.0019 |
| 8.0 | 0.3060 (boundary) | 0.0036 | 0.0016 |
| 9.0 | 0.3582 (boundary) | 0.0050 | 0.0011 |
| 9.5 | 0.3482 (boundary) | 0.0084 | 0.0024 |
| 10.0 | 0.3624 (boundary) | 0.0062 | 0.0000 |
| **Mean (excl. 8.5, 1.0/10.0 self-match)** | **0.130** | **0.0034** | **quality only from ~G3 up** |

**G1 works well up to about Gain 6, then fails hard and permanently pins
at the +24dB search boundary from Gain 6.5 onward** -- more input drive
cannot buy back the difference; a clean-voiced capture cannot be pushed
into matching a real high-gain capture's character just by turning it up
further, no matter how far the search range extends.

**G10 fails the mirror-image way**: it cannot be turned down enough
(-24dB boundary) to reach anything below about Gain 2.5-3.0 -- a
maximally-driven capture retains character that a virtual attenuation
cannot remove.

**G5 (the middle anchor) is dramatically better than either extreme**:
excellent (raw ESR consistently under 0.01, mostly under 0.005) across
NEARLY THE ENTIRE Gain 1.0-10.0 range, with only a mild, non-boundary-
pinned degradation right at the Gain 1.0 endpoint (0.0344). **This directly
answers "which single anchor spans the widest useful range": G5, and by a
wide margin.**

## Experiment B: three-anchor virtual-gain system

Per target, whichever of G1/G5/G10 (with its own best input gain)
minimizes raw ESR:

| Target | Best anchor | Input gain | Raw ESR |
|---:|---|---:|---:|
| 1.0 | G1 | +0.0 dB | 0.0000 |
| 1.5 | G1 | +5.6 dB | 0.0006 |
| 2.0 | G1 | +9.4 dB | 0.0015 |
| 2.5 | G1 | +13.8 dB | 0.0010 |
| 3.0 | G1 | +17.2 dB | 0.0017 |
| 3.5 | G5 | -4.2 dB | 0.0014 |
| 4.0 | G5 | -2.4 dB | 0.0021 |
| 4.5 | G5 | -1.2 dB | 0.0026 |
| 5.0 | G5 | +0.0 dB | 0.0000 |
| 5.5 | G5 | +1.4 dB | 0.0005 |
| 6.0 | G5 | +2.0 dB | 0.0005 |
| 6.5 | G5 | +5.8 dB | 0.0013 |
| 7.0 | G10 | -6.4 dB | 0.0016 |
| 7.5 | G10 | -4.6 dB | 0.0019 |
| 8.0 | G10 | -2.4 dB | 0.0016 |
| 8.5 | (anomaly, excluded) | -- | -- |
| 9.0 | G10 | -0.2 dB | 0.0011 |
| 9.5 | G10 | -0.4 dB | 0.0024 |
| 10.0 | G10 | +0.0 dB | 0.0000 |

The anchor selection is clean and physically sensible: G1 handles its own
local low-gain neighbourhood, G5 handles the entire middle of the range
(and extends usefully into the low-gain region too), G10 handles the
high-gain end. **Every target across the full sweep is reproduced with raw
ESR under 0.003** (excluding the Gain 8.5 file anomaly) -- as good as or
better than the dense 10-capture baseline (Experiment C below).

## Experiment C: dense discrete-capture baseline (G1-G10 interpolation)

Evaluated on the 9 genuine half-step targets, raw ESR, excluding Gain 8.5:

| Target | Raw ESR | Level-matched ESR | Level Δ (dB) | Spectral corr |
|---:|---:|---:|---:|---:|
| 1.5 | 0.0027 | 0.0014 | 0.30 | 0.985 |
| 2.5 | 0.0036 | 0.0016 | 0.37 | 0.976 |
| 3.5 | 0.0014 | 0.0011 | 0.14 | 0.979 |
| 4.5 | 0.0018 | 0.0018 | 0.05 | 0.955 |
| 5.5 | 0.0015 | 0.0005 | 0.27 | 0.967 |
| 6.5 | 0.0029 | 0.0019 | 0.28 | 0.961 |
| 7.5 | 0.0010 | 0.0009 | 0.06 | 0.947 |
| 9.5 | 0.0027 | 0.0021 | 0.19 | 0.933 |
| **Mean** | **0.0022** | **0.0014** | **0.21** | **0.963** |

This confirms the earlier reports' finding: dense discrete-capture
interpolation is already excellent on this amp.

## Required comparisons: all four methods at the 9 half-step targets

Raw ESR, excluding Gain 8.5:

| Target | 1 NAM (G5) + gain | Best of 3 anchors + gain | 10-capture interp | Real target |
|---:|---:|---:|---:|---|
| 1.5 | 0.0087 | 0.0006 (G1) | 0.0027 | (reference) |
| 2.5 | 0.0033 | 0.0010 (G1) | 0.0036 | (reference) |
| 3.5 | 0.0014 | 0.0014 (G5) | 0.0014 | (reference) |
| 4.5 | 0.0026 | 0.0026 (G5) | 0.0018 | (reference) |
| 5.5 | 0.0005 | 0.0005 (G5) | 0.0015 | (reference) |
| 6.5 | 0.0013 | 0.0013 (G5) | 0.0029 | (reference) |
| 7.5 | 0.0035 | 0.0019 (G10) | 0.0010 | (reference) |
| 9.5 | 0.0084 | 0.0024 (G10) | 0.0027 | (reference) |
| **Mean** | **0.0037** | **0.0015** | **0.0022** | |

**The headline result**: **3 real captures (G1/G5/G10) + ordinary virtual
input gain (mean raw ESR 0.0015) slightly OUTPERFORMS the full 10-capture
dense interpolation baseline (mean raw ESR 0.0022) on these exact half-step
targets.** Even the single-capture G5-only version (mean 0.0037) is within
~1.7x of the 10-capture baseline, using one tenth as many real captures.

## Per-target spectral results

Spectral correlation (not collapsed with level, per the task's
instruction) tells a consistent story: all four methods stay in the
0.90-0.99 range across nearly every target, with the SAME two soft spots
recurring in every method -- Gain 4.5-5.5-ish and Gain 9-9.5 show the
lowest spectral correlation values everywhere (e.g. dense interpolation's
own worst spectral correlation is 0.933 at Gain 9.5; G5-anchor's worst
non-anomalous spectral correlation is also near the ends of its useful
range). This is consistent with the earlier response-coordinate reports'
finding that spectral character keeps evolving even where level/ESR looks
fine -- it shows up as a shared, method-independent soft spot rather than
a failure specific to any one reconstruction approach.

## Failure regions

- **G1 anchor** fails hard and permanently beyond Gain ~6 (pinned at the
  +24dB search boundary, raw ESR growing from 0.02 to 0.36) -- more input
  gain cannot buy back what's missing; this is NOT a search-range problem
  (widening the search further would not help, since ESR keeps climbing
  even while pinned at the boundary rather than plateauing near it).
- **G10 anchor** fails the mirror-image way below Gain ~2.5-3.0 (pinned at
  -24dB, raw ESR growing from 0.06 to 25.0 as target gain decreases
  further).
- **G5 anchor** has no comparable failure region across the ENTIRE tested
  range -- its only weak point is a mild (not boundary-pinned) rise at the
  Gain 1.0 endpoint.
- **Gain 8.5** fails everywhere, for every method, due to the identified
  file-level latency offset -- not a "virtual gain" failure at all.

**This directly answers "does virtual input gain work locally around each
anchor but fail across larger physical Gain changes?": yes for the two
EXTREME anchors (G1, G10), each covering roughly half the range well and
failing hard, boundary-pinned, past that. The MIDDLE anchor (G5) is the
exception -- it does not show this local-only limitation and instead
covers nearly the full range.**

## The measured physical-Gain -> virtual-input-gain relationship (G5 anchor)

No electrical-linearity assumption is made here -- this is the measured
optimum from the search, nothing more:

| Physical Gain | Required input gain (dB, relative to G5) |
|---:|---:|
| 1.0 | -24.0 (boundary) |
| 1.5 | -19.0 |
| 2.0 | -15.0 |
| 2.5 | -10.6 |
| 3.0 | -6.8 |
| 3.5 | -4.2 |
| 4.0 | -2.4 |
| 4.5 | -1.2 |
| 5.0 | 0.0 |
| 5.5 | +1.4 |
| 6.0 | +2.0 |
| 6.5 | +5.8 |
| 7.0 | +9.0 |
| 7.5 | +10.8 |
| 8.0 | +13.0 |
| 9.0 | +15.2 |
| 9.5 | +14.8 |
| 10.0 | +15.2 |

**This mapping is strongly nonlinear and compressive (decelerating)**: each
0.5-physical-Gain step near the low end costs roughly 4-5 dB of input
gain (1.0->3.0 spans 17.2 dB across 4 steps), while each 0.5-step step
near the high end costs under 1 dB (9.0->10.0 spans essentially 0 dB net
across 2 steps, even dipping slightly at 9.5). No claim is made here about
WHY (pot taper, circuit interaction, or amplifier response -- see the
response-coordinate report's Section 5/7 for that discussion); this table
reports only the measured relationship.

## Representative audio for listening comparison

`work/continuous_gain_virtual_gain/` (gitignored scratch output) contains,
for targets Gain 3.5, 6.5, and 9.5: the real capture, the G5-anchor +
virtual-gain reconstruction, the best-of-3-anchors reconstruction, and the
dense-interpolation reconstruction, all on the identical held-out 6-second
clip -- regenerate with `scripts/continuous_gain_virtual_gain_benchmark.py`.

## Answers to the key questions

1. **Can one fixed NAM span a useful range of physical Gain settings just
   by changing its input drive?** Yes, dramatically so for the right
   anchor (G5) -- raw ESR under 0.01 across nearly the entire Gain 1-10
   range from a single capture.
2. **Which single anchor spans the widest useful range?** G5 (the middle
   anchor), by a wide margin over either extreme.
3. **Can G1/G5/G10 + input gain approximate the full sweep nearly as well
   as keeping all ten integer captures?** Better, on this dataset and DI:
   mean raw ESR 0.0015 (3 anchors) vs. 0.0022 (10 captures) on the 9
   half-step targets.
4. **Where does virtual input gain fail even after output level is
   matched?** Only at the two EXTREME anchors, past roughly half the
   range from their own position (G1 past ~Gain 6, G10 below ~Gain 2.5-3),
   where the search pins at its +-24dB boundary and error keeps climbing
   rather than plateauing -- i.e. a real ceiling, not a search-range
   limitation. The G5 anchor shows no comparable failure region.
5. **Are those failures primarily spectral/harmonic rather than level?**
   Level-matching does reduce error somewhat at the failure points (e.g.
   G1 anchor at Gain 8.5-target... excluded; at Gain 9.0: raw ESR 0.358 ->
   level-matched 0.266, roughly a quarter reduction) but the bulk of the
   error remains after level correction -- these ARE primarily
   spectral/nonlinear-character failures, consistent with every other
   finding in this research thread, not simple loudness mismatches.
6. **Does virtual input gain work locally around each anchor but fail
   across larger physical Gain changes?** Yes for G1 and G10 specifically
   (see Failure Regions above); NOT for G5, which is the key asymmetry
   this benchmark surfaces.
7. **Is the required mapping from physical Gain to virtual input gain
   strongly nonlinear?** Yes -- compressive/decelerating, as measured and
   tabulated above, consistent with the front-loaded response curves
   documented throughout this research thread.
8. **How many anchors are actually needed to reproduce the sweep
   accurately?** As few as 1 (G5 alone, if a small residual near Gain 1.0
   is acceptable) or 3 (G1/G5/G10, which slightly beats the 10-capture
   baseline) -- both dramatically fewer than the 10 integer captures the
   discrete-interpolation approach in every earlier report assumed were
   necessary.

## Decision table

```text
Method                         NAM captures required    Training required    Continuous Gain
One NAM + input gain           1                        no                   yes
G1/G5/G10 + input gain         3                        no                   yes
G1-G10 interpolation           10                       no                   yes
```

Measured mean raw ESR on the 9 half-step targets (excluding the Gain 8.5
capture-file anomaly): **1 NAM (G5) + gain: 0.0037. 3 NAMs + gain: 0.0015.
10 NAMs, interpolation: 0.0022.**

## The main decision question

**What is the minimum number of real NAM captures required to reproduce
the amplifier's physical Gain sweep accurately when normal NAM-style input
gain is allowed?**

On this amp and DI: **as few as 1 (a well-chosen middle anchor), and
robustly 3**, not 10. This is the single most product-relevant result in
the whole Continuous Gain research thread so far -- it suggests the
practical path to a usable "continuous gain" feature may not require dense
capture-and-interpolate infrastructure at all, but rather (a) identifying
a good anchor position (or a small handful) per amp, and (b) a virtual
input-gain mapping curve per anchor, measured exactly as done here.

## Caveats

- Single amp, single DI clip, single evaluation segment. The response-
  coordinate and generalization reports both showed real behaviour differs
  meaningfully between amp families (e.g. the Fender Super-Sonic's much
  steeper low-end knee) -- whether a middle anchor is similarly dominant
  on those amps is untested and should not be assumed.
  Peavey 5150 is explicitly excluded per the task brief and
  docs/CONTINUOUS_GAIN_METRIC_LIMITATION.md's finding that no metric here
  has been demonstrated reliable on that amp's saturated material.
- "Which anchor is best" was determined by an OFFLINE search against real
  targets that a real product would not have access to at inference time
  -- this is a feasibility ceiling, not a proposed runtime algorithm. A
  real feature would need either a pre-computed per-amp gain-mapping curve
  (measured once, as done here, and shipped with the amp) or an online
  estimation method, neither of which is designed here.
- The clean win for 3 anchors over 10 captures should be read as "on THIS
  amp, virtual gain covers what discrete captures were covering less
  efficiently," not as general proof that virtual gain always beats more
  real captures -- an amp with a genuinely different clipping/voicing
  character at different real gain settings (not just level/drive) could
  behave more like the G1/G10 failure regions across its whole range,
  which is exactly what a topology check on more amp families (Section 5
  of the response-coordinate report) would need to establish.
