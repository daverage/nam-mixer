# Continuous Gain Model -- Cross-Amp Generalization (doc Q10)

Follow-up to docs/CONTINUOUS_GAIN_PHASE2_RESULTS.md, answering
docs/CONTINUOUS_GAIN.md's Q10: "Does the technique generalise across clean
amps, edge-of-breakup amps, high-gain amps...?" Produced with
`scripts/continuous_gain_baseline_matrix.py` (now generalized with a
`--gain-prefix` option so it isn't tied to Marshall-style "G<n>" filenames).

## Source dataset

Fender Super-Sonic 60W Head mk.1, Vibrolux channel, flat EQ (T5 B5), Volume
1 through 10 -- a genuine full-integer-step sweep (spacing 1, the finest
resolution available for this amp; no `input_level_dbu` metadata). This is a
clean-into-breakup Fender voicing, a different amp family and a different
control (Volume, not Gain) from the two JCM800 datasets used in
CONTINUOUS_GAIN_PHASE2_RESULTS.md.

## The gain curve is far more front-loaded than either JCM800 dataset

Active-RMS progression (standard DI, 6s):

| Volume | Active RMS (dBFS) | Δ vs. previous |
|---:|---:|---:|
| 1 | -41.10 | |
| 2 | -36.18 | +4.92 |
| 3 | -27.68 | **+8.50** |
| 4 | -23.86 | +3.82 |
| 5 | -22.39 | +1.48 |
| 6 | -21.48 | +0.91 |
| 7 | -20.42 | +1.05 |
| 8 | -19.68 | +0.75 |
| 9 | -19.23 | +0.45 |
| 10 | -19.15 | +0.08 |

Volume 2->3 alone accounts for an 8.5 dB jump -- roughly a third of the
entire V1->V10 range in one step -- then the curve decelerates steadily. The
two JCM800 datasets both had a single, comparatively gentler knee; this
amp's low end is a much sharper clean-headroom-clipping onset.

## Leave-one-out validation (native spacing-1, all already the finest
available resolution)

| Hidden | Neighbours | standard | clean | metalcore |
|---:|---|---:|---:|---:|
| 2 | V1/V3 | 0.0647 | 0.0728 | 0.0660 |
| 3 | V2/V4 | 0.0503 | 0.0634 | 0.0450 |
| 4 | V3/V5 | 0.0236 | 0.0331 | 0.0194 |
| 5 | V4/V6 | 0.0058 | 0.0062 | 0.0054 |
| 6 | V5/V7 | 0.0069 | 0.0063 | 0.0062 |
| 7 | V6/V8 | 0.0077 | 0.0075 | 0.0076 |
| 8 | V7/V9 | 0.0062 | 0.0062 | 0.0060 |
| 9 | V8/V10 | 0.0052 | 0.0049 | 0.0045 |

(values are raw ESR)

## Findings

1. **DI material still doesn't break the baseline** -- standard/clean/
   metalcore stay within a similar band at every hidden position, same as
   both JCM800 datasets. This part of the earlier conclusion generalizes
   cleanly.
2. **Wide capture spacing is harmful here too, and MORE severely.**
   Predicting V6 (a flat-region point, not even the knee) from its native
   neighbours V5/V7 gives raw ESR 0.0069; from the extreme endpoints V1/V10
   it jumps to **0.1432** -- roughly 20x worse, a larger multiplier than
   either JCM800 case. Endpoint-to-endpoint interpolation crossing this
   amp's much sharper low-end nonlinearity is especially punishing.
3. **Level is still not the bottleneck.** The level-matched-oracle ablation
   at V6 makes raw ESR slightly WORSE (0.0069 -> 0.0083) despite fixing the
   RMS delta almost exactly (0.0001 -> 0.0009 diff) -- consistent with both
   JCM800 findings that the residual error is shape/harmonic, not level.
4. **Divergence from the Marshall density result -- the important new
   finding.** On the "Marshall JCM800 2203 - updated" dataset, tightening
   the bracket to the sweep's native (finest available) spacing fully closed
   the gap: knee-adjacent raw ESR fell to within the same 0.003-0.008 band
   as the flat region. Here, the SAME thing -- native, finest-available
   spacing-1 sampling -- leaves Volume 2/3/4 at raw ESR 0.02-0.07, still
   **3-10x higher** than the flat-region baseline (~0.005-0.008), even
   though this amp already has denser relative sampling (every integer step,
   vs. every-other on the original 1985 JCM800 unit). Capture density is not
   a universal fix at any fixed step size -- how MUCH density is enough
   depends on how sharp the underlying nonlinearity is. This amp's clean-to-
   breakup transition is steep enough that Volume-2-to-Volume-3 alone (one
   native step) still crosses too much of the curve's total change for
   linear interpolation to track it well.

## Revised product/Phase-3 implication

The Phase 2 report's conclusion ("capture placement matters more than
interpolation sophistication") still holds directionally, but needs one
addition:

> Capture placement should be guided by how much the measured level (or
> other per-capture metric) changes PER STEP, not by a fixed capture count
> or fixed knob-position spacing. A UI that surfaces the same
> active-RMS-progression measurement used here (locate the largest
> per-step jump) could tell a user not just WHERE to add a capture, but
> whether one more capture is likely enough, or whether the region is steep
> enough that even that may not fully close the gap -- exactly what
> distinguishes this Fender dataset's Volume 2-3-4 band from the Marshall
> knee that fully resolved with one extra capture.

This does not change the Phase 2 report's recommendation to avoid a broad
Hybrid/Character search -- both datasets still show the baseline performing
well away from their respective knees, and neither shows the interpolation
METHOD itself (as opposed to capture placement) as the dominant error
source. It does mean the "one more capture is probably enough" framing from
the Marshall result should not be treated as universal without checking the
per-step level-change magnitude first.
