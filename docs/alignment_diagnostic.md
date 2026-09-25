# A/B timing diagnostic

NAM Mixer can measure whether Amp A and Amp B have a **fixed time offset**
between them. It reports the result but **never corrects it**: alignment stays
off in every mode (`align_enabled=False`), and preview and training targets
are unchanged.

Code: `hybrid/core/align_diagnostic.py` (`analyse_alignment`). Tests:
`tests/test_align_diagnostic.py`. It runs once per `/api/render_pair` on the
two renders that call has just produced, so it needs no extra NAM inference.
The result is returned as `alignment_diagnostic`, and the Parallel Blend and
Dynamic Hybrid panels show it as a one-line **Timing** readout.

## Fixed offset vs. phase response

A fixed offset (latency) delays **every frequency by the same amount**, so it
shifts the whole signal and does not depend on what is being played. Shifting
Amp B by whole samples is a correct fix for that, and only for that.

Two different amps also have different **phase responses**: their filtering,
distortion, compression and cabinet delay different frequencies by different
amounts. That is part of their sound, not a timing error. Cross-correlation
cannot tell the two apart. The lag that best matches two differently-voiced
signals depends on how much energy the material has at each frequency, so it
moves when the material changes. `estimate_offset` returns that best-matching
lag, which is why the old aligner could "correct" a tone difference.

## Method

1. **Choose regions.** The diagnostic picks up to 12 non-overlapping
   0.25-second regions from the shared DI. Each one starts 10 ms before one of
   the DI's strongest note onsets: a 10 ms frame that rises at least 6 dB above
   the three frames before it and is above -50 dBFS. The selection is
   deterministic. Regions where either amp render is below -60 dBFS are
   skipped.
2. **Measure each region.** It runs `estimate_offset` on each region
   separately, searching ±256 samples (5.3 ms at 48 kHz) rather than the
   whole-render default of 2000. It also records the correlation at the chosen
   lag.
3. **Decide.** A region counts only if its correlation is at least 0.9 and the
   peak is not at the edge of the search range.
   - Fewer than 4 regions with signal → `insufficient_signal`.
   - Fewer than 4 regions that count → `ambiguous`: the amps differ too much
     for a sample shift to mean anything.
   - Otherwise, take the median lag of the regions that count. If fewer than
     80% of all regions with signal are within ±1 sample of it → `ambiguous`.
   - A median within ±1 sample of zero → `aligned`. Anything else →
     `fixed_offset`, with that median as the recommendation. A median that
     falls exactly between two lags is rounded toward zero.

The whole-render `estimate_offset` value is included in the result as
`whole_render_offset_samples`, for comparison only. It is never used for a
decision.

## Real-capture results (2026-09-25)

These are 12 pairs × 4 bundled DIs, rendered with the real `nam_render` and
`calibration_mode="auto"`. The captures are personal and not in the repo.
Every real capture is A2 (`SlimmableContainer`). The WaveNet and LSTM files
are NAMCore's bundled example models, included only to cover other
architectures. The **+7 check** re-runs the analysis after delaying Amp B's
render by exactly 7 samples.

| Pair | DI | Status | Whole-render | Per-region offsets | Median corr. | +7 check |
|---|---|---|---|---|---|---|
| same model (control) | clean_mayer | aligned | +0 | all +0 | 1.00 | fixed_offset +7 |
| same model (control) | moderate_brit | aligned | +0 | all +0 | 1.00 | fixed_offset +7 |
| same model (control) | high_thrash | aligned | +0 | all +0 | 1.00 | fixed_offset +7 |
| same model (control) | moderate_hotrod | aligned | +0 | all +0 | 1.00 | fixed_offset +7 |
| 6505+ Clean G1 vs G2 | clean_mayer | aligned | +1 | +1 ×10, +0 ×2 | 0.96 | fixed_offset +8 |
| 6505+ Clean G1 vs G2 | moderate_brit | aligned | +1 | +0/+1 | 0.95 | fixed_offset +7 |
| 6505+ Clean G1 vs G2 | high_thrash | aligned | +1 | +1 ×11, +0 | 0.97 | fixed_offset +8 |
| 6505+ Clean G1 vs G2 | moderate_hotrod | aligned | +1 | all +1 | 0.96 | fixed_offset +8 |
| 6505+ MidFwd G5 vs Scooped G5 | all four | aligned | +0 | all +0 | 0.99 | fixed_offset +7 |
| DG2 NoCab vs DG1 | all four | aligned | +0 | all +0 | 0.99 | fixed_offset +7 |
| Deluxe Reverb vs Twin Reverb | clean_mayer | ambiguous | -3 | -3 ×10, -2 ×2 | 0.58 | ambiguous |
| Deluxe Reverb vs Twin Reverb | moderate_brit | ambiguous | -3 | -9, -4..-2 | 0.60 | ambiguous |
| Deluxe Reverb vs Twin Reverb | high_thrash | ambiguous | -6 | -8..-5 | 0.58 | ambiguous |
| Deluxe Reverb vs Twin Reverb | moderate_hotrod | ambiguous | -7 | -8..-6 | 0.64 | ambiguous |
| Deluxe Reverb vs Mesa Dual G6 | clean_mayer | ambiguous | -38 | +252, +5, +4, +254, -35, -23, -40, -20 … | 0.49 | ambiguous |
| Deluxe Reverb vs Mesa Dual G6 | moderate_brit | ambiguous | +3 | +256, +2..+5 | 0.49 | ambiguous |
| Deluxe Reverb vs Mesa Dual G6 | high_thrash | ambiguous | +0 | -4..+3 | 0.35 | ambiguous |
| Deluxe Reverb vs Mesa Dual G6 | moderate_hotrod | ambiguous | +221 | -2..+4 and +205..+239 | 0.41 | ambiguous |
| AC30 clean vs 5150 boosted | clean_mayer | ambiguous | +462 | -80..+193 | 0.26 | ambiguous |
| AC30 clean vs 5150 boosted | moderate_brit | ambiguous | +464 | -103..+222 | 0.23 | ambiguous |
| AC30 clean vs 5150 boosted | high_thrash | ambiguous | +467 | -223..+197 | 0.17 | ambiguous |
| AC30 clean vs 5150 boosted | moderate_hotrod | ambiguous | +467 | -186..+147 | 0.35 | ambiguous |
| Bassman vs JCM800 RAT | clean_mayer | ambiguous | +28 | +27/+28 | 0.38 | ambiguous |
| Bassman vs JCM800 RAT | moderate_brit | ambiguous | +30 | +28..+37 | 0.37 | ambiguous |
| Bassman vs JCM800 RAT | high_thrash | ambiguous | +34 | +30..+43 | 0.36 | ambiguous |
| Bassman vs JCM800 RAT | moderate_hotrod | ambiguous | +32 | +32..+38 and -203..-148 | 0.28 | ambiguous |
| DG2 NoCab vs DG2 (amp_cab) | clean/brit/thrash | ambiguous | +2 | all +2 (one +1) | 0.70–0.77 | ambiguous |
| DG2 NoCab vs DG2 (amp_cab) | moderate_hotrod | ambiguous | +1 | +1/+2 and ±170..256 | 0.66 | ambiguous |
| Deluxe Reverb vs "Baked In" (amp_cab) | all four | ambiguous | +69..+74 | +23..+92, -8/-7 | 0.44–0.63 | ambiguous |
| A2 Deluxe vs WaveNet example | clean_mayer | ambiguous | +12 | +11/+12 | 0.67 | ambiguous |
| A2 Deluxe vs WaveNet example | moderate_brit | ambiguous | +11 | +6, +10..+12 | 0.82 | ambiguous |
| A2 Deluxe vs WaveNet example | high_thrash | ambiguous | +8 | +1..+9 | 0.62 | ambiguous |
| A2 Deluxe vs WaveNet example | moderate_hotrod | ambiguous | +9 | +5..+10 | 0.61 | ambiguous |
| A2 Deluxe vs LSTM example | clean_mayer | aligned | -2 | -1 ×6, -2 ×6 | 0.95 | fixed_offset +5 |
| A2 Deluxe vs LSTM example | moderate_brit | aligned | -1 | all -1 | 0.94 | fixed_offset +6 |
| A2 Deluxe vs LSTM example | high_thrash | aligned | -1 | -2..0 | 0.92 | fixed_offset +6 |
| A2 Deluxe vs LSTM example | moderate_hotrod | ambiguous | -1 | -2..0 (3 regions < 0.9) | 0.93 | ambiguous |

### Findings

- **No real pair showed a fixed offset.** Pairs that are genuinely similar are
  `aligned` on every DI (0 or +1 sample). A 7-sample delay added on purpose is
  detected for each of them.
- **Consistent is not the same as correct.** In Deluxe Reverb vs Twin Reverb,
  every region of a clip agrees closely, yet the agreed lag moves with the
  material: -3, -3, -6 and -7 samples across the four DIs. A real latency
  cannot do that. The first version of the diagnostic, which only required
  0.5 correlation, reported `fixed_offset` for this pair with a different
  "offset" on each clip. That is why the threshold is now 0.9. The same
  pattern appears in DG2 with vs. without its cab (+2, consistent with the
  cab's own phase delay) and in the WaveNet example (+6..+12).
- **The old whole-render estimate disagrees badly on dissimilar amps.** It
  returned +462..+467 samples (about 9.7 ms) for AC30 clean vs 5150, +221 for
  Deluxe vs Mesa on one clip, +28..+34 for Bassman vs JCM800 and +69..+74
  against a cab-baked capture. If enabled, it would have shifted Amp B by
  those amounts. The multi-region diagnostic reports all of them as
  `ambiguous`.
- **Limitation:** for pairs below the 0.9 correlation threshold, the
  diagnostic cannot detect a real latency either. It declines to judge rather
  than guess.

### Is automatic correction safe?

**No, not on the current evidence.** No real capture pair showed a stable
non-zero offset. The only pairs the diagnostic can judge (correlation ≥ 0.9)
are ones where a shift of a sample or two makes no audible difference.
Reconsider only if a real pair is found that is `fixed_offset` with the same
value on several different DIs. Even then, correction would need a fixed,
recorded offset applied identically in preview, the frozen design, target
generation and validation. Today, with alignment enabled, `training_target`,
`blend_training_target` and `validation` each call `align_to_reference`, which
re-estimates the offset on whatever audio it is given. It does not reuse the
offset that was auditioned.
