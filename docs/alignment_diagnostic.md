# A/B timing: diagnostic and fixed offset correction

NAM Mixer measures whether Amp A and Amp B have a **fixed time offset**
between them. When that offset is verified across several independent DIs, it
offers an optional **fixed timing offset correction**: Amp B shifted by one
integer number of samples. Original timing is always the default. Nothing is
corrected automatically, and general amp phase response is never treated as
latency.

Measurement and application are completely separate:

| Stage | Code | Measures? |
|---|---|---|
| Preview diagnostic | `hybrid/core/align_diagnostic.py` `analyse_alignment` | yes |
| Cross-DI verification | `hybrid/core/align_verification.py` `verify_fixed_offset_across_dis` | yes |
| Preview (Parallel + Hybrid), live stems | `build_fixed_blend` / `build_hybrid` via `apply_fixed_offset` | no |
| Frozen design | `alignment_enabled`, `alignment_offset_samples`, `alignment_method` | no |
| Target generation, teacher reconstruction, validation, completed-model comparison | `frozen_alignment_offset(design)` + `apply_fixed_offset` | no |

`tests/test_align_fixed_offset.py::test_no_production_module_calls_an_estimator`
fails if any module other than the two measurement modules calls
`estimate_offset` or `align_to_reference`.

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

## Diagnostic results on real captures (2026-09-25)

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

## Correction semantics

`apply_fixed_offset(audio, offset_samples, output_length)` uses the same sign
as `estimate_offset(amp_a, amp_b)`:

- **positive** = Amp B lags (arrives later than) Amp A; the correction trims
  that many samples from the start of Amp B;
- **negative** = Amp B leads; the correction pads the start of Amp B with zeros;
- the result is always exactly `output_length` samples, with the input's dtype.

Amp A is never shifted and neither capture is altered. The integer is chosen
once, by cross-DI verification, and then:

1. `/api/render_pair` returns it as `timing_correction.offset_samples` (only
   when verified) and keeps it in the render snapshot;
2. the UI offers **Original / Corrected** (default Original); Corrected sends
   `alignment_enabled: true` plus that exact integer, and the server rejects
   any other value or any request against an unverified render;
3. switching re-blends the cached renders; no NAM inference is run;
4. `/api/generate` freezes `alignment_enabled`, `alignment_offset_samples` and
   `alignment_method: "fixed-frozen-offset"` into the design, plus the
   diagnostic's report as `alignment_diagnostic` (provenance only);
5. target generation and every teacher reconstruction apply the frozen integer
   verbatim. Manifests record it as:

   ```json
   "alignment_correction": {"applied": true, "offset_samples": 7,
     "method": "fixed-frozen-offset",
     "description": "fixed timing offset correction: ... not a phase correction"}
   ```

**Sessions.** A saved session stores the user's timing *intent*, not an
applied correction: `settings.timing = {"choice": "original" | "corrected",
"offsetSamples": N | null, "method": "fixed-frozen-offset"}` (a generated
bundle's session is derived from its frozen design; a legacy enabled design is
shown as Original). Applying a session's settings applies no correction; the
Load action then prepares the amps again, and that render (like any later render) of the same Amp A, Amp B and DI runs the normal diagnostic and
cross-DI verification, and Corrected is restored only if that render verifies
exactly the saved N. Otherwise the choice stays Original with a note, e.g.
"Saved timing correction was +7 samples, but this render no longer verifies
that fixed offset. Original timing has been restored." A different Amp A/B/DI,
any later render, and any source change reset it to Original as before. Old
sessions without `timing` load exactly as before (Original).

**Backward compatibility.** Unaligned designs (`alignment_enabled: false`)
produce bit-for-bit identical targets: regenerating the bundle-generator
golden snapshots changed nothing except two added manifest keys
(`alignment_correction`, `alignment_diagnostic`), and every audio hash stayed
the same. Old design files without the new fields still load. A design with
`alignment_enabled: true` but no `alignment_method` would date from before
frozen offsets, when the offset was re-estimated on each audio it was applied
to. Such a design is refused (`LegacyAlignmentDesignError`), not reinterpreted.
None exists: every design found in `work/` and the desktop app's data (17), and
in the test fixtures, has alignment disabled.

## Cross-DI verification

The preview DI's diagnostic can agree region-to-region on a lag that is really
phase response (Deluxe vs Twin, above). Verification therefore re-renders both
amps, with exactly the inputs the preview used, on a fixed set of three bundled
DIs: `clean_mayer`, `moderate_brit`, `high_thrash`, or `moderate_hotrod` in
place of whichever of those is the preview DI. Each clip is capped at 20 s. The
same multi-region diagnostic runs on each. An offset is **verified** only if:

- the preview DI reports `fixed_offset`;
- every verification DI with enough signal reports `fixed_offset` (a single
  `aligned` or `ambiguous` DI rejects it), and at least 2 do;
- all those offsets, including the preview's, are within ±1 sample of one
  consensus integer (their median, halves rounded toward zero).

The whole-render estimate is never used, not even as a fallback.
Verification only runs when the preview DI already shows a fixed offset, so
it costs nothing for most pairs. When it runs, it takes about 3.3 s (6 NAM
renders of 20 s). Per-DI measurements are cached by source-model content hash
and render settings, and the verdict is recomputed from them each time. The
preview diagnostic itself adds 50–130 ms to a render.

## Acceptance on real captures (2026-09-25)

**Artificial truth.** Amp B's real NAM render was delayed by exactly N samples
at the `render()` seam, so preview, verification, target generation and
validation all saw a model with genuine latency N (official V3 training input
for the targets):

| Pair | +1 | +7 | +32 |
|---|---|---|---|
| 6505+ MidFwd G5 vs Scooped G5 | aligned (not offered) | verified +7 | verified +32 |
| DG2 NoCab vs DG1 | aligned (not offered) | verified +7 | verified +32 |
| 6505+ Clean G1 vs G2 | aligned (not offered) | verified +8 | verified +33 |

- +1 is within the ±1-sample "aligned" tolerance, so it is deliberately not
  offered.
- 6505+ Clean G1 vs G2 has its own natural lag of about one sample (it measured
  0/+1 without injection), so the true totals are about +8 and +33. Correcting
  by 8 or 33 gives the undelayed model advanced by exactly 1 sample, bit for
  bit, in preview and in the generated target.
  This is intended, not an off-by-one: the ±1 "aligned" tolerance only
  decides whether a pair *on its own* is offered a correction. Once a larger
  fixed offset verifies, the correction applies the complete verified
  relationship (natural +1 plus the added +7 = +8), never just the part that
  was added. `tests/test_align_verification.py::test_natural_one_sample_lag_is_part_of_the_verified_total`
  locks this in.
- For the other pairs, the corrected preview equals the undelayed render bit
  for bit, the generated target equals the undelayed target (max difference
  0.0), and the Parallel and Hybrid teacher reconstructions equal the
  corrected preview (max difference 0.0). The design and manifest carry the
  same integer.

**Natural pairs** (no injection), each tried with two preview DIs:

- **Never offered a correction:** AC30 vs 5150, Deluxe vs Mesa, Deluxe vs the
  cab-baked capture, Bassman vs JCM800, Deluxe vs Twin, DG2 with/without cab,
  and the WaveNet example. All are `ambiguous`, so verification never runs.
- **`aligned`:** 6505+ Clean G1 vs G2 and the LSTM example.

Listening files (not committed) are in
`work/alignment_acceptance/*_B_late_{7,32}_{original,corrected}.wav`.

## Is it safe to ship?

**As an optional, default-off choice: yes.** It is only offered when three
independent DIs plus the preview DI agree on one offset. Every downstream
stage provably applies the auditioned integer. No natural pair tested was
offered a correction. **Automatic correction: no.** It stays off. No real
capture pair has shown a genuine fixed offset, so the feature's practical
value is still unproven. It is a safety net for captures with real latency,
such as a mis-set latency calibration in a capture.
