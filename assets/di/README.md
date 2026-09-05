# Guitar/Bass DI Library

These WAV files were copied (not moved) from the [NAMtoClo](https://github.com/Goaltoday/NamtoClo) fork
at `resources/reference_clips/` — the bundled "Tone Match" reference clips that ship inside `NamToClo.exe`.
They were **not** written for this project; NAMtoClo uses them as post-conversion reference targets for
its own CLO fitting process. We are repurposing them here because they are already genre/gain-labeled,
already committed (not gitignored) in that repo, and already known-good real DI/performance material.

**These are copies. The originals in NAMtoClo are untouched.**

## Provenance and what we actually know

NAMtoClo's own source (`src/native_converter.hpp`) documents the naming convention: files are named
`<bucket>_<style>.wav` where bucket is one of `clean` / `moderate` / `high` / `bass`, matching an
automatic gain-bucket classifier (`AmpGainBucket::{Clean,Moderate,High}`) used to pick a reference
clip for a NAM model's fitted gain character. The `bass_*` clips are always an explicit user choice in
NAMtoClo, never auto-selected, because the classifier "does not distinguish bass from guitar."

The style suffixes (`mayer`, `smooth`, `brit`, `hotrod`, `metalcore`, `thrash`, `downtown`, `rollin`)
match filenames in NAMtoClo's gitignored `test_assets/tone3000_inputs/` directory (e.g. `Mayer - Guitar.wav`,
`Brit - Guitar.wav`, `Fast Thrash - Guitar.wav`, `Downtown - Bass.wav`), which strongly suggests these
clips originate from [Tone3000](https://www.tone3000.com/)'s standard DI test set, trimmed/renamed for
bundling. This is **inferred**, not confirmed — no NAMtoClo commit message or comment states this
explicitly, and the original tone3000_inputs are not present in this repository (that directory is
gitignored in NAMtoClo and explicitly not for republishing).

All genre, playing-style and "likely amp family" fields below are **inferred** from the filename alone
(and, where noted, from the tone3000_inputs filename it likely maps to) — none of this project's authors
have critically listened to confirm playing technique or amp target. Where a field cannot reasonably be
inferred, it says `unknown`.

## Why these clips, and what they are NOT

This DI library exists for **preview, auditioning, crossover analysis, automatic level-match testing,
and regression testing** of the hybrid blend engine in this project. It is explicitly **not** the
official NAM training/reamping signal, and hybrid target generation for actual A2 training must use a
proper calibrated reamp/DI signal (see the main project README, "Preview DIs vs. NAM training material").

## Files

All 8 files were copied as-is; no filenames were changed.

| Filename | Genre / style (inferred) | Playing type (inferred) | Expected gain range | Likely amp family (inferred) | Dynamic range | Best use | Crossover usefulness | Notes |
|---|---|---|---|---|---|---|---|---|
| `clean_mayer.wav` | Blues/pop clean, likely maps to tone3000 `Mayer - Guitar.wav` | Estimated: single-note lines and chords, moderate touch dynamics (John Mayer-style clean playing) | Clean | Fender/boutique clean (Dumble-adjacent) — estimated | High (19.3 dB p10-p90) | Clean-side auditioning; touch-dynamics testing | High — wide dynamic range makes it good for clean→breakup crossover testing | Longest silence-free clean clip in the set |
| `clean_smooth.wav` | Clean, likely maps to tone3000 `Smooth - Guitar.wav` | Estimated: sustained chords/arpeggios, smooth legato phrasing | Clean | Fender/Vox-style clean — estimated | High (16.3 dB p10-p90) | Longest clip (47.8s) in the set; good for sustained-note and long-form crossover behavior checks | Medium-high — long duration is useful for consistent-level as well as dynamic testing | Longest file overall (47.75s); ~1.5s trailing silence |
| `moderate_brit.wav` | British rock, likely maps to tone3000 `Brit - Guitar.wav` | Estimated: rhythm chords and riffs, "British" (Marshall/Vox-voiced) character | Edge-of-breakup / crunch | Marshall/Vox-style — estimated from name | Very high (29.6 dB p10-p90, highest of all 8 files) | Clean→crunch and crunch→high-gain crossover testing; best dynamic-range file in the set | Very high — largest measured dynamic range makes this the strongest crossover-transition test clip | Longest clip after `clean_smooth.wav` (41.6s); large dynamic range may include quiet passages plus loud hits |
| `moderate_hotrod.wav` | Rock/blues-rock, likely maps to tone3000 `Hotrod - Guitar.wav` | Estimated: rhythm playing, possibly harder-picked than `brit` (name suggests Fender Hot Rod-series amp) | Crunch | Fender Hot Rod series (DeVille/Deluxe) — estimated from name | Medium (15.9 dB p10-p90) | Crunch-range auditioning | Medium — decent dynamic range but not as extreme as `moderate_brit` | Only file with RMS slightly off the -28 dBFS norm (-28.6 dBFS) and the highest peak of the set (-8.0 dBFS) → highest measured crest factor (20.6 dB) |
| `high_metalcore.wav` | Modern metal/metalcore, likely maps to tone3000 `Metalcore - Guitar.wav` | Estimated: palm-muted chugging rhythm, tight low-end riffing | High gain | Mesa/EVH/5150-style — estimated from genre name | Medium (10.7 dB p10-p90) | High-gain auditioning; consistent-level rhythm testing (palm-muting tends to be fairly uniform in level) | Low-medium for crossover transitions (narrower dynamic range than the clean/moderate clips), but good for verifying the hybrid does NOT flutter under sustained high-gain rhythm | Shortest high-gain clip (20.4s) |
| `high_thrash.wav` | Thrash metal, likely maps to tone3000 `Fast Thrash - Guitar.wav` | Estimated: fast palm-muted riffing / picking, possibly leads | Very high gain | Marshall JCM800/5150/Peavey-style — estimated from genre name | High (16.6 dB p10-p90) — surprisingly dynamic for "thrash" | High-gain auditioning; crunch→high-gain crossover testing given its above-average dynamic range for the high-gain bucket | Medium-high — most dynamic of the two high-gain clips, useful for crunch→high-gain transition testing | Shortest file overall (17.75s); negligible lead-in silence (17ms) |
| `bass_downtown.wav` | Funk/groove bass, likely maps to tone3000 `Downtown - Bass.wav` | Estimated: funk-style groove playing (name suggests slap/pop or syncopated fingerstyle) | Clean-to-moderate (bass amps classify differently — see note above) | Ampeg/Fender bass amp-style — estimated | Medium (14.8 dB p10-p90) | Bass-specific amp auditioning only; NOT for guitar crossover default tuning | Low for guitar A→B crossover work; useful only if the hybrid pipeline is later extended to bass hybrids | NAMtoClo's own docs state bass reference clips are never auto-selected by its gain classifier — treat similarly here: don't use for guitar crossover calibration |
| `bass_rollin.wav` | Rock/groove bass, likely maps to tone3000 `Rollin' - Bass.wav` | Estimated: rock groove, driving eighth-note or root-note patterns | Clean-to-moderate | Ampeg/SVT-style — estimated | Lower (8.7 dB p10-p90, lowest of all 8 files) | Bass-specific amp auditioning; consistent-level rhythm testing (lowest measured dynamic range = most uniform level) | Low for guitar crossover work | Longest lead-in silence of the set before signal starts (0.353s) |

## Technical analysis

Generated with `scripts/analyze_di.py` (NumPy + `soundfile`; see that script for the exact method —
50ms-window RMS envelope, 10th/90th percentile spread for "dynamic range", peak vs. RMS for crest factor,
and a -50 dBFS-relative-to-peak threshold for silence detection). Raw output is also saved as
`assets/di/_analysis.json`.

| Filename | Sample rate | Channels | Duration (s) | Peak (dBFS) | RMS (dBFS) | Crest factor (dB) | Dynamic range, p10–p90 (dB) | Lead silence (s) | Trail silence (s) | Compressed or dynamic? |
|---|---|---|---|---|---|---|---|---|---|---|
| clean_mayer.wav | 48000 | 1 | 25.94 | -12.77 | -28.00 | 15.23 | 19.28 | 0.082 | 0.482 | Dynamic |
| clean_smooth.wav | 48000 | 1 | 47.75 | -9.80 | -28.00 | 18.20 | 16.33 | 0.155 | 1.467 | Dynamic |
| moderate_brit.wav | 48000 | 1 | 41.64 | -12.11 | -28.00 | 15.89 | 29.56 | 0.375 | 1.099 | Very dynamic (highest measured spread) |
| moderate_hotrod.wav | 48000 | 1 | 30.28 | -8.00 | -28.62 | 20.62 | 15.89 | 0.162 | 0.328 | Dynamic |
| high_metalcore.wav | 48000 | 1 | 20.41 | -8.38 | -28.00 | 19.62 | 10.67 | 0.244 | 0.735 | More compressed (lowest spread of the guitar clips) |
| high_thrash.wav | 48000 | 1 | 17.75 | -8.18 | -28.00 | 19.82 | 16.56 | 0.017 | 0.080 | Dynamic for a high-gain clip |
| bass_downtown.wav | 48000 | 1 | 23.84 | -9.13 | -28.00 | 18.87 | 14.77 | 0.006 | 0.981 | Moderately dynamic |
| bass_rollin.wav | 48000 | 1 | 28.93 | -8.85 | -28.00 | 19.15 | 8.70 | 0.353 | 0.583 | Most compressed / uniform of the set |

**Observations:**

- All 8 clips are **mono, 48kHz, 16/24-bit PCM** (verify bit depth per-file if it matters for your use —
  `soundfile` reports sample format separately from the values above; not tabulated here since it doesn't
  affect the blend engine, which works in float).
- 7 of 8 clips measure **exactly -28.00 dBFS RMS**; only `moderate_hotrod.wav` differs (-28.62 dBFS). This
  is almost certainly a deliberate normalization applied before these clips were bundled into NAMtoClo —
  useful to know, because it means RMS alone won't discriminate "compressed" vs. "dynamic" content here;
  use the p10–p90 dynamic-range column instead.
- `moderate_brit.wav` has by far the widest measured dynamic range (29.56 dB) of any clip — it is the
  single best candidate for crossover-transition testing in this set.
- `bass_rollin.wav` has the narrowest range (8.70 dB) — most useful as a "does the crossover stay stable
  under consistent level" regression case, not for testing dynamic transitions.
- No file has meaningful silence gaps in the middle (not measured directly, but lead/trail silence is all
  well under 1.5s), so none require trimming before use as-is.

## Classification for hybrid crossover test scenarios

| Filename | Clean → edge of breakup | Clean → crunch | Crunch → high gain | Dynamic touch response | Consistent-level rhythm | Lead |
|---|---|---|---|---|---|---|
| `clean_mayer.wav` | Good | Good | — | Good (wide dynamic range, expressive style) | Fair | Fair |
| `clean_smooth.wav` | Good | Fair | — | Fair | Good (long, smooth, sustained) | Good (sustained notes/chords) |
| `moderate_brit.wav` | Fair (already past clean) | Good | Good | Good (widest measured dynamic range of all 8) | Fair | Fair |
| `moderate_hotrod.wav` | — | Fair | Fair | Fair | Fair | Fair |
| `high_metalcore.wav` | — | — | Fair | Fair | Good (tight, narrower dynamic range = uniform palm-muting) | — |
| `high_thrash.wav` | — | — | Good | Fair | Fair | — |
| `bass_downtown.wav` | Bass only — not applicable to guitar crossover scenarios above | | | | | |
| `bass_rollin.wav` | Bass only — not applicable to guitar crossover scenarios above; best "consistent-level" bass regression case | | | | | |

"Good" / "Fair" / "—" are this project's own judgment calls based on the measured dynamic range and
inferred playing style, not a listening test. A clip can legitimately score "Good" in more than one
scenario (e.g. `moderate_brit.wav` is a strong candidate for both clean→crunch and crunch→high-gain
testing given its exceptionally wide dynamic range).

## Preview DIs vs. NAM training material — do not conflate these

These files are for **auditioning, crossover analysis, level-match testing, and regression testing only**.
They must **not** be used as the actual NAM training/reamping input for generating a synthetic A2 training
target. See the main project README for why: the training pipeline needs a proper, known, calibrated
reamp/DI signal (the same kind NAMtoClo itself uses as `nam_input_wav.wav`), not a genre-styled musical
performance clip.
