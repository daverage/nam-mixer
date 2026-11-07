# HybridN / Dynamic Hybrid: AC30 and Mesa investigation

Date: 7 November 2026. Scope: existing Dynamic Hybrid behavior, the saved 6 November run, and upstream NAM A2. Production code was not changed.

## Verdict

**The approach is valid for an amp that changes character with input level. This particular export does not appear to have lost the Mesa through training. The main demonstrated problem is how easily a player's input level can keep it on the AC30 side, combined with validation that does not test that musical expectation.**

I cannot establish the exact cause of your listening result without the signal level and player settings used when you heard it. I did reproduce a substantial reduction in Mesa selection with a modest input-level change. I did not reproduce an export that stays low gain at every level.

This is an automatic crossfade between two complete amp outputs. It does not interpolate physical gain controls or promise the gain range of either source at arbitrary playing levels. NAM A2 does not change that distinction.

## Evidence and reproducibility

The matching persistent run is `work/a2/AC30-6_TB_Brl_3_DI_and_Mesa_Boogie_Rev_D_Red_Koko_DI_Dynamic_Hybrid/`. Its manifest and session were used as the run record, rather than assuming settings from the current autosave. This report does not claim to have queried a separate SQL database.

The Downloads export's SHA-256 exactly matches the run manifest:

`0302a748bee5627605ac22f46b8c8af4237ec8411c4c52f4b6c06598e609f6a9`

Run source commit: `edfe82ff84cf7ba7af7de25d916c1f8ad5fa995c`. Review baseline: `933c1218760d345a3b4396dedbedbe250852f21b`. Within the eight examined implementation files, only `static/app.js` differs between those commits. Measurements use the stored design and source models with the current native renderer. No assertion is made that the current renderer binary is identical to the run's binary.

The run used:

| Setting | Saved value |
|---|---|
| Quiet branch, Amp A | AC30-6 TB Brl 3 DI |
| Loud branch, Amp B | Mesa Boogie Rev D Red Koko DI |
| Crossover / transition | -27.5 dBFS / 8 dB |
| Automatic Mesa output trim | -4.634 dB |
| Independent amp input trims | Both 0 dB |
| Preview pickup profile | Modern humbucker, +2.5 dB |
| Calibration | Requested Auto; fell back to raw because Mesa metadata is missing |
| Timing correction | Off |
| Cabinet | None |
| Training | Official A2 trainer 0.13.0, 60 epochs, MPS |
| Combined target output gain | +7.108 dB; no further safety reduction |

Measurements and a repeatable script are in [tools/hybridn_measurements.json](tools/hybridn_measurements.json) and [tools/measure_hybridn.py](tools/measure_hybridn.py). Source excerpts are preserved in [tools/hybridn_source_excerpts.txt](tools/hybridn_source_excerpts.txt). Run from repository root with `.venv/bin/python review/tools/measure_hybridn.py`. The script reads the original assets and writes only under `review/tools/`.

## Findings

### 1. Input-level routing explains a credible path to the reported result

The implementation uses a causal input envelope to select the blend weight. For this run, input envelopes at or below **-31.5 dBFS select AC30 only**; at or above **-23.5 dBFS select Mesa only**. The range between is a smooth crossfade. These are envelope levels, not input peaks. A pick transient exceeding the upper threshold does not mean that the smoothed envelope reaches it.

On the complete saved `clean_mayer.wav` preview, adding the selected +2.5 dB profile gives pure Mesa selection for 22.8% of active samples. Without that boost it is only 4.8%. Average Mesa weight changes from 58.9% to 39.1%. These are routing weights, not percentages of audible loudness or distorted energy.

A five-second excerpt around the strongest one-second RMS window demonstrates the sensitivity:

| Gain applied to the raw DI | Mean Mesa weight | Pure Mesa time | Full export ESR against exact teacher |
|---|---:|---:|---:|
| -12 dB | 0% | 0% | 0.0161 |
| -6 dB | 3.1% | 0% | 0.0247 |
| 0 dB | 46.5% | 7.8% | 0.0230 |
| +2.5 dB, saved preview profile | 66.7% | 27.6% | 0.0223 |
| +6 dB | 81.8% | 64.8% | 0.0217 |

Active means envelope above -60 dBFS; a gain change therefore also changes which tails count as active. Export error uses all excerpt samples after a 200 ms warm-up, rather than that activity mask. No input in this sweep exceeded 0 dBFS.

**Practical implication:** the same model can behave mostly like AC30 or mostly like Mesa depending on interface gain, guitar output, volume setting, or host input trim. A crossover selected using one boosted prerecorded DI cannot guarantee that your live guitar reaches the same regions.

### 2. The Full export retains the high-drive behavior on these probes

The saved training-run Full ESR is 0.0338 against its synthetic target. Independently, the Full export follows the reconstructed teacher on the guitar excerpt across six tested input gains with ESR 0.0134 to 0.0247 and level errors within 0.31 dB.

A separate 100 Hz sine sweep also shows the Full export changing character. At -30 dBFS sine peak, where the steady envelope selects AC30, its combined H2-H20 level relative to the fundamental is approximately -33 dB. At -18 dBFS peak, where the envelope selects Mesa, it is approximately -15.5 dB, close to the Mesa source's -15.2 dB. A single-frequency harmonic probe is a diagnostic, not a verdict on guitar tone, but it contradicts a complete disappearance of the high-drive branch.

Lite is less faithful: saved ESR 0.1282; excerpt ESR 0.0660 to 0.1121. Evaluate Full first when judging the design. Full/Lite selection changes the approximation, not the intended clean/high-gain position.

### 3. Calibration and preview level are unresolved across playback environments

AC30 contains `input_level_dbu = 6.5`; Mesa has no input calibration. The saved run explicitly falls back to raw levels for both. There is no evidence that the app secretly attenuated Mesa's input during generation. However, the physical input level represented by Mesa's digital input is unknown.

The +2.5 dB preview profile affects the signal entering both sources and the crossover detector. Training deliberately uses unboosted official excitation. **That omission is not inherently a bug:** it learns a function of actual incoming samples, which can reproduce the preview when the same profiled samples enter the exported model. Playback must nevertheless provide an equivalent input level. The output file has no input-level calibration metadata, so it cannot automatically establish that equivalence for the player.

The -4.634 dB automatic Mesa trim occurs **after** the source amp render. It changes blend balance and loudness, not Mesa's internal distortion. Likewise, the +7.108 dB final gain is after combination. Neither is evidence that the amp was under-driven.

### 4. Current validation proves approximation quality more than useful gain range

The saved validation report correctly says technical success does not establish perceptual equivalence. Its Full and Lite quality checks compare exported audio with the target. Dynamic Hybrid's separate quiet-playing checks are marked `not_applicable`; the special response sweep currently exists for Character Blend.

The official excitation does cover both branches: 48.7% AC30-only, 20.7% Mesa-only, 30.5% transition among active samples. So this is not evidence that training never saw Mesa. But passing a global error metric does not establish that the user's guitar reaches Mesa, that the crossfade occurs at useful playing strengths, or that its sustain sounds right.

### 5. The core DSP choices are reasonable; changing architecture is not the first fix

The blend is `A(x) * (1 - t(x)) + B(x) * t(x)`, with a smoothstep weight and a bounded, causal envelope. Source rendering, frozen output trim, and frozen timing choice are reused for training. Keeping the envelope finite allows the destination network's temporal support to cover it. This run records 6,332 samples of source and destination history and 3,840 for the envelope. Temporal fit is necessary, not proof of enough model capacity; the measured Full error is the stronger evidence here.

A linear amplitude crossfade is defensible for correlated amp outputs. Different phase responses can still cause cancellation in the transition, but this review has not established harmful cancellation as the cause. The stored timing diagnosis is ambiguous; applying an 11-sample correction just because several windows suggest it would be unsupported without verification. Do not replace the crossfade law or force alignment on this evidence.

Implementation evidence: `hybrid/modes/blend.py:42-77` (routing and blend); `hybrid/core/pipeline.py:115-204,216-277` (preview input and blend); `hybrid/core/envelope.py:168-226` (detector); `hybrid/modes/training_target.py:654-745` (teacher construction); `hybrid/core/render.py:110-185` (native rendering); `scripts/train_a2.py:609-655` and `hybrid/training/validation_report.py:48-131` (export checks); `static/app.js:1005-1073` (existing performance-relative crossover guide).

## What NAM 2 / A2 changes

The application is already using the official A2 generation of snapshot modeling, including a packed Full/Lite export. A2 improves the accuracy/compute tradeoff; it does not automatically interpret the source amps as endpoints of a gain knob. See the [official A2 release](https://www.neuralampmodeler.com/post/a2-is-released) and [architecture rationale](https://www.neuralampmodeler.com/post/architecture-a2).

The official `Sequential` architecture feeds one child model into the next. Replacing this blend with two Sequential amp children would create `Mesa(AC30(x))`, a different gain-staging system. Sequential is useful for an amp followed by a cabinet, not as a replacement for this automatic parallel crossfade. [NAMCore implementation](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/main/NAM/sequential.cpp#L119).

Packed/slimmable submodels provide different computational sizes against the same target. Their selection is not a tone control. [Official packed training documentation](https://github.com/sdatkinson/neural-amp-modeler/blob/main/docs/source/tutorials/packed-training.rst).

A genuine independently controlled gain/morph parameter requires an appropriate training and player contract. The separate parametric NAM fork supports knob-conditioned models, but adopting it brings a compatible-player requirement. It is not a quick fix for this export. [Parametric implementation](https://github.com/DLC86/neural-amp-modeler-parametric). More primary-source detail is in [NAM2_PRIMARY_RESEARCH.md](NAM2_PRIMARY_RESEARCH.md).

## Improvements worth making, in order

1. **Make source, exact teacher, and exported Full directly comparable before accepting a model.** Use the same guitar passage and input levels, with solo A, solo B, teacher, and Full/Lite exports. Apply output-only loudness matching for listening and retain raw levels for measurements. Preserve the frozen design and target output scalar when rendering the teacher. Success: a user can identify whether an unwanted clean sound already exists before training or appears only after export. This builds on the existing source preview and native renderer.

2. **Set the crossover using the guitar and input level that will actually be used.** Add a short recorded-DI calibration step or reuse a user-recorded DI; expose the actual lower/upper envelope thresholds alongside existing coverage tools. Show how much of normal, soft, and hard playing selects each branch. Offer an explicit input-level offset for auditioning the downloaded model, and save the preview profile/level guidance in its companion session. Success: normal playing reaches the intended side and a +/-6 dB input test makes the dependency visible. Do not infer an exact physical calibration value for Mesa from its name or genre.

3. **Add a Dynamic Hybrid response acceptance check.** Render a held-out guitar DI at several safe levels through the exact teacher and both export sizes. Report error separately for A-only, transition, and B-only regions, plus branch coverage and output level difference. Include sustained notes and attacks for listening. A technical failure means disagreement with the teacher; an expectation warning means the chosen DI never reaches the intended branch. Success: an export can no longer appear musically validated solely because global ESR passes. These measurements show why both checks matter.

4. **Confirm each source's drive at its selected operating level before freezing the design.** Existing independent input trims are the appropriate tool if a source is actually too clean when soloed; output trim only balances loudness. Keep an explicit warning for missing calibration and offer measured/manual calibration rather than a guessed fix. Success: the Mesa solo has the intended drive at the levels where it will be selected, and changing input drive is distinguished from making Mesa louder.

The immediate experiment for this pair is to audition Full with the same input level as the saved +2.5 dB preview, then sweep input level and crossover separately. If the exact teacher still sounds too clean at the desired playing strength, lower the crossover using that performance. If solo Mesa is too clean, adjust its input trim before rebuilding. If the teacher sounds right but Full does not, investigate export fidelity. More epochs are justified only by that last comparison.

## Limits

No live guitar recording, player configuration, or listening-level measurement from the reported playback was available. The audio comparisons here are numerical renders, not a claim that I listened to them. The guitar probe is the original design DI, not an independent musical hold-out, so it cannot establish broad generalization. The sine probe supplies an independent stimulus but limited musical relevance. These limitations are precisely why the proposed acceptance check includes another guitar DI. No production changes or retraining were performed.
