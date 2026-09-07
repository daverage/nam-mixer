> Review the current `daverage/hybrid-nam-builder` `master` branch and fix the **Character Blend low-level response bug** before any further A2 training.
>
> This task is specifically about the NAM blend tool, not NamToClo.
>
> We have now reproduced a serious issue in a trained Character Blend A2:
>
> ```text
> At normal input levels the NAM works.
> As input is reduced, the output eventually collapses to digital silence.
> In a NAM2 VST this happens at roughly -9 dB relative to the test guitar signal.
> ```
>
> This is not acceptable amplifier behaviour and it also makes the resulting NAM unsuitable for later GP-5/GP-50 conversion.
>
> The current Character Blend implementation appears to contain the root cause in `hybrid/character_blend.py`.
>
> The default analysis grid is:
>
> ```text
> -24, -18, -12, -6, 0, +6 dB
> ```
>
> and the current filter interpolation uses triangular weights:
>
> ```python
> weights = np.vstack([
>     np.maximum(
>         0.0,
>         1.0 - np.abs(envelope - level)
>         / max(1.0, np.diff(levels).mean())
>     )
>     for level in levels
> ])
>
> weights /= np.maximum(weights.sum(axis=0), _EPS)
> output = np.sum(np.vstack(filtered) * weights, axis=0)
> ```
>
> With 6 dB spacing, once the dry-input envelope falls sufficiently below the lowest analysed point, all weights become zero.
>
> The `_EPS` denominator prevents NaN, but it does not create a valid weight.
>
> The result is:
>
> ```text
> all weights = 0
> → output = 0
> ```
>
> So Character Blend is effectively introducing a hard low-level gate into the teacher target.
>
> A2 then learns that behaviour.
>
> ---
>
> # Primary objective
>
> Character Blend must behave like a real amplifier across low input levels.
>
> It must not become digitally silent merely because the input envelope falls below the measured analysis grid.
>
> The correct extrapolation rule is:
>
> ```text
> below lowest measured level:
> hold the lowest measured character
>
> above highest measured level:
> hold the highest measured character
>
> between measured levels:
> interpolate smoothly between the two neighbouring levels
> ```
>
> Do not solve this by:
>
> ```text
> adding noise
> adding a signal floor
> adding compression
> hard limiting
> artificially boosting low-level output
> ```
>
> Fix the interpolation itself.
>
> ---
>
> # Phase 1 — replace the current triangular weighting
>
> Replace the existing all-level triangular basis interpolation with an explicit adjacent-level interpolation.
>
> For every envelope sample:
>
> ```text
> if envelope <= levels[0]:
>     weight[0] = 1
>
> elif envelope >= levels[-1]:
>     weight[-1] = 1
>
> else:
>     find adjacent levels i and i+1
>     interpolate only between those two
> ```
>
> Example:
>
> ```text
> analysis levels:
> -24  -18  -12  -6  0  +6
>
> envelope = -35 dB:
> 100% -24 dB state
>
> envelope = -21 dB:
> 50% -24 dB state
> 50% -18 dB state
>
> envelope = -15 dB:
> 50% -18 dB state
> 50% -12 dB state
>
> envelope = +12 dB:
> 100% +6 dB state
> ```
>
> For every finite envelope sample:
>
> ```python
> weights.sum(axis=0) == 1.0
> ```
>
> within floating-point tolerance.
>
> Add an explicit assertion or debug invariant for this.
>
> ---
>
> # Phase 2 — test well outside the analysis range
>
> Add unit tests for envelope values including:
>
> ```text
> -120
> -80
> -60
> -48
> -36
> -30
> -24
> -21
> -18
> -15
> -12
> -9
> -6
> -3
> 0
> +3
> +6
> +12
> +24 dB
> ```
>
> Verify:
>
> ```text
> no NaN
> no Inf
> no all-zero weights
> weights always sum to 1
> endpoints clamp correctly
> interpolation is continuous
> ```
>
> ---
>
> # Phase 3 — add a low-level response regression test
>
> Create a synthetic test where both source amps are guaranteed to produce non-zero output for any non-zero input.
>
> A simple linear synthetic pair is enough.
>
> Sweep the input gain over:
>
> ```text
> +6 dB
> 0 dB
> -6 dB
> -12 dB
> -18 dB
> -24 dB
> -30 dB
> -36 dB
> -42 dB
> -48 dB
> ```
>
> Build Character Blend at each level.
>
> Verify:
>
> ```text
> output remains non-zero
> output RMS falls smoothly
> no abrupt cliff to digital silence
> ```
>
> Do not require perfect 1:1 linear scaling because real amp/compression behaviour can differ.
>
> The important invariant is:
>
> ```text
> a moderate input reduction must not produce a near-infinite output reduction
> ```
>
> ---
>
> # Phase 4 — add a real teacher-response sanity check
>
> Before generating a Character Blend A2 bundle, evaluate the finished teacher using a fixed reference DI at several gains:
>
> ```text
> 0 dB
> -6 dB
> -12 dB
> -18 dB
> -24 dB
> -30 dB
> -36 dB
> ```
>
> Record:
>
> ```text
> input RMS dBFS
> output RMS dBFS
> output peak dBFS
> relative gain
> output delta from previous test level
> ```
>
> Example:
>
> ```text
> Input    Output     Step response
>
>   0      -12.1
>  -6      -17.4       -5.3
> -12      -22.8       -5.4
> -18      -28.1       -5.3
> -24      -33.8       -5.7
> -30      -39.7       -5.9
> ```
>
> That is healthy.
>
> Something like:
>
> ```text
> -18      -28
> -24      -160
> ```
>
> must be treated as a failure.
>
> ---
>
> # Phase 5 — add a Character Blend export gate
>
> Do not allow training-bundle generation to silently proceed when the teacher has a hard low-level collapse.
>
> Add a structured compatibility/sanity result, for example:
>
> ```python
> @dataclass
> class LowLevelResponseCheck:
>     ok: bool
>     levels_db: list[float]
>     output_rms_dbfs: list[float]
>     max_step_error_db: float
>     dead_zone_detected: bool
>     warning: str | None
> ```
>
> Suggested policy:
>
> ```text
> -24 dB must remain audibly/non-trivially responsive
> -30 and -36 dB are diagnostic margin checks
> ```
>
> Do not use one arbitrary absolute RMS floor alone.
>
> Detect abrupt collapse relative to neighbouring levels.
>
> For example:
>
> ```text
> if input changes by -6 dB
> and output changes by -50 dB:
>     fail
> ```
>
> Exact thresholds should be derived conservatively from real known-good NAM behaviour.
>
> ---
>
> # Phase 6 — show the result in the UI
>
> In the Character Blend design or Create A2 section, show:
>
> ```text
> LOW-LEVEL RESPONSE
>
>  0 dB   -12.1 dBFS
> -6 dB   -17.4 dBFS
> -12 dB  -22.8 dBFS
> -18 dB  -28.1 dBFS
> -24 dB  -33.8 dBFS
> -30 dB  -39.7 dBFS
>
> ✓ continuous low-level response
> ✓ no dead zone
> ```
>
> Or:
>
> ```text
> ✗ low-level collapse detected
>
> -18 dB input → -28 dBFS output
> -24 dB input → silence
>
> Do not train this design.
> ```
>
> Make this visible before the user spends time on Kaggle training.
>
> ---
>
> # Phase 7 — ensure preview and training use identical logic
>
> The same `build_character_blend()` implementation must continue to power:
>
> ```text
> browser preview
> training target generation
> diagnostics
> ```
>
> Do not create a separate "fixed" path only for training.
>
> If the preview sounds healthy at low level, the generated teacher must use the exact same interpolation.
>
> ---
>
> # Phase 8 — inspect the Drive control separately
>
> While reviewing `character_blend.py`, also document the current Drive behaviour.
>
> It currently uses:
>
> ```python
> donor = np.where(drive_b >= 0.5, b, a)
> ```
>
> Therefore a constant Drive value behaves approximately like:
>
> ```text
> 0%–49.999% → Amp A donor
> 50%–100%   → Amp B donor
> ```
>
> So `Drive = 50%` is not a literal 50/50 nonlinear blend.
>
> At exactly 50%, Amp B is the donor.
>
> Do not silently redesign this as part of the low-level bug fix.
>
> Instead:
>
> 1. add tests documenting current behaviour;
> 2. make the UI/help text accurately describe it;
> 3. propose a separate follow-up design for a genuinely continuous nonlinear morph if practical.
>
> The low-level interpolation fix must remain isolated and testable.
>
> ---
>
> # Phase 9 — validate using the actual failing design
>
> Reproduce the design from the current real test:
>
> ```text
> Amp A:
> Fender Deluxe Reverb Head
>
> Amp B:
> SLASH AFD#2 Head
>
> Character Blend
>
> Tone:
> ~20% B
>
> Feel:
> ~20% B
>
> Drive:
> 50% B
>
> Cab:
> off
>
> High Def:
> 120 epochs
> ```
>
> First test the **teacher**, before retraining.
>
> Play/render the same DI through the corrected Character Blend at:
>
> ```text
> 0
> -6
> -12
> -18
> -24
> -30
> -36 dB
> ```
>
> Confirm the teacher remains responsive.
>
> Only once that passes, regenerate the training bundle and train a new Full A2 at 120 epochs.
>
> ---
>
> # Phase 10 — validate the trained NAM
>
> After training, run the exact same gain sweep through the exported Full A2:
>
> ```text
> 0
> -6
> -12
> -18
> -24
> -30
> -36 dB
> ```
>
> Compare:
>
> ```text
> teacher output RMS
> trained Full A2 output RMS
> ```
>
> Report the difference at each level.
>
> The trained A2 must not develop a new hard gate.
>
> Then manually verify in NAM2 VST:
>
> ```text
> gradually reduce plugin input
> ```
>
> and confirm audio fades/cleans naturally instead of suddenly disappearing.
>
> ---
>
> # Phase 11 — add permanent post-training validation
>
> Add this low-level sweep to the existing A2 validation system.
>
> A trained Character Blend should only be marked successful if:
>
> ```text
> Full A2 tracks the teacher across the low-level sweep
> no dead zone exists
> no infinite/undefined dynamics metric occurs
> ```
>
> Record the results in `training_manifest.json`.
>
> Suggested manifest section:
>
> ```json
> {
>   "low_level_response": {
>     "levels_db": [0, -6, -12, -18, -24, -30, -36],
>     "teacher_output_rms_dbfs": [],
>     "full_output_rms_dbfs": [],
>     "full_error_db": [],
>     "dead_zone_detected": false,
>     "pass": true
>   }
> }
> ```
>
> ---
>
> # Important constraints
>
> Do not:
>
> ```text
> modify Dynamic Hybrid behaviour - unless this may fix similar issues there
> modify Parallel Blend behaviour - unless this may fix similar issues there> add a noise floor
> add a gate workaround
> normalize every output level independently
> boost low-level output artificially
> change A2 architecture
> change training loss yet
> change NamToClo
> ```
>
> This task is specifically to fix the Character Blend teacher's interpolation and ensure the resulting NAM remains a valid amplifier across low input levels.
>
> ---
>
> # Desired end state
>
> Character Blend should behave like:
>
> ```text
> hard playing
>     → intended high-level character
>
> medium playing
>     → intended interpolated character
>
> soft playing
>     → intended lowest-level amp character
>
> very soft playing
>     → progressively quieter amp output
> ```
>
> It must **never** behave like:
>
> ```text
> soft enough
>     → zero
> ```
>
> simply because the input moved outside the analysis grid.
>
> The immediate success criterion is:
>
> ```text
> corrected Character Blend teacher
> → responsive through at least -24 dB input
> → preferably smooth through -36 dB diagnostic range
> → retrained Full A2 follows the same behaviour
> ```
>
> Treat this as a release-blocking Character Blend correctness bug, not a tuning improvement.
