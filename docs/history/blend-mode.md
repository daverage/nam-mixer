Work on the private repository:

daverage/hybrid-nam-builder

GOAL
----

Extend Hybrid NAM Builder into two clearly separated design modes:

1. DYNAMIC HYBRID
   The existing behaviour:
   quiet input -> Amp A
   transition according to input envelope
   loud input -> Amp B

2. FIXED BLEND
   A new behaviour:
   Amp A and Amp B are rendered in parallel and combined at a fixed user-controlled mix,
   independent of playing level.

Also add a SHARED CABINET IR stage which can be used with BOTH modes:

- preview through an optional cab IR
- optionally bake that cab IR into the generated A2 target
- preview-only cab use must NOT alter the trained A2
- baked cab use must alter the official training target and therefore become part of the A2

This work is generic Hybrid NAM Builder functionality.

DO NOT add GP50, SnapTone, CLO, NamToClo or target-device-specific logic to this repository.


BEFORE CODING
-------------

1. Pull/re-read current master and inspect the actual current code. Do not assume the repository
   is still at the commit described in this prompt.

2. In particular understand the existing responsibilities of:

   app.py
   templates/index.html
   static/app.js
   static/style.css

   hybrid/pipeline.py
   hybrid/blend.py
   hybrid/level_match.py
   hybrid/design.py
   hybrid/training_target.py
   hybrid/receptive_field.py
   hybrid/safety.py
   hybrid/render.py
   hybrid/calibration.py

   scripts/train_a2.py
   cloud/kaggle/train_a2_cloud.py
   hybrid/kaggle_training.py

3. Run the existing tests before making changes.

4. Preserve the currently proven Dynamic Hybrid behaviour exactly when:
   - mode = hybrid
   - no cab IR is selected

Do not make a broad rewrite of the existing working pipeline merely to accommodate Blend mode.


HIGH-LEVEL UX
=============

Turn the design UI into two modes using tabs near the top of the interface:

    [ Dynamic Hybrid ] [ Fixed Blend ]

Dynamic Hybrid should be the default tab so existing behaviour remains the default experience.

The tabs are DESIGN MODES, not completely separate applications.

The following should remain shared between both tabs:

- Amp A
- Amp B
- preview DI
- instrument/input profile
- NAM input calibration
- Render Amps
- test gain
- Amp A / Result / Amp B listening controls
- Cabinet IR
- official NAM training input
- A2 training quality
- local/Kaggle training
- output/download

Switching between Hybrid and Blend MUST NOT re-run NAM inference.

The already-rendered Amp A and Amp B pair should be reusable by either design mode.


TAB 1 — DYNAMIC HYBRID
======================

This is the existing product.

Keep the existing controls and behaviour:

- crossover point
- transition width
- auto level match near crossover
- manual Amp B trim
- journey between amps
- crossover coverage
- existing smoothstep/linear amplitude crossfade
- existing frozen effective B trim
- existing training-target behaviour

Do not change the maths.

Current conceptual behaviour remains:

    t = level-dependent smoothstep

    result = A * (1 - t) + B * t

The existing Hybrid tab without a cab must remain regression-compatible with current output.


TAB 2 — FIXED BLEND
===================

Add a new Fixed Blend mode.

The defining rule is:

    mix_b = 0.0 -> 100% Amp A
    mix_b = 1.0 -> 100% Amp B

    result = A * (1 - mix_b) + B * mix_b

Use LINEAR amplitude blending, not equal-power blending, for the same reason the current
Hybrid implementation uses linear amplitude blending: the two signals are correlated
renders of the same guitar input.

There is NO crossover envelope in Fixed Blend mode.

The mix does not change according to input level.

UI
--

Add a clear mix slider:

    Amp A        50 / 50        Amp B
    [----------------●-------------]

Range:
    0 to 100
Default:
    50
Step:
    1

Show both percentages live, e.g.

    Amp A 35% / Amp B 65%

Moving the mix slider must be cheap and immediate. It must only recombine the cached Amp A/B
renders and must NOT run either NAM again.

The centre preview button should dynamically say:

    Hybrid

on the Dynamic Hybrid tab, and:

    Blend

on the Fixed Blend tab.

Amp A and Amp B preview buttons remain available in both modes.


BLEND LEVEL MATCHING
--------------------

The existing Hybrid auto-level match is specifically based on the crossover region and
therefore must NOT simply be reused unchanged for Fixed Blend.

Add a fixed-blend-specific automatic B-to-A level match.

Use active playing material from the current rendered preview DI rather than a crossover band.

A reasonable implementation is:

- derive an active mask from the already available profiled input envelope
- exclude silence, using a documented threshold consistent with the project’s existing
  active-playing conventions
- calculate RMS of rendered A and B over the same active samples
- suggested B trim = A RMS dB - B RMS dB
- manual B trim remains an adjustment on top
- freeze the effective B trim when the design is generated for training

Do not recalculate this auto trim against the official NAM training excitation.

The same important rule that exists for Dynamic Hybrid applies:

    audition -> freeze the effective trim -> reuse that exact trim for target generation

The Blend UI can therefore retain a Level Match card, but its text should change according
to mode:

Hybrid:
    Auto level match Amp B to Amp A near the crossover

Blend:
    Auto level match Amp B to Amp A over active playing

Manual Amp B trim remains available in both modes.


CODE STRUCTURE FOR FIXED BLEND
==============================

Do not confuse the new user-facing "Fixed Blend" mode with the existing hybrid/blend.py,
which implements the level-driven crossfade.

Prefer a clearly named implementation such as:

    hybrid/fixed_blend.py

and/or:

    build_fixed_blend(...)

rather than overloading ambiguous functions.

Create an appropriate immutable design snapshot for Fixed Blend, analogous to HybridDesign.

For example:

    BlendDesign

It should freeze at least:

- amp_a_path
- amp_b_path
- mix_b
- auto_trim_db
- manual_b_trim_db
- effective_b_trim_db
- alignment state/offset
- instrument/input-profile provenance
- calibration provenance
- design preview DI provenance

Test gain remains audition-only and MUST NOT be frozen/applied to official training input.


FIXED BLEND TRAINING TARGET
===========================

Fixed Blend must be trainable into a normal A2 using the same training infrastructure.

For the official V3 training input:

1. validate official V3 exactly as today
2. do NOT apply the audition input profile
3. do NOT apply test gain
4. apply the same per-model NAM calibration rules as today
5. render official input through Amp A
6. render official input through Amp B
7. apply the frozen alignment behaviour
8. apply the frozen effective B trim
9. combine using the fixed mix:

       target = A * (1 - mix_b) + B * mix_b

10. optionally apply baked cabinet IR, if enabled
11. perform the existing safety/peak-ceiling behaviour
12. train A2 using the existing local/Kaggle infrastructure

Do not create an entirely separate training backend.

Preserve the existing bundle filenames required by the local/Kaggle pipeline where practical,
including input.wav, hybrid_target.wav and training_manifest.json, even if
"hybrid_target.wav" is now a legacy/internal filename for Blend mode.

Add an explicit mode field to provenance, e.g.

    design.mode = "hybrid"

or:

    design.mode = "blend"

so there is never any ambiguity about how a target was created.


SHARED CABINET IR STAGE
=======================

Add a shared Cabinet section visible in BOTH modes.

Conceptually the signal chain is:

Dynamic Hybrid:

    Amp A ----\
               dynamic hybrid ----> optional CAB ----> preview/output
    Amp B ----/

Fixed Blend:

    Amp A ----\
               fixed blend -------> optional CAB ----> preview/output
    Amp B ----/

The cabinet is AFTER the amp combination.

Do not apply different cabinets to A and B in this phase.

One shared cabinet only.


CAB UI
------

Add a Cabinet IR card/section with:

- WAV upload/select
- parsed IR information
- checkbox/toggle:

      Use cab in preview

- checkbox/toggle:

      Bake cab into A2

Behaviour:

No IR selected:
    both controls disabled/off

Use cab in preview:
    listening previews are passed through the cab

Bake cab into A2:
    the generated official training target is passed through the same prepared IR
    before safety processing and training

If "Bake cab into A2" is enabled, automatically ensure "Use cab in preview" is also enabled.

Do not allow the user to audition one signal chain and unknowingly bake a different one.

If Bake is later disabled, Preview may remain enabled.

Make the status very clear:

    Cab: preview only — exported A2 remains amp/head only

or:

    Cab: baked — exported A2 will include this cabinet

or:

    Cab: off


CAB PREVIEW SEMANTICS
---------------------

When "Use cab in preview" is enabled, apply the SAME cab to:

- Amp A preview
- Hybrid/Blend result preview
- Amp B preview

This makes A / Result / B comparisons fair.

Cab processing must happen BEFORE preview_safety_limiter.

The preview safety limiter stays preview-only.


CAB IR PROCESSING
=================

Create a reusable module, e.g.

    hybrid/cab_ir.py

The exact same core function must be used for:

- preview cab processing
- baked training-target cab processing

Do not implement two subtly different cab paths.

Requirements:

- WAV input
- float processing
- mono output
- if input IR is stereo/multichannel, deterministically downmix to mono
- reject NaN/Inf
- reject silent/empty IR
- resample the IR to the current audio sample rate when required
- scipy is already a runtime dependency, so scipy.signal/resample_poly/fftconvolve are available
- use causal FIR convolution
- return exactly the same number of samples as the source audio for streaming-equivalent behaviour:
      full causal convolution, then retain the first N output samples
- do not circular-convolve
- do not silently change the source audio sample rate
- do not apply the preview limiter inside the cab helper

Handle leading IR silence deterministically.

Because A2 training currently assumes synthetic latency 0, trim only meaningless leading IR
silence/onset delay in a documented deterministic way before convolution.

Do NOT:
- peak-align the IR to an arbitrary later peak
- minimum-phase-transform it
- alter its frequency response intentionally

Record any leading samples removed in metadata.

Do not silently introduce a new arbitrary cab EQ or tone-matching stage.

Keep this an ordinary FIR cab convolution.


CAB GAIN
--------

Do not add a new Cab Gain control in this phase.

Do not silently apply a different gain rule between Preview and Bake.

Whichever IR amplitude convention is chosen, it MUST be deterministic, documented, recorded
where appropriate, and identical in preview and training-target generation.

Avoid gratuitous normalization.

The existing preview safety limiter and training-target peak ceiling remain responsible for
their respective safety jobs.


CAB UPLOAD / STORAGE
====================

Add a safe upload endpoint similar in spirit to the existing NAM upload handling.

Store working copies under work/, not in the repository.

Validate file type/content.

Return useful metadata such as:

- filename
- path/token used internally
- original sample rate
- original channel count
- frame count
- duration
- SHA256
- prepared/trimmed information where available

Do not trust arbitrary client filesystem paths.


RECEPTIVE FIELD — IMPORTANT
============================

Do NOT ignore the existing receptive-field safety model.

A cab FIR applied AFTER the amp/hybrid/blend is a SERIAL temporal dependency.

Without baked cab:

Dynamic Hybrid required history is approximately:

    max(
        Amp A receptive field,
        Amp B receptive field,
        crossover-envelope history
    )

Fixed Blend required history is:

    max(
        Amp A receptive field,
        Amp B receptive field
    )

With a baked prepared FIR of L samples:

    required_history =
        base_required_history + (L - 1)

because the FIR is serial after the combined amp output.

Preview-only cab IR does NOT affect model receptive-field requirements.

Update BOTH local and Kaggle/cloud receptive-field checks so this is represented correctly
from the manifest.

If baked-cab target history exceeds the available A2 receptive field:

- DO NOT silently ignore it
- DO NOT pretend the target is exactly representable
- DO NOT truncate the IR behind the user's back merely to make the check pass
- abort generation/training with a clear actionable message

The message should explain that:

    the cab works for preview,
    but baking this particular IR after these source NAMs exceeds the A2 temporal history.

If useful, report:

- source/base required samples/ms
- prepared cab FIR samples/ms
- total required samples/ms
- A2 available samples/ms

This is particularly important because some current A2 source models already use essentially
the whole A2 receptive field.

Cab preview must remain available even if baking is formally impossible.


MANIFEST / PROVENANCE
=====================

Extend provenance cleanly.

For every generated target record:

    mode: hybrid | blend

For Blend record:

    mix_b
    mix_a
    auto trim
    manual trim
    frozen effective B trim

For Cab record at least:

    selected
    preview_enabled (if useful as audition provenance)
    baked
    original filename
    SHA256
    original sample rate
    prepared sample rate
    original channels
    original frame count
    prepared frame count
    leading samples trimmed
    FIR history samples

Do not put the raw IR itself into JSON.

For Dynamic Hybrid with no cab, existing manifest semantics must remain compatible.


METADATA / OUTPUT NAM
=====================

Do not invent unsupported NAM metadata enum values.

Inspect the installed/current official NAM metadata API.

If it has an official appropriate amp+cab/rig gear type, it may be used when a cab is baked.

If not, keep the existing supported AMP metadata and record the cab accurately in our own
manifest rather than inventing an enum.

Blend NAM names should make their origin understandable, for example:

    Blend FenderSuperReverb + JCM800 35-65

Dynamic Hybrid naming can remain as today.

Do not make filenames absurdly long.


UI DETAILS
==========

Update the page tagline to communicate both modes, e.g.

    Create dynamic input-driven hybrids or fixed blends from two NAM captures.

Dynamic Hybrid tab:
- existing controls
- journey visualization
- crossover coverage

Fixed Blend tab:
- fixed mix control
- no crossover control
- no transition width
- no journey/crossover coverage because those concepts do not apply

Optionally show a simple static A/B composition indicator for Blend, but keep this light.

Shared Cabinet section should be logically near Listen/output rather than buried in advanced
settings.

Shared Create A2 section should dynamically say what is being created:

    Create Hybrid A2

or:

    Create Blend A2

The explanatory text underneath should also change.

Keep the existing visual language/style rather than redesigning the whole application.


API
===

Make the API mode-aware while retaining sensible backwards compatibility.

For example:

POST /api/preview

    source = a | b | hybrid | blend

and the appropriate mode parameters.

POST /api/generate

    mode = hybrid | blend

Default mode to "hybrid" if omitted so old callers/tests continue to work where possible.

The current /api/blend_info endpoint is unfortunately named now that "Blend" becomes a real
design mode.

Do not let that naming confusion spread.

You may:
- retain it as a backwards-compatible Hybrid-only alias
- introduce a more generic/new endpoint for current UI use

but do not break existing callers unnecessarily.

Keep expensive NAM rendering isolated in /api/render_pair.

Hybrid crossover changes, Blend mix changes and Cab preview toggles must NOT trigger NAM
inference again.


CAB CACHE / PERFORMANCE
=======================

Cab convolution on a preview clip should be fast enough for interactive audition.

Do not unnecessarily re-read/resample the IR on every slider movement.

Cache prepared IRs keyed by at least:

- IR identity/hash
- target sample rate

It is fine to recompute the cheap fixed A/B blend on every mix slider movement.

Avoid unbounded caches.


TRAINING / KAGGLE
=================

The existing Kaggle workflow must continue to work for both design modes.

Do not create a second cloud system.

The generated manifest must contain everything the cloud worker needs to:

- understand mode
- validate RF correctly
- train against the already-generated target

Do NOT upload source NAMs or cab IRs to Kaggle if the existing cloud design does not require
them.

The target WAV already contains the rendered/baked result.

Only provenance needs the IR hash/metadata.

Preserve current privacy boundaries.


TESTS
=====

Add focused tests covering at least:

FIXED BLEND
- mix 0.0 returns Amp A exactly
- mix 1.0 returns trimmed Amp B exactly
- mix 0.5 gives expected linear blend
- mix is independent of input envelope
- manual B trim works
- Blend auto level match is based on active playing and does not use crossover logic
- frozen effective trim is reused for official target generation

HYBRID REGRESSION
- existing no-cab Hybrid tests still pass
- current Hybrid output is unchanged for equivalent inputs/settings
- current training bundle behaviour remains unchanged when no cab is selected

CAB
- identity/one-tap IR leaves audio unchanged
- known short FIR produces mathematically expected causal convolution
- output length equals input length
- stereo IR downmix is deterministic
- sample-rate conversion works
- invalid/empty/silent/non-finite IR is rejected
- preview and training use the same prepared IR path
- preview-only cab does NOT alter generated training target
- baked cab DOES alter target as expected
- cab is applied after Hybrid/Blend, not independently before the two branches
- manifest hashes/metadata are correct

RECEPTIVE FIELD
- Blend no-cab dependency is max(A, B)
- Hybrid no-cab remains max(A, B, envelope)
- baked cab adds L-1 serial samples
- preview-only cab adds zero training dependency
- over-capacity baked cab fails clearly
- exact-fit boundary is still permitted under the existing policy

API/UI BACKEND
- omitted mode defaults to Hybrid
- Blend preview accepts mix
- invalid mix is rejected/clamped according to documented API behaviour
- cab upload validation works
- both modes can generate valid bundles
- existing training endpoints still find/use those bundles

Run the ENTIRE pytest suite after the focused tests.


DOCUMENTATION
=============

Update:

README.md
CLAUDE.md
and any relevant phase/design docs

Document the conceptual distinction clearly:

Dynamic Hybrid:
    changes from Amp A toward Amp B according to playing level.

Fixed Blend:
    always combines the two amp responses at the chosen fixed ratio.

Cabinet:
    optional shared post-amp FIR;
    can be preview-only or baked into the trained A2.

Explicitly document that cab baking can increase temporal dependency and may be refused when
it exceeds the destination A2 receptive field.

Do not mention GP50 as the reason for Fixed Blend mode.


ACCEPTANCE CRITERIA
===================

The work is complete when:

1. The app has working Dynamic Hybrid and Fixed Blend tabs.
2. Existing Hybrid behaviour remains intact.
3. Amp A/B are rendered once and shared across both modes.
4. Blend has a live fixed A/B mix slider.
5. Blend preview is cheap and does not re-run NAM inference.
6. Blend can generate and train a real A2 target.
7. Both modes can load one shared cab IR.
8. Cab can be enabled for preview.
9. Cab can optionally be baked into the A2 target.
10. Preview and bake use exactly the same core IR processing.
11. Preview-only cab does not alter the trained target.
12. Baked cab provenance is recorded.
13. RF accounting correctly treats a baked cab as a serial FIR dependency.
14. No GP50/SnapTone/CLO-specific behaviour has been added.
15. Local and Kaggle A2 training workflows remain functional.
16. Full automated test suite passes.


SCOPE CONTROL
=============

Do NOT in this task:

- add A1 conversion
- add CLO conversion
- add GP50 logic
- add SnapTone optimisation
- add multi-cab blending
- add separate cab A / cab B
- add EQ
- add tone matching
- add saturation controls
- add three-or-more-amp blending
- redesign the application from scratch
- change the proven Hybrid crossover algorithm
- start another phase automatically after this one


FINISH
======

When implementation is complete:

1. Run the full test suite.
2. Report:
   - tests run / pass count
   - important files changed
   - any RF/cab-baking limitation discovered during implementation
   - whether existing Hybrid regression behaviour is preserved
3. Commit the completed work with a clear commit message.
4. Push the commit to the repository's existing master branch.
5. Stop there. Do not begin any follow-on feature work automatically.


ADDENDUM -- CABINET RF POLICY CORRECTED TO ADVISORY/APPROXIMATION
===================================================================

The original "RECEPTIVE FIELD -- IMPORTANT" section above described a
formal `base + (cab FIR length - 1)` total and said generation/training
"abort[s]" if that total exceeds the destination A2's receptive field.

That turned out to be too strict in practice: it made cabinet baking
impossible whenever the source NAMs themselves already use most/all of the
A2's receptive field (a common case -- both a Hybrid/Blend pair's Amp A and
Amp B RF can each already equal the full ~132 ms A2 receptive field on
their own), even though A2 is being TRAINED to approximate the rendered
teacher target, not to compile its signal graph exactly. A real Kaggle
training run was refused for exactly this reason.

The corrected, now-implemented policy distinguishes two tiers:

1. **CORE dependency (Amp A/Amp B RF, + the crossover envelope for
   Hybrid) is still a HARD requirement.** If this alone exceeds the
   destination A2's receptive field, generation/training is still refused
   exactly as originally specified -- this has NOT changed.
2. **A baked cabinet's formal serial FIR history (`fir_length - 1`) is
   calculated and reported honestly, but is now ADVISORY ONLY.** If
   `core + cab_history` exceeds the A2's receptive field, training
   CONTINUES: the A2 is understood to be learning an APPROXIMATION of the
   post-cab response within its available temporal capacity. Both
   `scripts/train_a2.py` and `cloud/kaggle/train_a2_cloud.py` print an
   explicit "CABINET APPROXIMATION" notice explaining this rather than
   refusing to train, and the manifest's `receptive_field` record
   distinguishes `hard_required_samples` (the gate) from
   `formal_total_required_samples`/`cab_requires_approximation` (advisory).

Nothing about the actual signal chain changed: the full prepared cabinet IR
is still convolved in its entirety for both preview and a baked training
target (see `hybrid/cab_ir.py`) -- only the POLICY deciding whether A2 is
allowed to attempt learning a cab that formally exceeds its receptive field
changed. Preview remains an exact convolution; a trained A2 whose baked cab
exceeded its receptive field may not reproduce the very end of a long IR's
tail exactly, and validation (ESR/RMS metrics plus listening) is how you
find out whether that approximation is good enough for a given IR.

`hybrid.cab_ir.PreparedCabIr` also gained cumulative-energy diagnostics
(`energy_99_samples`/`_999_`/`_9999_samples` and `energy_fraction_within`)
so a long IR's actually-meaningful length can be seen without ever
truncating the real convolution taps.
