# Phase 4 — Reliable design state, causal targets, and trustworthy validation

## Objective

Make the existing NAM Mixer workflow dependable from session loading through
target generation, training, validation, and listening comparison.

This is an implementation plan, not a statement that the work is complete.
Complete the milestones in order. Do not mark a milestone complete merely
because its UI exists or its tests use successful mocks.

The baseline reviewed for this plan is commit `73a47c4`. Line numbers may move;
use the named files and functions to find the implementation at the current
HEAD. Use the existing repository, not a new project.

## 0. Preparation and boundaries

1. Read `README.md`, `CLAUDE.md`, the applicable `AGENTS.md`, and this plan.
2. Inspect `git status` and current HEAD. Preserve unrelated work. Do not pull,
   reset, or overwrite changes merely to match the baseline above.
3. Use codebase-memory graph discovery as required by repository instructions.
   Verify source when the graph is stale or incomplete.
4. Run `python3 -m pytest -q` in the app environment. Record failures and skips
   before changing code. Do not install training dependencies into this environment.
5. Inspect the named implementation and tests before each milestone. If a
   defect has already been fixed, verify its acceptance criteria instead of
   duplicating the implementation.
6. Track each milestone as pending, in progress, complete, or blocked with
   evidence. Describe missing renderer/models/training dependencies explicitly.

Preserve these rules throughout:

- Keep native NAMCore inference and the separate local training environment.
- Keep automatic A/B alignment disabled in the app. Do not add an experimental
  alignment checkbox as a substitute for establishing a valid latency policy.
- Keep official training excitation separate from musical preview DIs. Never
  bake preview pickup-profile or test gain into the training excitation.
- Preserve intentional per-amp trims and NAM calibration in their existing roles.
- Preserve all three design modes, Sessions, NAM Tools, and both training backends.
- Keep the full prepared cabinet IR for preview and baking. Do not silently
  truncate it to pass a receptive-field check.
- Keep training-target peak control a single fixed gain reduction. Never use
  the preview limiter to make a training target.
- Do not claim that an objective check proves musical or perceptual equivalence.
- Do not launch paid/cloud training, upload private captures, or redistribute
  source models without the user's authorization and appropriate rights.

## 1. Bind every preview and generated bundle to the correct rendered sources

### Problem and evidence

`static/app.js:applySessionSettings` restores source paths and controls, then
adds a stale-render message. It does not clear an existing `havePair` flag.
The generation handler checks `havePair`; `app.py:api_generate` reads sources
from the process-wide `_rendered_pair_cache`.

Consequently, rendering one design and loading another session can leave the
old render usable under the new controls. A second browser tab can also replace
the shared server cache. A warning alone does not establish source identity.

### Implementation

1. Introduce one frontend invalidation function. Call it whenever a change
   requires new NAM inference: either source model, preview DI, input profile
   or custom gain, instrument where relevant, calibration/reference level,
   test gain, or per-amp input trim. Use it during session loading too.
2. Invalidation must clear the active render identifier and `havePair`, stop
   playback/live blending based on the old stems, discard cached audition
   audio, and disable actions that require the current rendered pair.
   Preserve saved/completed training artifacts as separately identified records.
3. Audit existing control-sync functions so they cannot re-enable generation
   merely because training is idle. A current render and ready training input
   must also be required.
4. Have `/api/render_pair` publish an opaque render identifier only after the
   pair and its metadata are complete. Associate it with the source identities
   and every input setting that affects rendering. Prefer content hashes over
   filenames for source identity, reusing existing hashing helpers.
5. Send that identifier with all operations that consume the cached pair:
   preview, blend information/curves where applicable, live stems, quiet-playing
   checks, and generation. Check it server-side before using the pair.
6. A mismatched or missing identifier must produce a structured stale-render
   response (for example HTTP 409 with a stable error code) and an instruction
   to render again. Update all internal clients/tests together. Never silently
   accept a legacy request that could consume the wrong pair.
7. A single cached pair is acceptable initially: another tab may invalidate it,
   but must never cause silent reuse. Capture pair and metadata as one snapshot
   under suitable synchronization; do not read changing cache fields separately.
8. Protect against out-of-order frontend responses. If settings change while a
   render is running, its eventual response must not make the newer design ready.
   Use a request generation counter or equivalent explicit state tracking.
9. Generation must use the verified snapshot throughout the operation. Include
   source hashes and the frozen render settings in bundle provenance. An ephemeral
   identifier alone is not sufficient provenance after an app restart.

### Tests and acceptance

- Render sources A/B, load a session for C/D, then attempt preview and generation:
  no old audio is returned and no bundle is generated until C/D is rendered.
- Change each render-dependent setting; old identifiers are rejected.
- Change only a cheap shape control; the pair remains usable without inference.
- Simulate two tabs and out-of-order render responses; stale operations fail clearly.
- A failed render never leaves the design appearing ready.
- Run backend route tests and a browser interaction check for session switching.
  Backend tests alone do not verify frontend readiness/playback behavior.

## 2. Make Character Blend donor transitions causal

### Problem and evidence

`hybrid/character_blend.py:_select_donor` locates future donor switches and
crossfades from `point - fade` to `point + fade`. That changes samples before
the control actually switches. Both preview and target generation call this code.

The review reproduced this with zero-valued Amp A, one-valued Amp B, and a
control that first switches to B at sample 50: output differed from sample 41
at a test sample rate of 1 kHz. The teacher therefore uses future information
when the level-dependent drive control changes donor.

### Implementation

1. Add a failing prefix-invariance test for `_select_donor`: inputs and controls
   identical before sample N must produce identical output before N regardless
   of what happens after N. Keep model/analysis configuration fixed.
2. Replace the centered transition with a bounded, causal transition starting
   at the detected switch. At sample t, use only control history through t.
3. Use an explicit donor-weight state. On a switch, ramp from the current weight
   toward the requested donor over a documented finite duration. If another
   switch occurs mid-ramp, restart from the current weight so it does not jump.
4. Preserve settled donor semantics: below 50% selects A, 50% or above selects
   B; after the transition the output is exactly the selected donor waveform.
   Do not turn the steady-state Drive control into a parallel mix.
5. Define initialization explicitly: a constant B selection from sample zero
   should begin at B, without an unintended startup fade from A.
6. Audit downstream gain/EQ assumptions during transitions in
   `build_character_blend`. Avoid introducing an abrupt compensation change
   while the donor waveform itself is smoothly transitioning.
7. Add full-teacher prefix tests with frozen analysis. Full-clip analysis is
   offline design configuration; freeze it before testing runtime causality.
8. Audit Character Blend temporal history accounting: donor transition,
   envelope, smoothing, and correction FIR dependencies. Update receptive-field
   reporting where necessary and document how serial/parallel dependencies
   combine. Do not assume the Dynamic Hybrid formula automatically covers it.
9. Version the changed teacher semantics in manifests/design serialization.
   Do not silently rewrite existing targets or claim old and new targets are
   equivalent. Define behavior for legacy sessions/designs and preserve old bundles.

### Tests and acceptance

- The sample-50 reproduction leaves all samples before 50 unchanged.
- Cover A-to-B, B-to-A, constant A/B, rapid reversals, short clips, and boundaries.
- Every transition weight stays in [0, 1] and settles within the documented bound.
- Identical inputs and frozen configuration give deterministic output.
- Preview and generation use the same revised teacher implementation.
- Run Character Blend, target-generation, and receptive-field tests. Listen to
  real switching examples before claiming the transition sounds better.

## 3. Compare equivalent teacher and trained-model signals

### Problem and evidence

`hybrid/character_training_target.py:generate_character_training_bundle`
records the low-level teacher sweep before cabinet baking, output gain, and
the final peak reduction. `check_full_low_level_response` compares the exported
model against that sweep, even though the model learns the processed target.
Intentional processing can therefore appear as a model error.

### Implementation

1. Keep the existing pre-processing sweep as a teacher-health check if useful,
   but name it separately from the reference used to validate the exported model.
2. After target generation determines the actual fixed output gain and peak
   reduction, build the validation reference sweep using the frozen teacher,
   baked cabinet only when enabled, and those same fixed gains.
3. Never recompute auto gain or normalize independently at each sweep level.
   That would erase the dynamic response the test is intended to measure.
4. Freeze and record the reference excerpt, frame range, sample rate, input
   gains, model/IR hashes, processing version, and applied output gains.
   Ensure the excerpt contains useful excitation; do not assume the first five
   seconds of every accepted input are suitable. Use deterministic selection
   or explicit rejection when no valid excerpt is available.
5. Use equivalent initialization/warm-up and sample windows for teacher and
   exported model. If using context before the measured excerpt, include it
   consistently and document which samples are scored.
6. Measure both absolute level error and response-shape error across input levels.
   Report a consistent level offset separately from an extra low-level collapse.
7. Share this comparison between local and Kaggle validation. Extend to Full
   and Lite with distinct outcomes rather than treating a Full result as Lite proof.
8. Version the reference schema. For old manifests without an equivalent
   reference, report the relevant check as unavailable/legacy, or explicitly
   regenerate it when all original assets are available. Do not silently pass it.

### Tests and acceptance

- A simulated perfect trained target passes with baked cabinet on/off, manual
  gain, auto gain, and a nonzero fixed peak reduction.
- A fixed level mismatch is reported separately from dynamic-response collapse.
- A model that becomes silent only at quiet levels is detected.
- Invalid, silent, or insufficient reference material cannot produce a false pass.
- Local/cloud callers receive equivalent structured results for the same inputs.
- Keep thresholds documented and configurable through a shared policy. Do not
  introduce unexplained thresholds or describe the existing 20 dB allowance as
  proof of good tone matching.

## 4. Separate training completion from validation quality

### Problem and evidence

`hybrid/kaggle_training.py:validate_downloaded_model` records Lite render
failure and low-level check failure without necessarily raising an error. Its
caller can still set the job to `complete`. The local trainer prints a failed
low-level verdict and continues. The frontend download panels expose little
of this distinction.

### Implementation

1. Keep process completion distinct from model assessment. Define a shared,
   versioned validation report consumed by both backends and Sessions.
2. Represent individual checks as passed, failed, or unavailable, with a
   reason. Include Full/Lite render validity, raw and gain-normalized ESR,
   level error, relevant quiet-playing results, and cabinet approximation notes.
3. Define an aggregate state such as `passed`, `needs_attention`, or `unavailable`.
   A failed required check or failed advertised Lite variant must never yield
   an unqualified validation pass. Missing checks must not count as passed.
4. Keep exceptions for operational failures distinct from a validly computed
   poor-quality result. Preserve imperfect exports and their reports for inspection.
5. Add the same result summary to local and Kaggle completion views: state,
   understandable explanation, Full/Lite outcomes, and expandable metrics.
   Keep raw paths and hashes secondary to the result and download actions.
6. Persist the report with the generated bundle and session artifact, tied to
   the exported model hash. If NAM Tools creates a modified model, do not carry
   forward a pass as though it validated those modified bytes.
7. Audit download, retry, cleanup, and restore behavior against the new report.
   A failed quality check must not trigger automatic deletion of useful artifacts.

### Tests and acceptance

- Full succeeds and Lite fails: training is finished, but validation needs attention.
- Quiet-playing validation fails: visible warning appears locally and on Kaggle.
- All required checks pass: the report says what was checked, without claiming
  perceptual equivalence.
- Missing/legacy reports display unavailable rather than a green pass.
- Report survives session reload/export/import with correct model identity.
- Verify both completion panels in the browser, including failure cases.

## 5. Establish a repeatable real-render regression harness

### Existing test boundary

`tests/test_render.py` requires one specifically named untracked Fender model
and a built renderer. `tests/test_pipeline_render.py` replaces inference with
an identity function. These tests are useful, but do not establish full
real-model coverage across design modes and training exports.

### Implementation

1. Add an opt-in integration entry point accepting explicit source-model paths,
   renderer location when needed, output directory, and selected musical DIs.
   Follow existing test/CLI conventions; do not hard-code personal paths.
2. Keep personal captures outside version control. A redistributable fixture
   may be added only after its permission/license has been established.
3. Record source/DI hashes, renderer/build identity where available, settings,
   timings, and a machine-readable report under ignored `work/` output.
4. Exercise real inference through Dynamic Hybrid, Parallel Blend, and Character
   Blend. Cover cabinet on/off, level extremes, and available calibrated/raw
   combinations. Report unsupported or unavailable combinations explicitly.
5. Check sample rate, length, finiteness, silence, determinism, and mode-specific
   invariants. Save useful stems and results for human listening.
6. Compare preview and frozen-teacher processing only with equivalent input,
   calibration, cabinet, and output-gain settings. Intentional preview profile
   differences must not be mistaken for processing drift.
7. Accept existing exported A2 files for Full/Lite validation. Keep expensive
   retraining separate and opt-in; the harness must not start cloud jobs itself.
8. Provide two modes: optional developer execution may skip missing assets;
   an explicitly requested release check must fail clearly if prerequisites
   are missing, so an entirely skipped run cannot be reported as successful.

### Tests and acceptance

- The harness runs against two real user-provided models and writes reproducible
  reports and listenable artifacts, or records exactly which prerequisites block it.
- Mocked orchestration tests are separate from evidence of real native execution.
- Include source pairs with distinct distortion/level behavior when available.
- Move tests that do not require native inference out of the module-wide real-model
  skip in `tests/test_render.py`, so basic error handling is always exercised.
- Record an actual real-render result before removing the README testing caveat.

## 6. Add target-versus-export listening on held-out musical material

1. Add a comparison action for completed models using a saved, frozen design and
   a musical DI that was not the training excitation. Support all three modes.
2. Render the teacher and Full/Lite model with equivalent external input gain.
   Apply calibration/per-amp trims only in the teacher path where specified by
   the frozen design; do not double-apply them to the trained model.
3. A baked cabinet is already represented by the trained model. Do not apply
   that cabinet twice. An optional audition-only cabinet must be applied equally
   to teacher and model after their output, and labeled accordingly.
4. Provide synchronized switching between teacher and model, a quiet-playing
   example, and basic metrics from the shared validation code.
5. Default to actual output levels. If adding loudness-matched listening, make
   it explicit and preserve the raw metrics; do not normalize each quiet/loud
   segment separately or hide dynamic-response differences.
6. Associate cached comparisons with design, model, DI, cabinet, settings, and
   hashes. Use the stale-result protection introduced in milestone 1.
7. If source captures/IRs are missing after session import, explain that teacher
   reconstruction is unavailable while keeping the embedded NAM usable.

Acceptance: users can hear and inspect differences on unseen musical material
for all three modes, and no comparison silently substitutes current controls
for the design that actually produced the trained model.

## 7. Improve renderer setup feedback

1. Reuse `hybrid/render.py:find_nam_render_exe` to expose a lightweight readiness
   check. Distinguish a found binary from a successfully verified renderer.
2. Show an actionable message before the first attempted render when it is
   missing or unusable. Put detailed paths/build instructions in expandable help.
3. Keep optional local-training and Kaggle readiness separate from preview readiness.
4. Add a retry action after installation. Do not compile/download software as a
   side effect of loading a page or checking status.
5. Cover missing binary, unusable binary, working renderer, and recovery without
   restarting the browser. A readiness check must not perform expensive NAM inference.

## Final verification and handoff

- Run targeted tests after each milestone and the full suite after integration.
- Run browser checks for restored sessions, stale renders, and both training
  result panels. Exercise failures as well as successful requests.
- Run the opt-in real-render harness when authorized assets and renderer exist.
  List unavailable evidence rather than replacing it with mocked results.
- Update README/CLAUDE only after behavior is implemented and verified. State
  remaining limitations accurately; keep the alignment and perceptual-quality
  boundaries unless there is evidence that changes them.
- Record schema/teacher-version changes and handling of legacy bundles/sessions.
- Review `git diff --check`, changed files, and ignored generated artifacts.
  Follow the user's current commit/push instructions; do not bundle personal
  models, audio outputs, credentials, or local index artifacts into commits.
- Handoff must list completed milestones, tests and real-audio evidence, unresolved
  blockers, and any user-visible compatibility changes.

## Completion checklist

- [x] 1. Session/render consistency is enforced by frontend and backend.
- [x] 2. Character Blend donor transitions are causal and versioned.
- [x] 3. Quiet-playing validation compares equivalent processed signals.
- [x] 4. Quality results are distinct from job completion and visible in Sessions/UI.
- [x] 5. Real-render harness exists and its actual execution status is recorded.
- [x] 6. Held-out musical comparison works with correct signal-chain semantics.
- [x] 7. Renderer readiness and recovery are clear before rendering.
- [ ] Final acceptance: automated regression checks and documentation are complete;
  interactive browser checks remain pending (no connected browser was available).
