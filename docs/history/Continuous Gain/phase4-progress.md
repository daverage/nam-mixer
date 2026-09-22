# Phase 4 implementation report

Status: implementation and automated checks complete; interactive browser
acceptance remains pending. This report records technical evidence, not a
claim of perceptual equivalence. No cloud training job was launched.

## Milestone status

1. **Complete — source identity and UI invalidation.** Every cached-pair consumer
   requires an opaque render ID. Render snapshots bind retained source bytes,
   hashes, DI, profile/calibration, test gain, and per-amp trims. Frontend
   invalidation stops old playback and uses generation counters to reject late
   render, preview, diagnostic, generation, and live-decode responses. Cheap
   shape controls continue to reuse the valid pair.
2. **Complete — causal Character Blend target.** Teacher semantics v2 uses a
   causal donor-weight ramp starting at each detected change and restarts a
   reversal from the current weight. Legacy version 1 remains readable and
   reproducible; unknown versions are rejected. Prefix, boundary, constant,
   reversal, short-clip, determinism, and frozen full-teacher tests pass.
   Temporal reporting includes envelope, drive/compensation smoothing, donor
   settling, and correction FIR. Repeated mid-ramp reversals are honestly
   recorded as stateful rather than assigned a false finite history bound.
3. **Complete — equivalent quiet-playing validation.** Reference schema v2
   selects a deterministic energetic excerpt with warm-up/context and hashes
   its source, model sources, and cabinet. The frozen teacher receives a baked
   cabinet and the fixed output/safety gains exactly once. Full and Lite are
   scored separately for raw/gain-normalized ESR, absolute level offset,
   response-shape error, and extra quiet attenuation. Invalid, tampered,
   legacy, silent, or insufficient evidence is unavailable rather than passed.
4. **Complete — completion versus validation quality.** Shared validation
   report schema v2 has passed/failed/unavailable checks and a documented v1
   policy. Local and Kaggle paths persist the same Full/Lite/quiet results;
   operational failures are separate from computed poor quality and retained
   exports remain downloadable. Result panels expose summaries plus expandable
   metrics. Sessions restore reports only when the embedded NAM hash matches;
   NAM Tools explicitly invalidates prior validation.
5. **Complete — real-render regression harness.** The opt-in schema-v2 harness
   accepts explicit renderer/source/DI/cab/export paths, records hashes and
   runtime/build identity, covers all modes, cab on/off, -12/0/+12 dB input,
   invariants, determinism, and frozen-teacher equivalence, and never trains.
   Developer runs record unavailable prerequisites; release runs fail them.
6. **Complete — held-out musical listening comparison.** Completed models can
   render a saved frozen teacher and Full/Lite on the selected musical DI at
   normal or -24 dB external input gain. One multichannel WAV provides
   synchronized raw-level switching. The teacher alone receives frozen
   source calibration/per-amp trims and baked-cab/output/safety processing.
   Metrics use the shared validation code. Cache identity binds manifest,
   design, model, DI, cabinet, level, sample rate, and hashes. Missing imported
   teacher assets return a stable explanation while preserving the NAM.
7. **Complete — renderer readiness and recovery.** A cheap `--help` probe
   distinguishes missing, found-but-unusable, and verified executables before
   inference. The UI provides expandable path/build help and an explicit retry;
   it never installs or compiles as a status-check side effect. API and
   JavaScript recovery tests prove retry works without restarting the app.

## Verification evidence

- Full Python suite: **415 passed, 5 skipped** (the skips are optional native /
  platform-dependent cases; no failures).
- JavaScript browser-state suite: **7 passed**; `node --check static/app.js`
  passes.
- `git diff --check` passes.
- Actual release harness report:
  `work/phase4-real-render/real_render_report.json` (ignored local evidence),
  schema 2, state `passed`, 2 musical DIs × 3 levels = 6 cases, 36 recorded
  mode/cab listenable results, exact rerender determinism, maximum frozen
  teacher error 0, no failures, elapsed 6.65 seconds.
- The real source pair had distinct clean/high-gain behavior. Auto-calibrated
  combinations were explicitly unavailable because neither source declared
  `input_level_dbu`; raw calibration was tested.
- Interactive browser execution was attempted against a running local server,
  but no in-app or extension browser was connected in this environment. It is
  recorded as unavailable manual evidence, not replaced by a mocked success.

## Compatibility and boundaries

- Follow-up fixes preserve completed artifacts when preview settings change
  and allow comparison retries after cancellation without accepting stale
  responses. Wizard instrument/pickup selections survive DI changes, and its
  crossover is retained on the first render. Targeted verification: 52 Python
  tests and 7 JavaScript state tests passed.

- Designs without `teacher_semantics_version` retain Character teacher v1.
  New Character designs use v2; unsupported future versions fail explicitly.
- Legacy quiet-reference or validation-report schemas are shown as unavailable.
- Imported sessions keep embedded NAMs usable even when source models, cabinets,
  or the original bundle needed for teacher reconstruction are absent.
- Automatic A/B alignment remains disabled by default. Cabinet IRs are not
  truncated. Preview pickup/test gains remain separate from official training
  excitation. Objective checks do not establish musical quality; listen to each
  export and use the supplied comparison as evidence, not a guarantee.
