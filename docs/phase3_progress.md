# Phase 3 progress log: real A2 hybrid training pipeline

Working notes on implementing docs/phase3.md (produce the first genuinely
usable NAM A2 hybrid model). Written up for reference since this spanned a
long session with a live background training run.

## Goal

Turn the existing two-NAM dynamic hybrid (Amp A at low input, Amp B at high
input, crossfaded by a dry-input envelope) into a real, trained NAM A2 model
that reproduces that behavior on its own, without needing the two source
amps or the blend logic at inference time.

## What got built

**`hybrid/design.py`** — `HybridDesign`, an immutable snapshot of an
auditioned crossover/transition/trim/calibration configuration, frozen via
`freeze_design()` from a real `RenderedPair`/`HybridResult` so generation can
never silently recompute auto-trim or the crossover against a different
signal than what was actually heard.

**`hybrid/envelope.py`** — `bounded_causal_envelope_db`, a genuinely
finite-memory causal envelope (three cascaded FIR stages: 20ms moving RMS →
5ms moving average → 55ms decaying-max release = 80ms total bounded history)
replacing the old `rms_envelope_db`'s one-pole release, which had
theoretically infinite recursive memory — unacceptable as the crossover
control signal baked into a training target for a finite-receptive-field
model. Wired in as the new default in `hybrid.pipeline.render_pair`.

**`hybrid/receptive_field.py`** — computes the real installed A2
(PackedWaveNet) receptive field straight from `neural-amp-modeler`'s own
`config_model_packed.json` (no torch import needed, pure JSON/dilation math),
and separately `compute_source_nam_receptive_field()` for an arbitrary
source `.nam` capture (handles both plain `WaveNet` and `SlimmableContainer`
schemas). Verified live against the real installed package.

**`hybrid/training_target.py`** — `generate_training_bundle()`: blends the
**official** NAM training excitation (never the preview DI, never with the
design's pickup-profile gain applied) through the frozen design. Only
per-model NAM calibration compensation is applied (same rule as preview).
Requires the input to MD5-match the official **v3.0.0** file specifically
(`OFFICIAL_V3_INPUT_MD5`), not just "any recognized NAM input" — the current
trainer's own data checks are calibrated around V3's validation-signal
layout. Writes `input.wav` (byte-identical copy, not a re-encode — see Bugs
Found below), `hybrid_target_raw.wav`, `hybrid_target.wav`,
`hybrid.hybrid.json`, `training_manifest.json`.

**`hybrid/validation.py` / `scripts/validate_a2.py`** — held-out validation:
render the same held-out DI through both the live two-NAM reference hybrid
(reusing the frozen design) and a trained A2 export, at several real
input-profile gains (applied as actual audio gain, never the deprecated
`dry_gain_db`), and compare via ESR/RMS/peak metrics. Runs entirely on the
native NAMCore renderer — no torch needed, works in the normal app
environment.

**`scripts/train_a2.py`** — standalone (no Flask) wrapper: environment
diagnostics → manifest/hash validation → receptive-field check → calls the
**real, verified** `nam.train.core.train()` → exports via `BaseNet.export()`
with real training + user metadata → validates the exported `.nam` through
NAMCore (Full and Lite) → compares against the target.

**`app.py`** — `/api/generate` is real now (was a 501 stub); added
`/api/training_input/status` and `/api/training_input/upload`. UI got a
compact "5. Create A2" card.

**`requirements-training.txt` / `scripts/setup_a2_env.ps1`** — pinned,
separate training environment (`neural-amp-modeler==0.13.0` + Torch) so the
main Flask app stays torch-free.

## Real environment used

- Main app env: Python 3.14, no torch (as designed).
- Training env: `.venv-a2` (gitignored), Python 3.14 (no 3.12 available on
  this machine), `neural-amp-modeler==0.13.0`, `torch==2.14.0+cpu`. Both
  installed cleanly despite the "3.14 too new for torch" assumption in
  `requirements-training.txt`'s comments — that assumption turned out wrong
  in practice and should be revisited if this doc is read later.
- CPU-only (no CUDA/MPS on this machine) — training is slow (~3 min/epoch
  for this dataset/model size).

## Key discoveries made by inspecting the real installed package

Rather than guessing at the `neural-amp-modeler` 0.13.0 API, it was
installed and inspected live (`inspect.signature`, `inspect.getsource`):

- The real training entry point is `nam.train.core.train(input_path,
  output_path, train_path, epochs=100, latency=None, ..., silent=False,
  ..., user_metadata=None, fast_dev_run=False) -> TrainOutput | None`. It
  always loads `_get_packed_model_config()` (the exact
  `config_model_packed.json` file the receptive-field check reads) — so this
  genuinely is the official A2/PackedWaveNet architecture.
- `train()` does **not** export a `.nam` on its own; `TrainOutput.model.net`
  is a `BaseNet` with a real `.export(outdir, basename, user_metadata,
  other_metadata)` method that does the actual writing. Upstream's own
  `nam.train.colab.py` calls this explicitly after training, passing
  `other_metadata={nam.train.metadata.TRAINING_KEY:
  train_output.metadata.model_dump()}` — `scripts/train_a2.py` now follows
  the same pattern.
- The packed config schema (verified, not assumed):
  `{"net": {"config": {"submodels": [{"name": "channels_3", "config":
  {"layers_configs": [{"kernel_sizes": [...], "dilations": [...]}]}}, ...]}}}`
  — `kernel_sizes` is a **per-layer list**, not one shared kernel size.
  Both submodels (`channels_3`/"Lite", `channels_8`/"Full") currently share
  the same 23-layer schedule → a 6332-sample (~131.9ms @ 48kHz) receptive
  field.
- The official NAM **v3.0.0** training input file's download link
  (`https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G/`) is
  embedded directly in `nam/train/gui/__init__.py`'s `_download_input_file`
  method (`_INPUT_BASENAMES`/`_LATEST_VERSION` in `nam/train/_names.py`
  confirm v3.0.0/`input.wav` is the current recommended version). Downloaded
  and verified via `nam.train.core._detect_input_version`: **strong MD5
  match** (`36cd1af62985c2fac3e654333e36431e`), mono, 48kHz, 3:10 duration.
  Saved at `work/training_input/input.wav` (gitignored).
- `nam.train.core._detect_input_version` identifies the official input by
  MD5 against a small hardcoded table (v1.0.0, v1.1.1, v2.0.0, v3.0.0,
  Proteus) — this is the same mechanism `hybrid/training_target.py` now uses
  itself (`OFFICIAL_V3_INPUT_MD5`), so the check works even without
  `neural-amp-modeler` importable (pure `hashlib`, no torch/nam needed).

## Bugs found and fixed during a live run (not caught by unit tests alone)

Running the real pipeline end-to-end (not just mocked tests) surfaced three
real bugs:

1. **`input.wav` was being re-encoded, not copied.** `generate_training_bundle`
   wrote `official_input.astype(np.float32)` via `soundfile` instead of
   copying the original file bytes. This silently changed the file's hash,
   breaking both `nam.train.core`'s own strong-match detection and our own
   provenance check in `scripts/train_a2.py`. Fixed: `shutil.copyfile`.
2. **Target hash mismatch: array bytes vs. file bytes.** The manifest
   recorded `sha256(raw_array_bytes)` but `scripts/train_a2.py` checks
   `sha256(file_on_disk)` — a WAV file's bytes include the RIFF/fmt header,
   so these never matched. Fixed: hash the written files, not the in-memory
   arrays.
3. **Blocking interactive plots hung a headless run.** `core.train(...,
   silent=False, ...)` triggers `matplotlib` plot windows during latency
   calibration; in a non-interactive/piped run this blocked indefinitely (a
   16-minute run was killed before this was diagnosed — CPU was still being
   consumed by *something*, but no forward progress). Fixed: `silent=True`.
   docs/phase3.md section 31 had already called this out ("suppress
   interactive plots where supported") — worth reading requirements twice
   before writing the wrapper.

## The receptive-field "zero margin" finding

`check_receptive_field` in `scripts/train_a2.py` computes
`max(envelope history, Amp A receptive field, Amp B receptive field)` — not
just the envelope, because Amp A/Amp B run on the same dry input in
*parallel* with the crossover envelope (see `hybrid.pipeline.render_pair`).

For the JCM800/Fender pair, this surfaced something real: **both source
captures are themselves already-trained A2 models** with the exact same
131.9ms receptive field as the new A2 being trained. So the crossover
envelope (80ms, comfortable) isn't the binding constraint — the source amps'
own receptive field is, and it exactly equals (not merely fits inside) the
new A2's own capacity.

First instinct was to treat `required == available` as a hard failure
(`>=` check). This was **wrong** and got corrected on review: Amp A, Amp B,
and the crossover envelope are parallel branches feeding a **memoryless**
per-sample blend — their temporal requirements take the max, not the sum,
and a memoryless combination adds no extra history on top of whatever the
slowest branch already needs. So `required == available` is a legitimate
exact fit (zero *temporal* margin, which is a different thing from *model
capacity* — receptive field only measures how far back the network can
look, not whether it has enough parameters to learn the composite function
within that reach). Fixed to `required > available` as the actual failure
condition, with a prominent warning (not an abort) at exact fit.

This reframes the experiment nicely: we're asking one A2 to learn a
function built from *two other A2 models* plus a level-dependent crossfade
between them — a genuinely interesting test of whether one A2 can absorb
that composite behavior within the same temporal reach it would need for
either source amp alone.

## Status at time of writing

- All review fixes applied, tested (132 passed / 3 skipped main env, 133
  passed / 2 skipped training venv), committed, and pushed:
  - `b875494` — initial phase-3 pipeline
  - `e692280` — input-copy / hash-file fixes (applied directly)
  - `61c4016` — V3 enforcement, minimal-attenuation safety, source-amp RF
    check, real export metadata, `silent=True` fix
  - `598c884` — held-out validation pipeline
- `--quick` (1-batch) smoke test passed completely: V3 strong-match, NAM's
  own data checks pass (Replicate ESR = 0.0 — exact match on the repeated
  validation section, as expected for a fully deterministic synthetic
  target), export carries real `UserMetadata` + `TrainingMetadata` (verified
  by inspecting the exported `.nam` directly), NAMCore loads and renders
  both Full and Lite submodels.
- **Real 100-epoch training is running now** in the background
  (`work/a2/jcm800_to_fender/a2_output_full`, PID 11972, started ~18:29,
  log at `/tmp/train_full.log` — a session-temp path, not durable). CPU-only,
  ~3 min/epoch, so a ~5-hour run. As of epoch 10/100: ESR down from 1.693 to
  1.039, decreasing steadily. Survived a session restart (it's a plain
  detached OS process, not tied to the Claude Code session).

## Next steps once training finishes

1. `scripts/train_a2.py` will have already run the Full/Lite NAMCore
   validation + target-comparison metrics and updated
   `training_manifest.json`'s `training` section — check that first.
2. Run `scripts/validate_a2.py` against held-out DIs (`clean_smooth.wav`,
   `high_thrash.wav`, and a bass DI) at the profile gains docs/phase3.md
   section 25 lists (-7, -3, 0, +1, +4.5, +6 dB), to see whether the trained
   A2 actually preserves "quiet → dirty JCM800, loud → clean Fender" —
   that's the real pass/fail question, not the training-set ESR.
3. Listen to the generated `sequence.wav` files (reference vs. Full vs.
   Lite) per case.
4. Write the final report per docs/phase3.md section 40 once validation
   results are in.

## Files/paths to know about (all gitignored, not in the repo)

- `.venv-a2/` — the training environment.
- `work/training_input/input.wav` — the real official V3 file.
- `work/a2/jcm800_to_fender/` — the generated bundle + training output for
  this run (`hybrid_design.json`, `input.wav`, `hybrid_target.wav`,
  `hybrid_target_raw.wav`, `hybrid.hybrid.json`, `training_manifest.json`,
  `a2_output/` = the `--quick` smoke test's output, `a2_output_full/` = the
  real training run's output).
