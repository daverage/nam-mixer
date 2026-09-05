# Kaggle GPU training backend

Adds Kaggle GPU (NVIDIA T4) as a selectable A2 training backend alongside the
existing local trainer (`scripts/train_a2.py`), driven from the "5. Create A2"
card. See `hybrid/kaggle_training.py` for the implementation and
`cloud/kaggle/train_a2_cloud.py` for the script that actually runs inside the
Kaggle kernel.

## One-time setup

```bash
pip install kaggle
kaggle auth login
```

That's it -- once `kaggle auth login` succeeds, Hybrid NAM Builder's "Create
A2" card shows "Connected ✓" and the "Train A2" button becomes usable. This
app never reads, stores, or logs your Kaggle credentials; authentication is
entirely the Kaggle CLI's own business (`~/.kaggle/kaggle.json` /
`KAGGLE_API_TOKEN` / the CLI's own OAuth flow).

## What "Train A2" actually does

`POST /api/kaggle/train` returns almost immediately (job state `preparing`)
-- the actual work runs on a background thread
(`KaggleJobManager.submit_async`), never inline in the Flask request. A real
production upload was observed taking several minutes under real network
conditions (see "Why the upload can take minutes" below); blocking the
request for that long left the dev server unresponsive with no way to show
progress, and any interruption (Flask restart, machine sleep) lost the job's
state entirely. Poll `GET /api/kaggle/jobs/<job_id>` for progress; every step
below persists to `job.json` immediately, so a restart mid-upload only loses
the ability to keep watching that one attempt live, never the record of what
happened.

1. Stages into two SEPARATE job-specific staging directories -- never your
   source `.nam` files or any preview DI:
   - **dataset staging**: only `input.wav`, `hybrid_target.wav`,
     `training_manifest.json`, and `cloud_job.json` (already produced by
     "Generate Training Bundle") plus `dataset-metadata.json`.
   - **kernel staging**: only `train_a2_cloud.py` plus `kernel-metadata.json`.
     The cloud worker script never rides inside the dataset payload, and the
     kernel push never re-uploads the training data -- the kernel gets it
     via `dataset_sources` instead (the cloud script searches
     `/kaggle/input/**/input.wav` for it, so this split needs no script
     change).

   State: `preparing`.
2. Computes the dataset's `owner/slug` reference and **persists it to
   `job.json` before the upload even starts** -- this is the intended
   reference, not proof the dataset exists yet, but it means the reference is
   never lost even if the upload is later interrupted. Uploads to a **unique,
   private** Kaggle dataset, streaming the CLI's own progress output into
   `work/a2/<design_id>/kaggle/<job_id>/logs/kaggle.log` line-by-line as it
   happens (never `subprocess.run(capture_output=True)`, which would give no
   visibility until the whole upload finishes). State: `uploading_dataset`.
3. **Mandatory remote verification, tolerant of Kaggle's post-create eventual
   consistency** -- `kaggle datasets status == ready` is NOT trusted as proof
   the upload is complete (a real incident showed Kaggle reporting "ready"
   for a dataset containing only 1 of 5 intended files). Calls
   `kaggle datasets files <ref> -v` and confirms every required file is
   present with a size matching the local staged copy. A separate real
   incident (dataset `andrzejmarczewski/hybrid-a2-20260905t200558z-
   ab2994879d66`) proved the very FIRST such call after a successful upload
   can transiently 403/404/come back empty even though the dataset is
   genuinely complete -- manually re-checking the same dataset moments later
   showed `status: ready` and a full, correctly-sized file listing. So this
   step polls `datasets status` then `datasets files` in a bounded settling
   window (`DATASET_VERIFY_TIMEOUT_S` = 120s, retried every
   `DATASET_VERIFY_INTERVAL_S` = 3s), logging each attempt to `kaggle.log`.
   Once the listing is actually readable, an incomplete or wrong-sized
   payload fails **immediately** -- it is never retried, since waiting
   cannot fix a genuinely wrong upload. If the dataset never becomes readable
   within the timeout, the job fails with a clear message and the dataset is
   left in place for diagnosis (never auto-deleted, never blindly retried
   under a new slug). State: `verifying_dataset`.
4. Only after verification passes: creates a **unique, private** Kaggle
   kernel (a Python script, not a notebook) requesting an `NvidiaTeslaT4`
   accelerator, and pushes it (from the kernel staging directory only).
   State: `creating_kernel`.
5. `kernels push` exiting 0 is also not trusted alone -- confirms the kernel
   actually resolves via `kaggle kernels status <ref>` (bounded retry for
   Kaggle-side eventual consistency) before ever considering it submitted.
   State: `verifying_kernel`, then `queued`.
6. Polls the kernel's status without blocking Flask (`running`).
7. Once the kernel finishes, downloads the ENTIRE kernel output directory
   (`downloading`), then locates `*.nam`/`training_result.json` locally.
   A real production job failed here with `Invalid regex pattern
   '*.nam|*.json': nothing to repeat at position 0` -- Kaggle's
   `--file-pattern` flag is a **regular expression**, not a shell glob, and
   that string was never valid regex. The kernel output for this app is
   tiny compared to the training dataset, so production download omits
   `file_pattern` entirely and filters locally instead of trying to get a
   CLI-side regex right -- `KaggleCli.kernels_output`'s optional
   `file_pattern` parameter still validates via `re.compile` if a caller
   ever does pass one, so this class of mistake fails fast instead of only
   at the Kaggle API boundary.
8. Runs the exact same local verification the local trainer runs on its own
   output: parses the `.nam`, renders the official training input through it
   with the native NAMCore renderer (Full and Lite submodels), and compares
   against `hybrid_target.wav` (`validating`). A job only reaches `complete`
   after this passes -- a cloud "success" that fails local verification is
   reported as `failed`, with the Kaggle dataset/kernel left in place for
   debugging.
9. Optionally cleans up the private dataset/kernel once you're satisfied
   (`POST /api/kaggle/jobs/<job_id>/cleanup`) -- never deletes the downloaded
   local `.nam`, even if cleanup itself fails.

### Recovering a job that trained successfully but failed on download

If Kaggle actually finished training (kernel status `COMPLETE`) but the LOCAL
download/validation step failed for a reason unrelated to training itself
(the `--file-pattern` incident above being the real-world example),
`POST /api/kaggle/jobs/<job_id>/recover?design_id=<design_id>`
(`KaggleJobManager.retry_download`) re-downloads and re-validates that
completed kernel's output **without ever recreating the dataset, kernel, or
training run**. It refuses unless the job is currently `failed` and the
remote kernel actually reports a completed status -- retrying a download for
a kernel that never finished would just produce a different, misleadingly-
labeled failure. Safe to retry: it clears any partial previous local output
directory before re-downloading, so a half-written prior attempt can never
leave stale files behind to confuse `.nam`/`training_result.json` discovery.

### Why the upload can take minutes

Root-cause investigation of a real stuck-upload incident (direct CLI
reproduction, reading the installed `kaggle` package's
`ResumableUploadContext` source) found the upload mechanism itself works
correctly (repeatable ~22-25s uploads of the full ~64MB payload on a good
connection) -- but Kaggle's own client-side retry logic is bounded yet can
legitimately run long under real transient network conditions (up to 10
resumable-upload attempts, each with its own HTTP-level retry with
exponential backoff). This is why the upload timeout
(`DATASET_UPLOAD_TIMEOUT_S`, 30 minutes) is generous and why progress is
streamed live rather than waited-out silently -- that retry behavior is
legitimate, not a hang to short-circuit aggressively. It is NOT caused by
`subprocess.run(capture_output=True)`, WAV-specific handling, or multi-file
upload being unreliable -- all specifically ruled out by direct reproduction
against the real account.

Training hyperparameters (batch_size=16, ny=8192, seed=0, latency=0, plus the
epoch count -- see "Training quality presets" below) are defined once in
`hybrid/a2_training_settings.py` and shared by both the local trainer and the
cloud worker -- this is enforced by `tests/test_a2_training_settings.py`, so
the two paths cannot silently drift apart.

### Training quality presets

Both trainers accept an epoch-count preset instead of a single fixed value:

| Preset      | Epochs | Use case                          |
|-------------|--------|------------------------------------|
| `draft`     | 20     | Fast preview of the crossfade      |
| `standard`  | 60     | Normal use (UI default)            |
| `high_def`  | 120    | Best result, longest run           |

`hybrid/a2_training_settings.py`'s `A2_EPOCH_PRESETS`/`settings_for_preset()`
are the single source of truth; `cloud/kaggle/train_a2_cloud.py` mirrors them
as `EPOCH_PRESETS`/`settings_for_preset()` (parity enforced by
`tests/test_a2_training_settings.py`, same mechanism as the other shared
constants). Only `epochs` differs between presets -- batch_size/ny/seed/
latency/etc are identical across all three.

- **Local**: `python scripts/train_a2.py <manifest> --epoch-preset draft|standard|high_def` (defaults to `standard`; ignored if `--quick` is also passed, which is a separate 1-epoch smoke test, never a quality preset).
- **Kaggle**: `POST /api/kaggle/train` accepts an `"epoch_preset"` field in its
  JSON body (defaults to `"standard"`); `KaggleJobManager.submit`/
  `submit_async` take an `epoch_preset` kwarg and reject an unrecognized
  value immediately (never silently falls back). The chosen preset is
  recorded in `job.epoch_preset` and staged into the dataset's
  `cloud_job.json`, which the cloud worker reads to select its own
  `settings_for_preset()` call -- falling back to `standard` with a printed
  warning only if `cloud_job.json` is somehow missing or names an unknown
  preset (defensive, since the app itself already validates before staging).

## Accelerator

We request `NvidiaTeslaT4` and only `NvidiaTeslaT4`. Kaggle's own current
documentation warns that `NvidiaTeslaP100` is not usable with the default
Kaggle PyTorch image (its cu128 build lacks sm_60 kernels) -- P100 is never
requested by this app, and `hybrid.kaggle_training.KaggleCli.kernels_push`
actively refuses it.

## Privacy

- Every Kaggle dataset and kernel this app creates is **private**.
- Only the explicit allow-list of files above is ever uploaded -- your
  source `.nam` amp captures and preview DIs never leave your machine.
- Kaggle credentials never enter this repo, its logs, or `job.json`.

## Local training remains fully supported

The "Local" radio option in the "Create A2" card just surfaces the existing
`python scripts/train_a2.py <manifest>` command, exactly as before. Nothing
about the local training path changed.

## Troubleshooting

```bash
kaggle --version
kaggle quota
kaggle kernels status <owner>/<kernel-slug>
kaggle kernels logs <owner>/<kernel-slug>
```

- **"Kaggle CLI not installed"** -- `pip install kaggle`.
- **"Kaggle CLI is not authenticated"** -- run `kaggle auth login` yourself
  (the app's "Connect Kaggle" button just launches this for you).
- **Quota shows "unavailable"** -- this is never fatal; some Kaggle CLI
  versions don't support `kaggle quota` or it fails intermittently. Training
  itself is the authoritative check.
- **A job is stuck "already in progress"** -- only one active Kaggle job per
  design is supported. Wait for it to finish/fail, or check
  `work/a2/<design_id>/kaggle/<job_id>/job.json` directly. A job stuck by a
  bug in a previous version of this app (e.g. an unresolvable kernel
  reference) is automatically detected and failed the next time it's
  refreshed or a new submission is attempted for that design -- it will
  never block forever.

### Job states

`preparing` -> `uploading_dataset` -> `verifying_dataset` -> `creating_kernel`
-> `verifying_kernel` -> `queued` -> `running` -> `downloading` ->
`validating` -> `complete` (or `failed` from any step). `cleanup_pending`/
`cleaned` describe the separate post-completion cleanup step, not training
progress.

## Manual end-to-end smoke test (not part of `pytest tests/`)

This requires real Kaggle credentials and consumes a small amount of your
Kaggle GPU quota. Do not run this in CI.

1. `kaggle auth login` (once).
2. Generate a training bundle from the UI (or `POST /api/generate`).
3. `POST /api/kaggle/train` with `{"design_id": "<your design id>"}`.
4. Poll `GET /api/kaggle/jobs/<job_id>?design_id=<design_id>` until `state`
   is `complete` or `failed`.
5. On success, confirm `output_nam_path` exists locally and `local_validation`
   shows `full.rendered_ok: true`.
6. `POST /api/kaggle/jobs/<job_id>/cleanup?design_id=<design_id>` to remove
   the private Kaggle dataset/kernel.

A full 100-epoch run is not launched automatically by this app or by any
setup script -- only an explicit `POST /api/kaggle/train` call starts real
Kaggle GPU usage.
