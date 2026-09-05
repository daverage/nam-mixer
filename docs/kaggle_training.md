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

1. Stages **only** `input.wav`, `hybrid_target.wav`, and
   `training_manifest.json` (already produced by "Generate Training Bundle")
   into a job-specific staging directory -- never your source `.nam` files or
   any preview DI.
2. Creates a **unique, private** Kaggle dataset from that staging directory.
3. Creates a **unique, private** Kaggle kernel (a Python script, not a
   notebook) requesting an `NvidiaTeslaT4` accelerator, and pushes it.
4. Polls the kernel's status without blocking the Flask server.
5. Once the kernel finishes, downloads the exported `.nam` and
   `training_result.json`.
6. Runs the exact same local verification the local trainer runs on its own
   output: parses the `.nam`, renders the official training input through it
   with the native NAMCore renderer (Full and Lite submodels), and compares
   against `hybrid_target.wav`. A job only reaches "complete" after this
   passes -- a cloud "success" that fails local verification is reported as
   failed, with the Kaggle dataset/kernel left in place for debugging.
7. Optionally cleans up the private dataset/kernel once you're satisfied
   (`POST /api/kaggle/jobs/<job_id>/cleanup`) -- never deletes the downloaded
   local `.nam`, even if cleanup itself fails.

Training hyperparameters (epochs=100, batch_size=16, ny=8192, seed=0,
latency=0) are defined once in `hybrid/a2_training_settings.py` and shared by
both the local trainer and the cloud worker -- this is enforced by
`tests/test_a2_training_settings.py`, so the two paths cannot silently drift
apart.

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
  `work/a2/<design_id>/kaggle/<job_id>/job.json` directly.

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
