# Standalone desktop app: removed 2026-09-18

## Reported symptoms

While running a Kaggle GPU training job from the packaged standalone desktop
app (`desktop/`, built with PyInstaller + pywebview), the user reported:

- The app kept opening new instances of itself repeatedly while training was
  in progress.
- Uncertainty about whether the Kaggle dataset upload was actually happening.
- The packaged binary was large.
- The app icon wasn't reliably compiled into release builds.

## Investigation (static code review; no packaged build was actually reproduced)

**Repeated app instances -- leading suspect.**
`hybrid/kaggle_training.py`'s `KaggleCli._build_argv` had a frozen-build
branch that, when no real `kaggle` CLI was on `PATH`, shelled out to
`[sys.executable, "--run-kaggle-cli", *args]` -- i.e. it relaunched the
packaged app's own executable as a subprocess, relying on
`desktop/main.py`'s `__main__` guard to detect the `--run-kaggle-cli` flag
and act as a plain CLI instead of opening a window. The frontend polled
`GET /api/kaggle/jobs/<id>` every 10 seconds during an active job
(`static/app.js`), and each poll called `kernels_status`/`username`, so this
self-relaunch fired repeatedly throughout a training run. This pattern only
existed because the desktop build has no separate real Python interpreter to
shell out to -- a normal `python app.py` server just runs the real `kaggle`
console script or `python -m kaggle`, with no self-relaunch trick needed at
all. This was assessed as the most likely cause of the reported symptom, but
**it was never confirmed by actually reproducing the bug on a packaged
build** -- only inferred from reading the dispatch code and the poll
interval.

**Kaggle upload trustworthiness.**
The dataset/kernel push logic itself (`datasets_create_streaming`,
`kernels_push`, etc.) looked correct on review, but it runs through the same
`_build_argv` dispatch as above, so it was reasonable to trust it less on
the packaged build specifically, pending the point above.

**Icon not always compiled in -- confirmed real bug (fixed, now moot).**
`desktop/build.spec` failed the build loudly if the `nam_render` binary was
missing, but silently fell back to `icon_arg = None` if
`desktop/icons/icon.icns`/`.ico`/`.png` were missing -- a build could
succeed with no app icon and nobody would notice until the shipped app
looked wrong. This was fixed to fail loudly, matching the `nam_render`
check, before the whole desktop app was later removed.

**Binary size.**
The build used `--onedir` (not `--onefile`, so no per-launch re-extraction
overhead), but still bundled a full CPython + numpy/scipy + the `kaggle`
package (via `collect_all`) + pywebview + a second copy of `hybrid/` for the
training runtime + the native `nam_render` binary. This was assessed as
inherent packaging overhead for a self-contained Python GUI app, not a
distinct bug -- but it's the cost of carrying a full interpreter along
specifically to make the self-relaunch trick above work.

## Decision

All three real problems (spurious instances, upload trust, size) trace back
to the same root architecture: the desktop wrapper faking a second Python
interpreter by re-invoking its own packaged binary. Rather than hardening
that pattern further, the standalone desktop app was removed entirely.
Local server (`python app.py` / `scripts/run.sh` / `scripts/run.ps1` /
`scripts/run.command`) is now the only supported way to run NAM Mixer, and
already has one-click setup flows for its own dependencies (nam_render
download, local A2 training environment, Kaggle auth, Ollama model pull)
plus a Settings "Getting started" checklist surfacing all of them in one
place.

## What was removed

Commit `d983a29` on `master` (pushed 2026-09-18):

- `desktop/` (pywebview launcher, PyInstaller spec, icons)
- `requirements-desktop.txt`
- `docs/standalone_packaging.md`
- `.github/workflows/build-desktop.yml`
- `tests/test_desktop_main.py`
- The `sys.frozen`/`--run-kaggle-cli` self-relaunch branch in
  `hybrid/kaggle_training.py`'s `KaggleCli._build_argv`
- The `sys.frozen`-gated Python bootstrap branch in
  `hybrid/local_training.py`'s `_bootstrap_python` (replaced with an
  unconditional check: use the app's own interpreter if it already meets
  the training version floor, otherwise fall back to
  `NAM_MIXER_TRAINING_PYTHON`/a system-Python search -- this now also
  benefits the plain server workflow, not just the removed desktop case)
- The dead `pywebview`/`isDesktopApp()` save-file path in `static/app.js`
- Every desktop/pywebview/PyInstaller reference in `README.md` and in-code
  docstrings/comments (this history doc is the intentional exception)

No GitHub releases or tags existed for this repo at the time of removal
(checked via `gh release list` and the Releases API), so there was nothing
to delete there.

## If desktop packaging is ever revisited

Don't reintroduce the self-relaunch-as-CLI pattern for the packaged binary.
A real separate Python interpreter -- the same approach
`hybrid/local_training.py` already uses for the local A2 training
environment -- is a much sounder way to shell out to `kaggle` (or any other
CLI) from a packaged app. Also actually reproduce and confirm the
"repeated instances" bug on a real packaged build before assuming any fix
has addressed it -- that was never done here.
