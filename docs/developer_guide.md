# NAM Mixer developer guide

Running from source, building the renderer, tests, and how the code is laid
out. For what the app does, see the [README](../README.md) and the
[user guide](user_guide.md).

## Contents

- [Run from source](#run-from-source) (macOS, Linux, Windows) · [Optional local AI](#optional-local-ai-tone-recipes)
- [Desktop builds](#desktop-builds)
- [Running the app and the tests](#running-it)
- [Project layout](#project-layout)
- [Current status and limitations](#current-status-and-limitations)
- [Relationship to NAMtoClo](#relationship-to-namtoclo)

## Run from source

Every platform needs **Python 3.10+** plus a working `nam_render` - the
native NAM inference executable. You don't need a C++ compiler to get one:
CI builds `nam_render` for macOS, Linux, and Windows on renderer releases (see
`.github/workflows/build-nam-render.yml`). Version tags such as `v0.2.0` also
build and attach the complete NAM Mixer desktop installers for macOS, Linux,
and Windows (see `.github/workflows/build-desktop.yml`). Downloading a prebuilt binary
is the default path below. Building from source is the fallback, for a
platform/architecture CI doesn't cover or if you'd rather not run a
downloaded binary.

After first-time setup below, `scripts/run.sh` (macOS/Linux) or
`scripts/run.ps1` (Windows) is a one-line way to relaunch later - it creates/
activates `.venv`, installs/updates dependencies only when
`requirements.txt` has changed, opens the browser on the selected local port,
and starts the app; it does not fetch `nam_render` for you.

<details open>
<summary><strong>macOS</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
scripts/download_nam_render.sh
scripts/run.sh
```

`scripts/run.sh` (and `app.py` directly) default to port 5001, not 5000 --
macOS reserves 5000 for AirPlay Receiver, which silently returns 403 instead
of refusing the connection. To use a different port, run
`PORT=5003 scripts/run.sh`.

<details>
<summary>Build from source instead</summary>

Requires Xcode's Command Line Tools (`xcode-select --install`) and
CMake 3.18+. The first build downloads NeuralAmpModelerCore + its
dependencies (a few hundred MB, one-time):

The normal renderer supports both conventional NAM and canonical
Sequential/Linear NAM. If you previously built the older renderer, remove
`native/nam_render/build` first so CMake picks up the new pinned core.

```bash
rm -rf native/nam_render/build  # only needed when replacing an older build
cmake -B native/nam_render/build -S native/nam_render -DCMAKE_BUILD_TYPE=Release
cmake --build native/nam_render/build --config Release --target nam_render -j 4
```

</details>
</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
scripts/download_nam_render.sh
scripts/run.sh
```

<details>
<summary>Build from source instead</summary>

Requires a C++ toolchain and CMake 3.18+, e.g. on Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake
```

Then:

```bash
rm -rf native/nam_render/build  # only needed when replacing an older build
cmake -B native/nam_render/build -S native/nam_render -DCMAKE_BUILD_TYPE=Release
cmake --build native/nam_render/build --config Release --target nam_render -j "$(nproc)"
```

</details>
</details>

<details>
<summary><strong>Windows</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/download_nam_render.ps1
powershell -ExecutionPolicy Bypass -File scripts/run.ps1
```

If `Activate.ps1` is blocked, run PowerShell as: `powershell -ExecutionPolicy Bypass`.

> **Don't double-click `nam_render.exe`.** It's a command-line helper tool
> that `hybrid/core/render.py` calls automatically with the right arguments - it's
> not the app. Double-clicking it in File Explorer runs it with no
> arguments, so it prints a usage error and the console window closes
> instantly, which looks like a crash but isn't one. Always launch the app
> itself via `scripts/run.ps1` (or `python app.py`); that's what finds and
> invokes `nam_render.exe` for you behind the scenes.

<details>
<summary>Build from source instead</summary>

Requires [CMake](https://cmake.org/download/) and the "Desktop development
with C++" workload from the
[Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
(the MSVC toolchain `nam_render` needs), then:

```powershell
Remove-Item -Recurse -Force native/nam_render/build  # only needed when replacing an older build
cmake -B native/nam_render/build -S native/nam_render -DCMAKE_BUILD_TYPE=Release
cmake --build native/nam_render/build --config Release --target nam_render
```

This produces `native/nam_render/build/Release/nam_render.exe`, supporting
both conventional NAM and Sequential/Linear NAM models.

</details>
</details>

Then run `scripts/run.sh` (macOS/Linux) or `scripts/run.ps1` (Windows) - the
launcher opens the browser on the selected local port, and the app discovers
the `nam_render` executable it just downloaded/built automatically, wherever
it landed for your platform. From here:

- **Live preview/design/audition works immediately** - no further setup.
- **Local A2 training** needs a separate, dedicated training environment
  (never the app's own Python environment - see `requirements-training.txt`):
  run `scripts/setup_a2_env.sh` (macOS/Linux) or `scripts/setup_a2_env.ps1`
  (Windows) to create it, then follow the Torch install command it prints
  for your platform/GPU.
- **Kaggle GPU training** needs no local training environment at all - see
  [Setting up Kaggle GPU training](user_guide.md#setting-up-kaggle-gpu-training) in the user guide.

### Optional: local AI tone recipes

The Tone Wizard works without an AI service. The provided `.env` enables its
optional local-AI suggestion with your local Ollama model (`gemma4:e4b` at
`http://127.0.0.1:11434/v1`). Start Ollama, then launch NAM Mixer with:

```bash
scripts/run.sh
```

The launcher reads `.env` automatically. To change machines or models, edit
your untracked `.env`; `.env.example` documents the available defaults.

The base URL must be a loopback HTTP URL (`localhost`, `127.0.0.1`, or `::1`)
and is deliberately read only from the process environment. The browser cannot
choose an endpoint. Only the written tone request and a fixed recipe schema
are sent to the local service-never NAM files or audio. If the model is down,
slow, or returns invalid settings, NAM Mixer uses its built-in recipe rules.

## Desktop builds

Installers are built by `.github/workflows/build-desktop.yml` and published on the project's
[GitHub Releases](https://github.com/daverage/nam-mixer/releases) page for every version tag (`v*`). Each release carries the same five files under fixed names, so the
[README download links](../README.md#download) always point at the newest release.

To get a specific version, open its page under [Releases](https://github.com/daverage/nam-mixer/releases) (for example `https://github.com/daverage/nam-mixer/releases/tag/v0.3.3`) and download from **Assets**.

The desktop release bundles the Flask backend, web UI, training support files,
and the native `nam_render` executable; users do not need to install Python or
the renderer separately. Local A2 training still creates its own training
environment the first time it is used, and Kaggle training still needs Kaggle
authentication.

Releases are currently unsigned, so macOS may require **Open** from the Finder
context menu (or an approval in Privacy & Security) on first launch. If a download link returns
"not found", no release has been published yet: use the browser setup in [Run from source](#run-from-source) or run the
desktop build locally as documented in [`desktop/README.md`](../desktop/README.md).

## Running it

See [Run from source](#run-from-source) above for first-time setup on your
platform. Once dependencies are installed and `nam_render` is built:

```bash
scripts/run.sh
# or: PORT=5001 scripts/run.sh
```

```bash
python3 -m pytest -q
# `pytest -q` works too; pytest.ini resolves the repository modules.
```

The test suite exercises `hybrid/core/envelope.py`, `hybrid/modes/blend.py`,
`hybrid/core/level_match.py`, `hybrid/core/align.py`, `hybrid/core/align_diagnostic.py`, `hybrid/core/align_verification.py`, `hybrid/core/safety.py`,
`hybrid/core/nam_loader.py`, `hybrid/core/input_profiles.py`, `hybrid/core/calibration.py`,
`hybrid/core/coverage.py`, `hybrid/core/pipeline.py`, `hybrid/modes/design.py`,
`hybrid/modes/training_target.py`, `hybrid/modes/fixed_blend.py`,
`hybrid/modes/blend_training_target.py`, `hybrid/modes/character_blend.py`,
`hybrid/modes/character_training_target.py`, `hybrid/core/cab_ir.py`,
`hybrid/core/receptive_field.py`, `hybrid/training/a2_training_settings.py`, and
`hybrid/training/kaggle_training.py` against synthetic signals and mocked renders -
none of it requires torch, `neural-amp-modeler`, or the native `nam_render`
tool to be built. `tests/test_render.py` exercises real NAM inference and is
skipped automatically unless both `native/nam_render` has been built (see its
README) and a real `.nam` file is present at
`assets/nam_models/FenderSuperReverb1977_Clean.nam`.
`tests/test_receptive_field_parity.py` loads the local and Kaggle trainers
side by side and fails the suite if their receptive-field policies ever
diverge, rather than letting that drift go unnoticed.

For repeatable opt-in real-render coverage across all three design modes, use
the explicit harness rather than relying on that personal-model test:

```bash
python3 scripts/real_render_regression.py \
  --amp-a assets/test_cabs/Clean_NoCab_Fender_Deluxe_Reverb_Head_2.nam \
  --amp-b 'assets/test_cabs/HighGain_NoCab_SLASH AFD#2 Head.nam' \
  --di assets/di/high_thrash.wav --di assets/di/clean_smooth.wav \
  --cab 'assets/nam_models/V30 LL 4FB 4x12 SM57 1.00in 0.0in SA73.wav' \
  --out-dir work/real-render-regression --release
```

It writes listenable stems and `real_render_report.json` beneath ignored
`work/`; `--release` fails if prerequisites are missing or a checked case is
invalid/silent. The Phase 4 implementation was exercised with this command in
release mode: 6 cases and 36 listenable artifacts passed, with no invariant or
determinism failures. Calibration combinations remain explicitly unavailable
when both source models lack `input_level_dbu` metadata.

`torch`/`neural-amp-modeler` are no longer in `requirements.txt` - inference
is handled entirely by the native `nam_render` tool now. They only matter for
local A2 training, and live in `requirements-training.txt`
instead (install into a separate, supported-Python-version environment when
you get to that step).

## Project layout

```text
hybrid-nam-builder/
├── app.py                 -- Flask entry point
├── requirements.txt
├── hybrid/                -- core library (no Flask/UI dependencies)
│   ├── paths.py            -- REPO_ROOT (stays at this level: derived from its own location)
│   ├── core/               -- NAM I/O and shared signal processing
│   │   ├── nam_loader.py       -- parse .nam files + calibration metadata
│   │   ├── render.py           -- NAM inference (shells out to native/nam_render)
│   │   ├── render_bootstrap.py -- in-app "download nam_render" for Settings
│   │   ├── envelope.py         -- dry-input level/envelope extraction
│   │   ├── level_match.py      -- crossover-region auto level-match trim
│   │   ├── align.py            -- offset measurement + apply_fixed_offset (frozen integer, off by default)
│   │   ├── align_diagnostic.py -- read-only multi-region A/B timing diagnostic
│   │   ├── align_verification.py -- cross-DI verification of a fixed A/B offset
│   │   ├── cab_ir.py           -- shared cabinet IR convolution (preview + baked target)
│   │   ├── receptive_field.py  -- mode/cab-aware temporal-dependency accounting
│   │   ├── safety.py           -- NaN/clip checks, non-limiting peak ceiling
│   │   └── pipeline.py, calibration.py, input_profiles.py, coverage.py, audio_metrics.py
│   ├── modes/              -- the three design modes
│   │   ├── blend.py, design.py, training_target.py -- Dynamic Hybrid (+ metadata.py, wizard.py)
│   │   ├── fixed_blend.py, blend_training_target.py -- Parallel Blend
│   │   └── character_analysis.py, character_blend.py, character_training_target.py -- Character Blend
│   ├── continuous_gain/    -- Continuous Gain: probe, audit, profile, selection, anchors, bundle, project, validation, excitation, multi_blend
│   ├── training/           -- A2 settings, local/Kaggle backends, validation + reports, nam_tools, provenance, embedded-cab export
│   └── services/           -- Settings registry and .env, local recipe assistant, research, update check, NAM Inspector
├── routes/                 -- Flask route modules registered by app.py
│   └── continuous_gain.py  -- /api/cg/* routes for the Continuous Gain tab (static/cg.js)
├── native/nam_render/      -- C++ NAM inference tool (NeuralAmpModelerCore), see its README
├── assets/nam_models/      -- user's own .nam amp captures (gitignored)
├── assets/di/              -- genre/style DI library + its own README
├── templates/, static/     -- minimal HTML/CSS/JS UI (no build step)
├── tests/                  -- unit tests for the hybrid/ modules
├── scripts/                -- run/setup/download helpers, train_a2.py, validate_a2.py, analyze_di.py (regenerates assets/di/_analysis.json)
└── work/                   -- gitignored scratch output directory
```

## Current status and limitations

### Implemented

- **Real NAM inference:** `hybrid/core/render.py` calls the native `nam_render`
  C++ CLI in `native/nam_render/`, built against
  [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).
  This keeps inference independent of Python `torch`/`neural-amp-modeler` and
  lets NAMCore remain authoritative for `.nam` model loading. Build steps are
  in `native/nam_render/README.md`.
- **`.nam` loading and calibration:** `hybrid/core/nam_loader.py` reads model
  metadata, including `input_level_dbu` and `output_level_dbu` when present.
  Older or uncalibrated files remain usable and are reported as having
  unavailable calibration metadata.
- **NAM Tools:** Output volume and Metadata create validated new copies;
  Cab Embed supports learned A2 and exact experimental Sequential exports;
  NAM Inspector provides read-only structural details and a short NAMCore
  render check, including Full/Lite checks for packed A2 models.
- **The complete app workflow:** the Flask app and browser UI connect real
  rendering, preview, coverage analysis, all three design modes, cabinet IR,
  Sessions, target generation, local training, and optional private Kaggle
  training. The main routes include `/api/render_pair`, `/api/preview`,
  `/api/blend_info`, `/api/blend_curve`, `/api/input_profiles`,
  `/api/profile_coverage`, and `/api/generate`.
- **Renderer readiness:** the app performs a cheap `--help` launch check before
  preview inference, distinguishes missing from unusable executables, shows
  expandable setup help, and can retry after installation without a restart.
- **Mechanical safeguards:** envelope following, blend math, level matching,
  optional alignment, receptive-field checks, NaN/Inf and silence checks, and
  non-limiting training-target peak control are implemented and covered by
  automated tests against synthetic signals.

### Important validation boundaries

- The Phase 4 real-render harness passed with two distinct local source models,
  two musical DIs, all three design modes, cabinet on/off, and input levels of
  -12/0/+12 dB. It produced deterministic rerenders and exact frozen-teacher
  equivalence for that run. Auto-calibrated cases were unavailable because
  those source files did not declare input-level metadata. Most automated
  pipeline tests still use mocked renders. Other real source-model
  combinations can still expose latency, calibration, or musical problems;
  listen to every preview and validate every exported model against its target.
- **A/B timing correction is optional and off by default.**
  Different amps naturally delay different frequencies differently; that is
  part of their sound and is never "corrected". The **Timing** readout
  measures the A/B offset in several regions of the DI
  (`hybrid/core/align_diagnostic.py`), then re-checks it on three other DIs
  (`hybrid/core/align_verification.py`). Only when all of them agree on one
  integer does it offer **Original / Corrected**. Corrected shifts Amp B by
  that exact integer in preview, and the same integer is frozen into the
  design and applied verbatim to the training target and validation, never
  re-measured. No natural capture pair tested so far has been offered a
  correction; see docs/alignment_diagnostic.md.
- Character Blend is a deterministic teacher design, not a perceptual-match
  guarantee. No automated system judges tone, feel, or musical quality; the
  checks can only catch mechanical problems such as clipping, discontinuities,
  silence, or non-finite samples.
- Local A2 training needs the separate environment described in
  `requirements-training.txt`. Kaggle training needs a configured Kaggle
  account and network access; neither is required to run the preview UI.
- **Continuous Gain** reproduces its frozen reference configurations
  bit-for-bit and has been exercised end to end with local and Kaggle GPU
  training in the browser - see
  [`docs/continuous_gain_tab.md`](continuous_gain_tab.md). Training
  measurements are technical checks, not a substitute for listening to the
  exported model with your own captures and playing setup.
- Uploaded amp/cab files and internal render-source copies are only freed
  when the session(s) that used them are deleted (a startup sweep also clears
  render copies that never became a saved session) - see `app.py`'s
  `_sweep_orphaned_uploads`/`_sweep_orphaned_render_sources`.

## Relationship to NAMtoClo

This project reuses the genre/style DI WAV files bundled in the
[NAMtoClo](https://github.com/Goaltoday/NamtoClo) fork this repository's author
also maintains, purely as audio fixtures (see `assets/di/README.md`). It shares
no code with NAMtoClo and solves a completely different problem - NAMtoClo
converts a single existing NAM model to Valeton hardware's CLO format; this
project builds new hybrid NAM training material out of two existing NAM
models. Nothing here depends on NAMtoClo at runtime.
