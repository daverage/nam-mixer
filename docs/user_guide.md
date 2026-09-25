# NAM Mixer user guide

Everything NAM Mixer can do, in detail. New here? Start with the
[README](../README.md) for the short version.

## Contents

- [How NAM Mixer works](#how-nam-mixer-works)
  - [This is NOT model-weight merging](#this-is-not-model-weight-merging)
  - [Why the dry input's level controls the transition](#why-the-dry-inputs-level-controls-the-transition)
  - [Why automatic level matching is needed](#why-automatic-level-matching-is-needed)
  - [A/B timing](#ab-timing-fixed-offsets-phase-response-and-why-nothing-is-auto-aligned)
  - [Input profile vs. crossover vs. NAM calibration](#input-profile-vs-crossover-vs-nam-calibration--three-separate-knobs)
  - [Preview DIs vs. NAM training material](#preview-dis-vs-nam-training-material--an-important-distinction)
- [The Builder, step by step](#the-builder-step-by-step) · [Kaggle GPU setup](#setting-up-kaggle-gpu-training)
- [Design modes and the shared Cabinet stage](#design-modes-and-the-shared-cabinet-stage)
- [Continuous Gain](#continuous-gain-one-amp-one-nam)
- [Sessions](#sessions)
- [NAM Tools](#nam-tools-volume-metadata-cabinet-embedding-and-inspection)
- [Settings](#settings)
- [Safety: training target vs. live preview](#safety-training-target-vs-live-preview)

## How NAM Mixer works

### This is NOT model-weight merging

We are **not** averaging, interpolating, or otherwise combining the weights of
two NAM neural networks. That's a different (and much harder, largely unsolved)
problem — two independently-trained networks don't share a latent space, so
blending their weights doesn't produce anything musically meaningful.

Instead, the approach is:

1. **Render** the same input audio through Amp A and through Amp B separately
   (two ordinary, unmodified NAM inference passes).
2. **Blend the resulting audio**, sample-by-sample, using a crossfade curve
   driven by the *level of the original dry input* — quiet input leans toward
   Amp A's render, loud input leans toward Amp B's render, with a smooth
   transition in between.
3. That blended audio becomes a **synthetic training target**.
4. **Eventually**, train a fresh NAM A2 model against that synthetic
   input/output pair, producing one model that reproduces the hybrid behavior
   directly — at that point the two source `.nam` files are no longer needed at
   inference time.

Step 4 is available through the UI for either Kaggle GPU or local training.
Every generated model still needs listening and validation; the tool does not
make a claim of perceptual equivalence merely because training completes.

### Why the *dry input's* level controls the transition

The crossover has to be driven by something that exists independently of which
amp is currently being auditioned — otherwise there's a chicken-and-egg problem
(you'd need to know the blend to compute the blend). The dry guitar signal
before either amp model touches it is the one signal that's the same regardless
of which amp(s) you render it through, so it's the natural, stable control
signal for "how hard is the player driving this right now."

Concretely: `level = envelope(dry_input)`, and that per-sample level (in dBFS)
is what decides the Amp A/Amp B mix weight, not the loudness of either
processed output. See `hybrid/core/envelope.py`.

### Why automatic level matching is needed

Two independently-trained/captured NAM models will almost never happen to sit
at the same loudness for a given input level. If Amp B is simply louder than
Amp A at the crossover point, the transition will sound like a volume jump
rather than a change in amp character — exactly the artifact we're trying to
avoid. Rather than normalizing the two amps' *overall* loudness (which doesn't
guarantee anything about how they compare specifically **at the crossover
point**, where it actually matters), this project measures each amp's loudness
using only the portion of the render that falls near the chosen crossover level
and computes a trim from that. See `hybrid/core/level_match.py`.

### A/B timing: fixed offsets, phase response, and why nothing is auto-aligned

When two amp captures are mixed (Parallel Blend) or crossfaded (Dynamic
Hybrid), their timing matters. Two very different things can make Amp B
look "late" compared with Amp A:

- **A fixed timing offset.** The whole capture is delayed by the same number
  of samples, e.g. from a latency setting when it was captured. It is the
  same on every note and every piece of material, and shifting Amp B by that
  exact amount fixes it completely.
- **Phase response.** Real amps delay low, mid and high frequencies by
  different amounts. That is part of how an amp sounds, not a mistake, and
  it changes with what you play. No single shift can "fix" it; forcing one
  just trades one comb-filtered frequency region for another.

**NAM Mixer does NOT blindly phase-align captures.** After **Render Amps**
it shows a **Timing** result. It measures the timing on several separate
note attacks, and when those agree it re-checks the same two amps on other,
independent DIs. A correction is offered only when every check agrees on one
repeatable fixed offset. That is what a real latency looks like, and phase
response does not pass it. The possible results are:

- **Fixed offset detected — Amp B lags/leads Amp A by N samples:** you get an
  optional **Original / Corrected** choice.
- **No stable fixed offset detected:** the amps are already lined up.
- **No trustworthy fixed timing correction identified:** the difference
  looks like phase response or is inconsistent, so nothing is offered.

There is **no automatic correction**. Original is always the default;
Corrected is only ever your explicit choice. If you choose it, that exact
sample count is frozen into the design and reused unchanged for preview,
the training target, validation and export, and it is never re-measured.
A saved session remembers the choice, but restores Corrected only after the
next render verifies the same offset again. The technical details are in
[docs/alignment_diagnostic.md](alignment_diagnostic.md).

### Input profile vs. crossover vs. NAM calibration — three separate knobs

It's easy to conflate these; they operate at different stages and have very
different costs:

- **Input profile** (`hybrid/core/input_profiles.py`, `render_pair()`): simulates a
  different instrument/pickup driving the signal chain *before* it reaches
  either NAM. This changes the actual audio both amps receive, so changing it
  is EXPENSIVE — it requires re-running NAM inference for both amps. See
  `docs/history/INPUT_PROFILE_RESEARCH.md` for the research behind the presets and
  why active pickups deliberately have no fixed preset.
- **Crossover / transition width / manual trim** (`build_hybrid()`): changes
  only how the *already-rendered* Amp A/B responses are blended together.
  This is CHEAP — pure numpy, no NAM inference, safe to recompute on every
  slider move.
- **NAM input calibration** (`hybrid/core/calibration.py`): when both `.nam`
  captures report their own recording calibration (`input_level_dbu`),
  applies the official NAM plugin's per-model compensation formula so two
  differently-calibrated captures see the same virtual physical input level.
  This also happens at render time (it's folded into `render_pair()`
  alongside the input profile), but it's a distinct concept from "how hard is
  the (virtual) instrument driving the amps" — one is about the instrument,
  the other is about reconciling two amp captures' own assumptions about
  their input level.

### Why the genre/style DI files are included

`assets/di/` contains real recorded DI (direct input) guitar/bass performances,
copied from the [NAMtoClo](https://github.com/Goaltoday/NamtoClo) project (see
`assets/di/README.md` for exactly which files, where they came from, and their
measured characteristics). They exist so you can **audition** a hybrid
crossover on real playing dynamics — palm-muted riffing, clean chords, dynamic
picking, etc. — without needing your own guitar/interface set up, and so the
project has a consistent, versioned set of fixtures for regression testing the
blend engine itself.

### Preview DIs vs. NAM training material — an important distinction

```text
genre/style DI (assets/di/*.wav)
    ↓
preview, auditioning, crossover analysis,
automatic level-match testing, regression testing

official NAM v3.0.0 training input (a proper calibrated reamp/DI signal, e.g.
the kind of stimulus NAMtoClo itself uses as nam_input_wav.wav — bundled at
assets/training/, seeded automatically on first run; see that folder's README)
    ↓
Amp A render
Amp B render
    ↓
dynamic blend
    ↓
synthetic training output
    ↓
A2 training
```

The genre DIs are **musical performances** — useful for judging how a hybrid
*sounds* and behaves, but not designed to exercise the amp's full frequency/
level response the way a proper reamping signal is. The **actual** synthetic
training pair used to eventually train an A2 model must come from a real NAM
training/reamping signal, not from `assets/di/`. This distinction is
deliberate and should not be blurred — see `assets/di/README.md` for more.

## The Builder, step by step

The Builder walks you through five steps along the top of the page. You can
jump between them at any time; the amps, test performance, cabinet, output
safety and training tools are shared by all three design modes.

1. **Choose amps** — pick your Amp A and Amp B `.nam` files, a test
   performance (preview DI), your instrument and pickup (input profile), and,
   if needed, advanced NAM input calibration or per-amp input trims. **Prepare
   amps for comparison** runs the two NAM renders; this is the only slow step.
2. **Compare amps** — listen to Amp A, Amp B and the current result on the
   chosen test performance. **Test gain** pushes the level up or down (it
   re-renders after you stop dragging) so you can check the change-over beyond
   what that clip's own dynamics reach. The input profile only shapes this
   preview; see [Input profile vs. crossover vs. NAM calibration](#input-profile-vs-crossover-vs-nam-calibration--three-separate-knobs).
3. **Shape the sound** — choose a design mode (Dynamic Hybrid crossover and
   transition, Parallel Blend mix, or Character Blend tone/feel/drive), keep
   the level steady with level matching, and check the **Timing** result.
   These controls recombine the prepared amps instantly; nothing is
   re-rendered.
4. **Finish & polish** — optional finishing: a cabinet IR, output gain, and
   (Dynamic Hybrid) an on-demand view of how the change-over behaves.
5. **Make a model** — the official NAM training input is bundled and ready
   (upload a different one only if you want to). **Create training files**
   freezes the current design into an immutable `HybridDesign`
   (`hybrid/modes/design.py`) and plays the *official* training input (never
   the preview DI, and never with the input-profile gain) through Amp A/Amp B
   with that frozen design (`hybrid/modes/training_target.py`). It writes
   `input.wav`, `hybrid_target.wav` (+ `hybrid_target_raw.wav` for
   comparison), the design JSON and `training_manifest.json` under
   `work/a2/<design_id>/`. Then train a real A2 (PackedWaveNet) model on it:
   - **Kaggle GPU** (recommended — no local Torch install, trains on a free
     private Kaggle T4) — see [Setting up Kaggle GPU training](#setting-up-kaggle-gpu-training)
     below, or
   - **Local**, with `scripts/train_a2.py` (needs a separate Torch/
     `neural-amp-modeler` environment — see `requirements-training.txt` /
     `scripts/setup_a2_env.sh` (macOS/Linux) / `scripts/setup_a2_env.ps1`
     (Windows) — never the app's own Python environment).

   Both paths validate the exported `.nam` identically: loading and
   rendering its Full and Lite branches through the native NAMCore renderer,
   recording raw/gain-normalized ESR, and checking quiet response against an
   equivalent processed reference when one is available. Training completion
   and technical validation quality are reported separately.

### Setting up Kaggle GPU training

Kaggle GPU is optional — the preview/design UI and local training work fully
without it. To enable the one-click "Train A2" button on the "Create A2"
card:

1. **Have (or create) a free [Kaggle](https://www.kaggle.com) account.**
2. **Verify your phone number** on
   [kaggle.com/settings](https://www.kaggle.com/settings) (Account tab) —
   Kaggle requires this before it will grant GPU/TPU accelerator quota to any
   account, regardless of how you submit the job. This is the step people
   most often miss.
3. **Install and authenticate the Kaggle CLI.** The Create A2 card offers an
   explicit **Install Kaggle CLI** button when it is missing; alternatively,
   install it once in your normal shell (the app never imports Kaggle's
   library; it invokes the CLI/module only for cloud operations):
   ```bash
   pip install kaggle
   kaggle auth login
   ```
   This opens a browser to sign in and stores a credential locally
   (`~/.kaggle/kaggle.json` or `KAGGLE_API_TOKEN`) that the Kaggle CLI
   manages entirely on its own — NAM Mixer never reads, stores, or logs it.
4. Reload the "Create A2" card in the app; it should show **"Connected ✓"**
   and enable the **Train A2 (Kaggle)** button.

Each training run stages a private, uniquely-named Kaggle dataset + kernel
under your account, polls it without blocking the app, downloads the result,
and re-validates it locally before calling the job complete — see
the in-app setup checklist for the mechanics, troubleshooting, and how to
recover a job that trained but failed to download. Kaggle's free T4 quota is weekly and account-wide; the "Create A2"
card shows your remaining quota before you submit.

## Design modes and the shared Cabinet stage

The workflow above describes **Dynamic Hybrid** mode, the original/default
mode. Two further modes are available from the same mode selector:

- **Dynamic Hybrid** (`hybrid/modes/blend.py`, `hybrid/modes/design.py`,
  `hybrid/modes/training_target.py`): changes from Amp A toward Amp B according to
  playing level, via the crossover/transition envelope described above.
- **Parallel Blend** (`hybrid/modes/fixed_blend.py`, `hybrid/modes/blend_training_target.py`):
  always combines the two amp responses at one constant, user-chosen ratio
  (`result = A * (1 - mix_b) + B * mix_b`), independent of playing level —
  no crossover envelope at all. Its own auto level-match uses the DI's
  ACTIVE playing material (silence excluded) rather than a crossover band,
  since there's no crossover region to match around (see
  `hybrid.modes.fixed_blend.compute_active_trim`). After the amps are prepared,
  **Parallel compatibility** measures the actual weighted sum across active
  frequency bands. It reports whether both amps have material level and whether
  their sum shows strong repeatable cancellation on the selected performance.
  If reversing Amp B's polarity clearly removes a measured cancellation, the
  panel recommends it and lets you play **Original polarity** and **Amp B
  flipped** directly through the shared **Compare the sound** player. The
  panel then offers a check across two more performances. Nothing is changed
  automatically; the selected choice is frozen into preview, training,
  validation, and export.
- **Character Blend** (`hybrid/modes/character_blend.py`): uses a continuous,
  residual-bounded nonlinear carrier plus measured EQ and compression
  corrections to produce a deterministic teacher design. Tone, Feel, and
  Drive are not a simple parallel waveform mix. Drive preserves exact Amp A/B
  carriers outside a 35-65% soft region and can optionally vary by input level.

All modes share Amp A/Amp B, the preview DI, input profile/calibration,
render, test gain, the Listen controls, the Cabinet IR stage, the official
training input, A2 quality, and training — switching tabs never re-runs NAM
inference; the already-rendered `RenderedPair` (`hybrid/core/pipeline.py`) is
reused by whichever mode you're auditioning.

A third, mode-independent stage — **Cabinet IR** (`hybrid/core/cab_ir.py`) — sits
AFTER the amp combination in any mode, and in the Continuous Gain workflow.
Preview is independent of training. When an IR is selected, the choices are:

- **Amp only** (`none`): train/export an amp-only A2. The IR is only for
  previewing; load it in your player for the cabinet (exact, and swappable).
- **Learned cab** (`learned`, the supported way to put a cab in a NAM): the
  training material is played through the amps and then the cabinet IR, like
  a NAM player feeding an IR loader, and the result is the training target.
  The cabinet is fixed into the trained NAM, which makes it a full-rig capture
  (`[Learned Cab]`, gear type `amp_cab`). One NAM is trained, tested with the
  cabinet on both sides, and downloaded. It is an approximation: see
  "Baked cabinet" below.
- **Create both** (`embedded`, experimental): retain the tested amp-only A2
  and derive a second NAM
  containing an explicitly extracted Full WaveNet followed by canonical Linear
  FIR taps in a NAM **Sequential** model, offered as a separate
  `-with-cab.nam` download. This option is available when *Settings → Advanced
  → Enable experimental NAM architectures* is on; with that setting off it is
  hidden and the server refuses it.

The cabinet output folds the recorded post-cab safety scalar into the Linear
weights and is downloadable only after validation using the bundled,
Sequential-capable renderer. **NAM format validity and NAM A2 compatibility
are two separate claims**: A2-only players may reject a `Sequential` model.
Use the separately provided head-only download for broad compatibility. The
setting is enforced server-side as well as in the UI, so a client cannot
request the experimental export while it is disabled.

**Receptive-field policy: one hard check, two advisory ones.** Amp A/Amp B
(+, for Hybrid and Character, the bounded crossover envelope) are the CORE
dependency — this must fit inside the destination A2's actual receptive
field, or generation/training is refused exactly as before. Two further
dependencies are always calculated and reported honestly, but never block
training by themselves, because in both cases the A2 is being trained to
*approximate* the rendered teacher rather than to compile its signal graph
exactly:

- **Character processing.** Character Blend's donor transitions, smoothing,
  and correction filters add their own formal temporal dependency
  (`formal_character_required_samples`). If it exceeds the A2's receptive
  field, training still proceeds as an approximation — the Full/Lite export
  and quiet-response checks stay authoritative for judging the result.
- **Baked cabinet.** A baked cabinet adds `len(ir) - 1` samples of *serial*
  temporal dependency on top of the core (see `hybrid/core/receptive_field.py`'s
  `combine_required_history`). A baked cab whose formal total exceeds the
  A2's receptive field does not block training either — the A2 learns an
  approximation of the post-cab response within its available capacity.

Either case prints an explicit "CABINET APPROXIMATION" or "CHARACTER
APPROXIMATION" notice from `scripts/train_a2.py`/the Kaggle cloud worker —
validate the result by listening and by checking the printed ESR/RMS
metrics against the baked target. The exact same full-length cabinet IR is
used for preview and for baking in either case; only the training-time
gating differs. Local and Kaggle training implement this policy
independently but are kept from silently diverging by
`tests/test_receptive_field_parity.py`.

Because raw WAV/FIR length is a poor proxy for how much of a captured IR is
actually audible signal, the Cabinet card also reports cumulative-energy
diagnostics (e.g. "99.9% energy by: 42.7 ms" for a nominally-500ms IR) —
purely informational, never used to shorten the actual convolution.

## Continuous Gain: one amp, one NAM

The **Continuous Gain** tab is separate from the Builder's two-amp modes. You give it several fixed-gain captures of **one amp and
channel** (for example Gain 1 … 10), and it builds **one standard `.nam`** whose whole gain range you explore with an ordinary NAM
player's **Input gain** — no model switching, no extra runtime processing. It *approximates* the amp's range; it does not reproduce every
knob position exactly.

It follows the Builder's guided flow — four stages, one centred column of cards:

1. **Add captures** — drag in the `.nam` files and confirm each one's physical gain position (file names only *suggest* it). Missing or
   duplicate positions, wrong sample rates and too-few captures are flagged before anything runs.
2. **Analyse & select** — every capture is probed, audited (VALID / CORRECTED / SUSPECT; nothing uncertain is silently fixed) and
   profiled; a training subset is chosen from those **measurements** (Automatic), or you use all captures or pick your own (Custom). You
   see why each capture was chosen or omitted, how well the omitted ones are reproduced, the measured response, and the Input-gain
   mapping that will be used. If no subset meets the coverage rule it says so and falls back to all eligible captures.
3. **Train** — training files are created and trained with the **same Kaggle / local training section as the Builder** (same setup,
   presets, progress and logs). The default recipe is the validated one: official NAM input plus real guitar DIs at level offsets,
   response-distance anchors on −20…+14 dB, one peak-ceiling output scale (shown as the Output gain to set in your player).
   Fixed 4 dB anchors are available under *Advanced* as an explicit alternative.
4. **Test & export** — automated checks (standard NAM structure, Full/Lite renders, output safety, per-position measured comparison with
   your real captures on held-out DIs, direction reversals), an optional Input-gain sweep and A/B clips against the originals, and an
   export package (`.nam`, JSON metadata, player guide). Export is never blocked by validation or by listening.

Projects appear in **Sessions** labelled *Continuous Gain* (Sessions → **Load** opens this tab on that project; **Delete** removes its
files). Analysis parallelises the native renders and caches probes per capture, so re-analysing after adding a capture is quick.

What to expect: a 60-epoch model takes roughly half an hour to train; validation figures are *measurements*, not a listening result —
always listen. The tab's design, the training-material experiments and the frozen configurations it reproduces are documented in
[`docs/continuous_gain_tab.md`](continuous_gain_tab.md).

## Sessions

The **Sessions** tab is a project library, not a popup. Use it to save the
current controls under a name, load or inspect an earlier design, export a
portable NAM Mixer JSON file, import one, or delete a session. Generated
training bundles are also saved as sessions automatically so their completed
NAM can be downloaded again or opened in **Tools**. Validation reports are
bound to the completed NAM's SHA-256; editing the NAM invalidates that report.
A completed bundle also offers a synchronized Teacher/Full/Lite comparison on
the selected musical DI at normal or quiet input level. It reconstructs the
teacher from the saved design, never from current controls.

A session restores the selected settings and app-managed NAM/cabinet file
references, but it deliberately does not render automatically. After loading,
use **Render Amps** to rebuild the pair and verify that the referenced files
are still available. A saved Corrected timing choice comes back only if that
render verifies exactly the same fixed offset again; otherwise timing stays
Original and the Timing result says why. Training manifests under `work/a2` are separate from
sessions and are not interchangeable with session JSON files.

## NAM Tools: volume, metadata, cabinet embedding, and inspection

The **Tools** tab has four tools for an existing NAM capture: **Output
volume**, **Metadata**, **Cab Embed**, and **NAM Inspector**. Choose a `.nam`
file in Tools to open it automatically, or select **Use latest generated NAM**
to open the most recent locally generated model. The selected NAM is the
read-only source for each operation: Output volume and Metadata create a new
downloadable copy, Cab Embed creates a new learned or Sequential result, and
NAM Inspector only reports on the file. No tool overwrites the source.

**NAM Inspector** reports the architecture, declared sample rate, calibration,
available model metadata, and formal receptive-field information. It performs
a short deterministic NAMCore render and, for packed A2 models, checks both
Full and Lite branches. Inspection is read-only: no NAM fields or weights are
changed. Cabinet reporting is evidence-based: a Sequential Linear stage is
reported as an embedded cabinet stage, explicit `amp_cab`/`amp_pedal_cab` metadata is reported
as metadata evidence, and a plain model is not declared to contain a cabinet
based on its sound. Missing or ambiguous evidence is shown as unknown.
The technical view also distinguishes neural receptive field from any
Sequential FIR history and reports total formal dependency only when its
stages expose enough information to calculate it. It does not score tone
quality or claim that a cabinet exists because a response sounds filtered.

### Output volume

The output-volume tool changes the recognised final audio `head_scale` by
`10 ** (dB / 20)`. `head_scale` controls output after the model has generated
its signal, so this changes output level without retraining or changing the
learned tone, distortion, dynamics, or input response. It also updates each
available `metadata.loudness` by the same dB amount. It deliberately does not
change `metadata.gain`, which describes separate metadata/calibration intent.

Modern A2 `SlimmableContainer` NAMs have one final audio model per submodel;
the tool edits only `config.submodels[*].model.config.head_scale`. Older
single-model files are supported only when their root `config.head_scale` is
present. Unknown layouts are refused rather than guessed. Before saving, a
recursive JSON diff must match exactly the approved output-scale and loudness
paths, so weights and every other model field stay unchanged.

The slider starts at the model's measured loudness and sets the desired final
level. For example, moving from -20 dB to -17 dB applies a +3 dB change. A +6
dB change uses a multiplier of approximately `1.995262`; -6 dB uses
`0.501187`. Boosts above +12 dB are allowed but may clip in a host or target
hardware. The volume tool is unavailable when the capture has no measured
loudness or no recognised output scale.

The same safe operation is available from a terminal:

```sh
python3 scripts/nam_volume.py Mesa_Boogie.nam +3
python3 scripts/nam_volume.py Mesa_Boogie.nam +6 --dry-run
python3 scripts/nam_volume.py Mesa_Boogie.nam -6 --output Mesa_Boogie_quieter.nam
```

### Metadata

The metadata editor changes descriptive fields only: **name**, **modeled by**,
**gear make**, **gear model**, and **tone type**. It does not edit weights or
technical facts such as gear type and calibration. Export date, trainer
details, and measured loudness remain untouched; use Output volume to change
loudness. Existing values are preserved unless explicitly edited or cleared.
Gear make/model and tone type are optional; enter them only when you know the
values are accurate. Models NAM Mixer trains name `NAM Mixer` as their creator
(**modeled by**).

### Cab Embed

**Cab Embed** adds a selected cabinet IR to one existing NAM capture. The
default **Learn into a new A2 NAM** path renders the official NAM training
input through the source capture and frozen cabinet IR, then hands the normal
bundle to the same local or Kaggle trainer used by the Builder. It preserves
the source capture's input calibration and tone type, records both source and
IR hashes, and exports an `amp_cab` model. For supported A2
`SlimmableContainer` sources, **Exact Sequential embed** is also available
when experimental architectures are enabled. It appends the prepared IR as a
Linear stage without retraining and is immediate. This experimental
Sequential export is validated through NAMCore, but players that support only
A2 may not load it. Your source NAM is unchanged and remains the head-only
model.

## Settings

The Settings tab (next to Sessions) covers anything that used to only be
configurable via a shell environment variable or an `.env` file. Its
**Updates** section has a **Check for updates** button. NAM Mixer also
checks once when it starts; if a newer version exists, a notice offers
**Download** (the installer for your computer: the Apple Silicon DMG, the
Windows installer, or the Linux AppImage or `.deb`, matching how you
installed it) or **Not now**. Running from source, it links to the release
notes with a `git pull` hint instead. The check is an anonymous read of the
public GitHub releases list; nothing about you or your files is sent. Tick
**Advanced → Don't check for updates when NAM Mixer starts** to check only
when you click the button. Its **Getting started** section gives an at-a-glance
checklist of what's ready and what's still optional (renderer, training
input, local A2 training environment, Kaggle, AI provider) so a fresh
checkout doesn't require hunting for each setup button individually.

- **NAM render executable** — normally auto-detected (downloaded via
  `scripts/download_nam_render.*`, or the in-app **Download nam_render
  automatically** button). Change it here only if you built/downloaded a
  custom `nam_render`.
- **AI Assistant** — choose **Local**, **Cloudflare Workers AI**, or **Custom
  OpenAI-compatible**. Local uses a localhost `/v1` endpoint and needs only a
  model name. Custom remote endpoints must use HTTPS; their hostname is
  resolved and private/loopback/link-local/reserved addresses are rejected.
  Conversation Markdown exports can optionally include a structured debug
  appendix containing the exact bounded model messages, research queries and
  results, warnings, and parsed responses. API keys and authorization headers
  are never included.
- **Cloudflare Workers AI** — in the Cloudflare dashboard, open **Workers AI →
  Use REST API → Create a Workers AI API Token**, then copy the token and the
  account ID. In NAM Mixer choose Cloudflare, enter the 32-character Account
  ID, the token, and a JSON-mode model such as
  `@cf/meta/llama-3.3-70b-instruct-fp8-fast`; the app builds
  `https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1` and sends
  the request to `/chat/completions` for you. A manually created token needs
  account-scoped **Workers AI Read** and **Workers AI Edit** permissions (the
  dashboard's “Create a Workers AI API Token” template is the easiest route).
  Cloudflare quota, capacity limits, and billing may
  apply. **Test connection** performs a small real JSON-mode inference request,
  so it may consume quota.
- **Automation boundary** — NAM Mixer automates endpoint construction,
  authentication headers, JSON-mode requests, connection testing, and safe
  error handling. It deliberately does not create Cloudflare accounts/tokens,
  store tokens in browser code, or silently rotate/reuse tokens when switching
  providers. Use the explicit **Clear API token** action to remove a saved
  token from `.env` and the running process.
- **Local AI assistant** — any host that speaks the OpenAI-compatible `/v1` API
  works (Ollama, LM Studio, llama.cpp server, ...); Google's Gemma
  (`gemma4:e4b`) is a good default if you don't already run something
  else, and a **Pull via Ollama** button offers a one-click download when
  Ollama is installed but nothing's running yet.
- **TONE3000 API key** — enables the TONE3000 tab's capture search. Get a
  key from your account at [tone3000.com](https://www.tone3000.com); saved
  keys are never echoed back by the app once entered.
- **Advanced → Enable experimental NAM architectures** — off by default.
  Turning it on reveals **Create both**, which exports the tested amp-only
  NAM plus an exact Sequential embedded-cabinet derivative. Sequential NAMs
  are supported by NAM Mixer’s renderer, but some players accept only A2;
  use the separate head-only download for broad compatibility. With the
  setting off, the Sequential option is hidden and cannot be requested.
  **Learned cab** remains the broadly compatible way to train a cabinet into
  a new A2 NAM.
Settings are saved to the source checkout's own `.env` file — never uploaded
anywhere.

## Safety: training target vs. live preview

The generated hybrid **training target must never be run through a limiter** —
that would distort the very dynamic behavior we're trying to capture. If the
generated hybrid exceeds a target peak ceiling (default -3 dBFS), a single
fixed gain reduction is applied to the whole file instead
(`hybrid.core.safety.apply_peak_ceiling`). A limiter (`hybrid.core.safety.
preview_safety_limiter`) exists only as a speaker/headphone safety net on the
live preview/playback path and must never touch a file destined to become (or
derive) a training target.
