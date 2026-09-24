# Continuous Gain tab (integration of the frozen FC recipe)

One amp/channel, N fixed-gain captures -> ONE standard `.nam` driven by the player's Input gain. Code: `hybrid/continuous_gain/*.py`
(pure library), `routes/continuous_gain.py` (`/api/cg/*`), `static/cg.js`. Training reuses the existing local and Kaggle trainers.

## Workflow
1. **Add captures** - upload `.nam` files, confirm each physical gain position (file-name parsing is only a suggestion).
2. **Analyse & select** - `cg_probe` (probe bank) -> `cg_audit` (Phase 4A: VALID/CORRECTED/SUSPECT) -> `cg_profile` (4B) ->
   `cg_selection` (4D exhaustive subset search, no imposed count, `k*` coverage rule). Modes: **Automatic** (the `k*` set; if no
   subset meets the coverage rule it says so and falls back to all eligible captures, never an invented optimum), **Use all**,
   **Custom**. Two views: measured source response, and the Input-gain mapping used to build the target.
3. **Train** - `cg_project.generate_bundle` writes an ordinary A2 bundle (`mode: "continuous_gain"`) into `work/a2/<id>/`. The tab
   does not re-implement training: it hosts the Builder's own training section (Kaggle connection/install, local environment setup,
   epoch presets, progress, logs, downloads) via `window.namTrainingHost` in `static/app.js`, pointed at this design, and reloads on the
   `nam:training-complete` event.
4. **Test & export** - `cg_validation`: standard-NAM compatibility (Full/Lite), output safety, per-position comparison with the real
   captures on held-out DIs (training anchors and omitted references), direction reversals, optional sweep/comparison audio.
   Export (`.nam` + JSON + player guide) is never gated on validation or listening.

## Production default = the frozen FC recipe
Selection: Phase 4D `k*` set. Anchors (`cg_anchors.response_anchors`): arc length of the measured profile mapped onto
[-20, +14] dB, >= 4 dB apart, rounded to 0.1 dB. Target (`cg_bundle`): level-driven blend of the real captures
(`multi_blend`), official input + DIs at -32..+20 dB offsets, one -0.2 dBFS peak-ceiling gain (`output_scale_c`), verified 4A timing
corrections only. Fixed 4 dB v3 anchors exist only as an explicit Advanced option.

## Custom-input bundles and the trainers
The training input is the official NAM file plus DI segments, so the manifest declares `training_input.custom_split` and
`train_stop_samples`. `scripts/train_a2.py` and `cloud/kaggle/train_a2_cloud.py` apply the same data-config patch the FC models were
trained with (`tests/test_cg_trainer_parity.py`), check the input by its recorded SHA-256 instead of the official MD5, and gate on the
core receptive field recorded per capture. Continuous Gain bundles are excluded from the Sessions list.

## Reproduction of the frozen JCM800 / Vibrolux FC configurations
`scripts/cg_reproduce_fc.py <amp>` (needs the archived `work/p4`, `work/p4e`, and your captures). Verified: capture sets, anchors,
levels, alignment, input and target audio SHA-256, output scale, split points - identical for both amps. Analysis from the real
captures also reproduces the archived profile/audit/selection exactly. `CG_REPRODUCE_AUDIO=1 pytest tests/test_cg_reproduction.py`.
Exported structure matches the frozen FC `.nam` (same architecture, layer sizes and weight-array lengths); the new export additionally
carries standard NAM user metadata (name, modeled_by).

## Where the research scripts went
The Phase 2-5 research scripts (`p4*`, `fc_*`, `p5_*`, `pl_*`, `tr_*`, v3 `cg_*`, `single_nam_*`, `continuous_gain_*`) were removed from `scripts/` on 2026-09-21. They are in `~/Documents/hybrid-nam-builder-archive/research_scripts_2026-09-21.tar.gz` (extract repo-relative) and in git history. `scripts/` keeps the app infrastructure plus `single_nam_common.py` and `cg_reproduce_fc.py`, which the frozen-configuration reproduction still uses.

## Sessions
Each project keeps a normal Sessions record (`work/sessions/<project id>.nam-mixer.json`, written by `routes.continuous_gain.sync_session` through the
app's own session writer): kind "Continuous Gain", how far it got, the selection/anchors, and - once trained - the NAM plus the trainers'
validation report (attached only if it belongs to that exact NAM). Sessions -> Load opens the Continuous Gain tab on that project; Delete
removes the record, the project folder and the training bundle. The tab has no project list of its own.

## Training material
The frozen recipe trains on the official NAM input plus three guitar DIs at eight level offsets. `hybrid/continuous_gain/excitation.py` generates a
single deterministic file that covers the same level range in one pass (synthesised guitar-like playing under a slow gain sweep);
`generate_bundle(recipe=..., load_di=..., official_transform=...)` lets experiments swap the material.

## Speed (no change to any result)
The slow steps are many independent native renders, so they run in parallel (`hybrid.continuous_gain.parallel.pmap`, order-preserving; default half the
cores capped at 6, override with `NAM_MIXER_CG_WORKERS`): capture probing, the per-capture renders of each training segment, and the stage-4
comparisons/sweeps. Probes are cached per project by capture file hash (`probe_cache.json`), so adding or removing a capture only probes what is new.
Measured on the JCM800 set (19 captures): analysis 176 s -> 49 s (8 s when nothing changed); bundle audio (Vibrolux, 6 captures) 116 s -> 25 s;
stage-4 validation 69 s -> 15 s. Outputs are identical: same probes/audit/profile/selection (and equal to the archived Phase 4 profile), same
training-audio SHA-256 as the frozen FC bundles, same validation figures and audition files. Training time and the Kaggle upload are unchanged.
