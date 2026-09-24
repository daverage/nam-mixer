# Repo review & tidy-up tracker

Branch: `tidy/repo-review` (from `master` @ `e3b0a1d`, v0.3.5)

**Rule:** keep deleting, moving, and changing code in separate commits. Do the
phases in order, with the test suite passing at every step, so a failing test
always points at a single kind of change.

Status key: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` skipped/decided against

---

## Phase 0: Set a baseline

- [x] Create branch `tidy/repo-review`
- [x] Run `python -m pytest tests/` and write down what passes and what skips (see Baseline below)
- [x] Re-index codebase-memory at the current `HEAD` (2026-09-23: re-index succeeded, 4505 nodes / 15115 edges)

### Baseline (2026-09-23, master `e3b0a1d`, `.venv-a2` Python 3.11)

**673 passed, 1 failed, 16 skipped** in ~43 s.

Already failing on master (not caused by the tidy work, fix in Phase 3 group 4):
- `tests/test_local_llm.py::test_local_llm_teaches_mode_selection_from_signal_behaviour`:
  asserts `"constant mixture"` is in the local-LLM system prompt; the prompt no
  longer contains it (likely `533f98b`/`d4e642c`).

Expected skips (environment-dependent):
- `test_cg_reproduction.py` ×6: frozen FC manifest absent / `CG_REPRODUCE_AUDIO` unset
- `test_namcore_sequential_gate.py` ×5: `NAM_RENDER_BASELINE`/`NAM_RENDER_SEQUENTIAL_EXE` unset
- `test_receptive_field.py` ×2, `test_train_a2.py` ×1: NAM installed (unavailable-path not exercised) / no real captures
- `test_render.py` ×2: no real `.nam` in `assets/nam_models/` + built `nam_render`

Tooling: `ruff` available (system Python 3.11); `vulture` not installed.

---

## Phase 1: Find what can go (a list first, no deleting yet)

Build a list of deletion candidates with evidence for each. The user reviews
it before anything is removed. Delete in one commit per category.

- [x] Python code nothing uses: `vulture` (run with and without `tests/`),
      `ruff --select F401,F841,F811`, grep to confirm. (The codebase-memory
      Cypher dialect allows only one `WITH` and can't express "no non-test
      callers", so vulture + grep were used instead.)
- [x] Scripts: every file in `scripts/` checked for references
- [x] Files that probably shouldn't be tracked
- [x] Docs: duplicates, stale locations, broken path references
- [x] Frontend: every static/template file referenced; JS functions scanned
- [x] **User reviews the candidate list below and fills in Decision** (2026-09-23: "please remove")
- [x] Delete approved items, one commit per category: A 4ec0ac8, C1 83a27ec, D5 d5c9db9, D1 (see Log)

### Candidate list (2026-09-23)

Decision column: `delete` / `keep` / `move` / `?` (user to decide). My
recommendation is in **bold**.

#### A. Dead Python/JS code (unused even by tests, confirmed by grep)

| Item | Evidence | Rec. | Decision |
|------|----------|------|----------|
| `hybrid/audio_metrics.py:129` `band_energy_dbfs` | no reference anywhere | **delete** | **removed** (4ec0ac8) |
| `hybrid/metadata.py:47` `HybridMetadata.write_sidecar` | no reference anywhere | **delete** | **removed** (4ec0ac8) |
| `hybrid/research.py:312` `tone3000_notes` | no reference anywhere | **delete** | **removed** (4ec0ac8) |
| `hybrid/kaggle_training.py:420` `KaggleCli.datasets_create` | only named in a test (not called by prod) | **check test, likely delete** | **removed** with its test stub (4ec0ac8) |
| `scripts/single_nam_common.py` `render_capture`, `render_file_nam`, `load_law`, `save_json` | only `capture_path`/`official_input` are imported (by `cg_reproduce_fc.py`); the rest belong to the abandoned single_nam law | **delete** | **removed**; the whole module is now trimmed to those two functions (4ec0ac8) |
| `static/app.js:1421` `notImplementedAction` | defined, never called | **delete** | **removed** (4ec0ac8) |
| 10 unused imports (ruff F401): `hybrid/training_target.py`, `hybrid/cg_validation.py`, `hybrid/cg_project.py` ×2, `cg_routes.py`, `scripts/cg_reproduce_fc.py` ×2, 3 test files | not monkeypatch targets (grepped tests) | **delete (`ruff --fix`)** | **removed** (4ec0ac8) |
| 8 unused locals (ruff F841) | 5 in tests: delete. `hybrid/local_llm.py` `model`, `base_url` and `cg_routes.py` `STAGES` might be real bugs (a value computed then ignored), so hand these to the Phase 3 review | **tests: delete; prod: Phase 3** | tests **removed** (4ec0ac8); prod 3 → Phase 3 |

False positives, ignored: every Flask `api_*` route (decorator-registered),
pydantic `@field_validator` methods in `local_llm.py`, `HTMLParser.handle_*`
overrides in `research.py`, and anything under `packaging/backend/dist/`
(gitignored build output).

#### B. Production code used only by tests

| Item | Rec. | Decision |
|------|------|----------|
| `a2_training_settings.settings_for` | ? | |
| `audio_metrics.envelope_error_db`, `framed_spectral_correlation`, `multi_resolution_log_spectral_distance` (CG research metrics, cited in history docs) | ? | |
| `cab_ir.PreparedCabIr.energy_fraction_within` (CLAUDE.md documents it as a diagnostic) | **keep** | |
| `cg_excitation.level_swept_excitation`, `active_rms` | ? | |
| `kaggle_training.KaggleJobManager.submit` | ? (check whether the routes use a different entry point) | |
| `local_llm.suggest_recipe` | ? | |
| `multi_blend.*.designated_gain_db` | ? | |
| `nam_provenance.build_export_name`, `agreed_tone_type` (named in comments in `cab_ir.py`/`train_a2.py`) | ? | |
| `receptive_field.cab_fir_serial_history_samples` | **keep** (RF policy maths, tested) | |
| `sequential_nam.package_embedded_sequential` | ? | |
| `envelope.rms_envelope_db` (DEPRECATED; only tests + `compare_envelopes.py`) | **keep for now**; revisit if C1 is deleted | |

Recommendation: decide these in the Phase 3 review of each group, where the
surrounding code is being read anyway. Don't bulk-delete them here.

#### C. Scripts

| Item | Evidence | Rec. | Decision |
|------|----------|------|----------|
| C1 `scripts/compare_envelopes.py` | one-off old-vs-new envelope comparison; only referenced in the `envelope.py` docstring | **delete** (and fix the docstring) | **removed** (83a27ec) |
| All other scripts | referenced by README/CLAUDE.md/tests/code/CI | **keep** | |

#### D. Duplicated / questionable tracked files

| Item | Evidence | Rec. | Decision |
|------|----------|------|----------|
| D1 4 `.nam` files byte-identical between `deliverables/` and `docs/history/Continuous Gain/phase4e/models/` (`*_P4E_B_s0` = `RECOMMENDED_*`, `v3/*_3Captures` = `alt_v3_C3`) | md5 match | now both copies are under `docs/history/Continuous Gain/` (`deliverables/` + `phase4e/models/`); delete the `deliverables/` copies? | **removed** (user ran `git rm`; README points at phase4e/models) |
| D2 `deliverables/` (8 trained `.nam`s in git) | release artifacts | archive | **archived** → `docs/history/Continuous Gain/deliverables/` (67fab8c) |
| D3 `deliverables/README.md` links `docs/CONTINUOUS_GAIN_FINAL_CANDIDATES.md` | file is now under `docs/history/Continuous Gain/` | **fix link** | **fixed** (67fab8c) |
| D4 `desktop/src-tauri/icons/Square*Logo.png`, `StoreLogo.png` (11 files) | Windows Store/MSIX icons; not in `tauri.conf.json`, CI builds macOS `app` only | ? delete, unless a Windows MSIX build is planned | **keep** (user) |
| D5 `assets/di/peaks/moderate_brit.wav.reapeaks` | REAPER waveform-peak cache, referenced nowhere | **delete** (+ ignore `*.reapeaks`) | **removed** (d5c9db9) |

Verified keep: `work/.gitkeep` (keeps the ignored dir), `.codebase-memory/.gitattributes`
(indexer merge rule; graph files are ignored), `packaging/backend/dist/` (ignored).

#### E. Docs

| Item | Evidence | Rec. | Decision |
|------|----------|------|----------|
| E1 `docs/Continuous Gain/continuous_gain_ui_package/` (proposal + 5 SVGs) | the CG tab is implemented (`docs/continuous_gain_tab.md` is the live doc) | **move to docs/history** | **archived** (c4aaa29) |
| E2 `docs/Continuous Gain/AMP_CONTROL_RESEARCH.md` | research background | ? keep as live doc or move to history | **archived** (c4aaa29) |
| E3 `docs/settings_ux_refactor_plan.md` | plan doc, no status marker (last touched 2026-09-22) | ? move to history if implemented | **archived**: implemented in 6ee39c7/08701be (c4aaa29) |
| E4 `docs/history/Continuous Gain/phase3.md` + `phase3_progress.md` | about the Hybrid A2 training target, not Continuous Gain; misfiled | **move to `docs/history/`** (Phase 2) | |
| E5 ~9.6 MB of PNG/.nam under `docs/history/` | history plots/models | ? keep (only D1 dupes removed) | |

#### F. Broken doc-path references (a mechanical fix; own commit in Phase 2)

~50 source/test/doc files point at `docs/<name>.md` paths that moved into
`docs/history/` (`phase3.md` in ~20 files, `blend-mode.md` in ~15,
`kaggle_training.md`, `INPUT_PROFILE_RESEARCH.md`, `blend-mode-fixes.md`, and
3 CG docs). Fix after the Phase 2 moves so each path is only rewritten once.

#### Frontend

All `static/` and `templates/` files are referenced. 199 JS functions were
scanned; only `notImplementedAction` (A) is unused. CSS selectors were not
scanned (low value).

---

## Phase 2: Tidy the structure (moves only, no behaviour change)

User decisions (2026-09-23): 5 subpackages, drop the `cg_` prefix inside
`continuous_gain/`, move only `cg_routes.py` into `routes/`; splitting
`app.py` into blueprints is a refactor, so it goes on the Phase 3 list.

- [x] `hybrid/paths.py` `REPO_ROOT` replaces six `Path(__file__).parent.parent`
      computations, so later moves can't silently change paths (14d8863)
- [x] `hybrid/` → `core/`, `modes/`, `continuous_gain/`, `training/`, `services/`
      via `git mv` + import rewrite; monkeypatch strings updated (d5fdf43)
- [x] `cg_routes.py` → `routes/continuous_gain.py` (3fdb5c7)
- [x] Main-app phase docs (phase2/3/4 + progress) moved out of
      `docs/history/Continuous Gain/` into `docs/history/` (c4e1b70)
- [x] Path references updated in 74 live files, plus README tree,
      CLAUDE.md, AGENTS.md (20581f8). `docs/history/` and this file's
      candidate list keep old paths as historical records. Comments citing
      the archived research scripts (`scripts/p4_*.py`, `fc_*.py`, …) are
      left as provenance.
- [x] Tests unchanged after every step (673/1/16; JS 17/17)
- [x] Codebase-memory re-indexed (4507 nodes)
- [x] D1 duplicate `.nam` files removed

---

## Phase 3: Review each file (group by group, dependencies first)

For each group: `/code-review high <path>`, then fix what it finds in a
separate commit.

- [x] 1. `hybrid/core/`: reviewed 2026-09-23 (10 findings + 4 minor). Fixed:
      cab_ir missing-file/double-read/cache-path (a05df5b), render OSError →
      NamRenderError (2f038f6), render_bootstrap atomic replace + CPU-aware
      asset + 100-release page (551a73c), silent-amp auto-trim in core AND
      Parallel Blend (5bea99f), preview limiter NaN (df838d2). **Rejected:**
      "trim on the shifted envelope" contradicted a deliberate, tested design
      (b8fa25c reverted in 3e4242e). **Not changed, for the user to decide:**
      - ~~`align.estimate_offset` O(max_lag·n)~~: user approved the FFT
        version; it matches the old loop's chosen lag exactly (8e139fd).
      - ~~Input-only calibration label~~: three-way status, Tools panel shows the input level (8212158).
      - ~~Silent DI coverage~~: reports "No active playing detected" (c7ad644).
      - ~~Input-peak warning ignores per-amp gain~~: per-amp input peaks + warning naming the gains (38897c5).
- [x] 2. `hybrid/modes/`: reviewed 2026-09-23 (10 findings, all Character
      Blend except the duplication one). Fixed: preview analysed the raw DI
      instead of profiled_dry, and the analysis cache was keyed on the .nam
      only (2d09f87); the preview low-level check omitted per-amp input gain
      (bd4e000); generation didn't check that the .nam still matches the
      frozen hash (4895d51); the gate now runs before the full renders
      (b12ca6b). **Checked, not a problem:** the gate's t=0 excerpt. The
      official V3 input has ~1 s of near-silence, then real playing.
      **Not changed, for the user to decide:**
      - ~~(5) Character RF record vs v3 teacher~~: now derived per teacher version;
        v3 adds the residual-envelope branch, is exactly bounded, and was checked
        against the real teacher; still advisory (2b3454d).
      - ~~(8) Per-level correction memory~~: streamed, bit-identical on the full V3
        input; the mix step drops ~1.9 GB → ~0.65 GB and the overall peak goes
        1905 → 1531 MiB (now set by `_select_donor`, ~1.3 GB, and the envelope,
        ~0.9 GB, which are untouched) (8133ad3).
      - ~~(9) level_window_db dropped at freeze~~: recorded and restored; legacy
        designs load as 3.0 (f902e1c).
      - (7) Stitched-sample spectrum: **v2 contiguous-frame method implemented
        as opt-in analysis version 2 (626a5d3); default is still v1.**
        Comparison on real captures: per-cell EQ-correction changes up to
        8 dB (mean 0.7–1.3 dB), teacher band changes ≤ 0.7 dB, difference
        signal −24 to −33 dB. **Waiting for the user's listening review:**
        `work/character_spectrum_listening/` (README + 18 WAVs +
        comparison.json).
      - (10) Duplicated bundle generators: deferred until the above is
        settled. When done, keep each mode's format-specific behaviour, add
        before/after manifest + target regression tests, and document any
        intentional manifest correction separately.
- [x] 3. `hybrid/training/` + `hybrid/continuous_gain/` + `hybrid/services/`: reviewed
      2026-09-24 in three parts (the first attempts hit the usage limit and
      then stalled twice, so training was split into kaggle_training.py and
      the rest). All verified findings fixed, each with a test that fails
      without the fix:
      - **Full/Lite swapped everywhere (1e48510):** NAMCore slim 0.0 = Lite and
        1.0 = Full (proved by rendering each extracted submodel), but the
        app, CG validation, Kaggle re-validation, the Character quiet check
        and train_a2/validate_a2 all used 0.0 for Full. There are now
        `SLIM_FULL`/`SLIM_LITE` constants and a real-render test. **Reports
        and manifests written before this have Full and Lite metrics
        swapped.**
      - Services: `.env` newline injection (57e7f5b); local-LLM width
        validator 2-18 (4a990fa); empty env var masking `.env` (a2b9199);
        update check per_page (5c2bd73); Ollama pull stuck on 'running' plus a
        single recommended model (cd5c616).
      - Continuous Gain: <2 eligible captures crash and NaN anchors for
        identical responses (bb710fe); manifest records the applied ceiling
        (328f6a3); audit 'corrected' only when applied (12526b4).
      - Training: nam_tools 0 dB / no-metadata edits rejected (c041776);
        Windows CRLF log lines lost and split UTF-8 (2138831); venv check
        missing `nam`, the packaged app probing itself, and slow status polls
        (0da0b10); embedded completion leaking exceptions and validating
        zero samples (2e0b4ce).
      - Kaggle: secret redaction (partial values, JSON keys, bytes) (76cd08a);
        stuck 'preparing' plus unverified kernels never deleted (9964294);
        status parsed from the kernel ref (e.g. a username containing
        'running') (8d4a2c3); double downloads, permanent orphans and the
        cancel race (481d614); logs (286f00b); unexpected validation errors
        (6f71205).
      **Not changed, for the user to decide:**
      - Embedded-package tolerance 3e-6 (absolute) may be too tight for long
        real IRs near 0 dBFS (float32 NAMCore vs fftconvolve). Plausible but
        unproven here: the Sequential renderer isn't configured on this
        machine.
      - `nam_provenance.build_export_name`/`export_name_suffix` are test-only
        and have diverged from the production naming in
        `a2_training_settings.user_metadata_kwargs` and `sequential_nam`
        (e.g. legacy baked cab: '[Learned Cab]' vs '[Amp Only]'). Which one
        is canonical? `_TONE_TYPES` is also copied three times.
      - CG `_shift`/`_db` helpers are copied across bundle/validation/probe
        (a consolidation refactor; could go with #10).
      - `stage()` still writes the legacy 'uploading' state (the UI labels it
        correctly, so it's cosmetic).
- [ ] 4. The Flask layer (`app.py`, `routes/`), `static/*.js`, `desktop/`, `cloud/`, `native/`, `scripts/`
- [ ] Coverage report (`pytest --cov`): untested code is where review finds the most problems and where dead code hides
- [ ] Final check: `/code-review ultra` on the whole tidy branch before merging

### Findings carried into Phase 3

- ~~`app.py:1419` undefined `removed`~~: an unreachable line; removed (8c5901c).
- ~~CG reproduction manifest path~~: fixed; 4 tests now run and pass (c4908bb).
- ~~Unused locals in local_llm / CG routes~~: removed (507178c).
- ~~Already failing on master: local_llm mode-selection test~~: the prompt had also started advertising a 1-24 dB width against the UI's 2-18 dB slider; fixed both (9de4114). **Suite is fully green.**
- Split `app.py` (3,285 lines, ~60 routes) into `routes/` blueprints (a refactor).
- Category B (production code used only by tests): decide per group.

### Rules to check in every group

- [ ] The envelope must stay causal.
- [ ] `apply_peak_ceiling` is for training targets; `preview_safety_limiter` is for preview only.
- [ ] The cheap reblend path must never re-run NAM inference.
- [ ] `dry_gain_db` must never be wired to a user-facing control.
- [ ] The local and cloud receptive-field checks must stay the same (the parity test covers this).
- [ ] Kaggle credentials must never be logged.

---

## How it runs

- Phases 1 and 2 are worked through together with the user in-session. They
  need the user's judgement on what to keep.
- Phase 3 is one review per group, not per file (~6–8 reviews), so links
  between files stay visible to each review.

## Log

- 2026-09-23: Phase 0 done. Branch created, codebase-memory re-indexed OK,
  baseline 673/1/16 (1 failure already on master, see Baseline).
- 2026-09-23: Phase 1 candidate list written (A–F). Waiting for the user's
  decisions before deleting anything.
- 2026-09-23: User decisions: archive the CG test leftovers and docs that are no
  longer in use; keep the Windows Store icons. Done: `deliverables/` →
  `docs/history/Continuous Gain/deliverables/` (67fab8c); settings UX plan, CG UI
  proposal, and amp-control research → `docs/history/` (c4aaa29). Tests unchanged
  (673/1/16). Still to confirm: A, C1, D1 (duplicate .nam), D5.
- 2026-09-23: User approved all remaining deletions. Done: A (4ec0ac8), C1
  (83a27ec), D5 (d5c9db9). Tests unchanged after each (673/1/16; JS 17/17).
  D1 (4 duplicate .nam copies in `docs/history/Continuous Gain/deliverables/`)
  was blocked twice by the auto-mode permission classifier, so the user needs
  to run it or allow it.
- 2026-09-23: Phase 2 done (14d8863, d5fdf43, 3fdb5c7, c4e1b70, 20581f8). New
  layout: hybrid/{core,modes,continuous_gain,training,services}, routes/.
  Findings for Phase 3 are listed under Phase 3. D1 still pending (user).
- 2026-09-23: D1 done (user ran `git rm`, bfef3c9). Phase 1 and 2 are complete. Starting Phase 3 group 1 (`hybrid/core/`).
- 2026-09-23: Phase 3 group 1 (`hybrid/core/`) done, see the checklist. Also
  fixed the known findings (CG manifest path, local_llm width/test, unused
  locals, dead return). Suite fully green: 686 passed, 12 skipped. Process
  lesson: b8fa25c was pushed with a failing test because the shell chain
  didn't gate on pytest's result; commits and pushes are now gated on it.
- 2026-09-23: The four group-1 items the user decided on are done (align FFT, calibration label,
  silent-DI coverage, per-amp peak warning). 690 passed, 12 skipped. Verifying the group 2 (`hybrid/modes/`) findings next.
- 2026-09-23: Phase 3 group 2 (`hybrid/modes/`) done, see the checklist. 694 passed, 12 skipped.
- 2026-09-23: modes #5, #8 and #9 done; #7 implemented as opt-in v2, stopped for the listening review; #10 deferred. 709 passed, 12 skipped.
- 2026-09-24: Phase 3 group 3 done (training, Continuous Gain, services). 755 passed, 12 skipped. Next: group 4 (Flask layer, JS, desktop, cloud, native, scripts); #7 still awaits the listening review; #10 deferred.
