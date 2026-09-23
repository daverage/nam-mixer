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

- [ ] 1. `hybrid/core/` (envelope, safety, level_match, align, cab_ir, receptive_field, render, pipeline, …)
- [ ] 2. `hybrid/modes/` (the three design modes and their target generators)
- [ ] 3. `hybrid/training/` + `hybrid/continuous_gain/` + `hybrid/services/`
- [ ] 4. The Flask layer (`app.py`, `routes/`), `static/*.js`, `desktop/`, `cloud/`, `native/`, `scripts/`
- [ ] Coverage report (`pytest --cov`): untested code is where review finds the most problems and where dead code hides
- [ ] Final check: `/code-review ultra` on the whole tidy branch before merging

### Findings carried into Phase 3

- `app.py:1419` has an undefined name, `removed` (ruff F821). It was already there before Phase 2, and running that line raises `NameError`.
- `tests/test_cg_reproduction.py` and `scripts/cg_reproduce_fc.py` look for
  the frozen manifest in `docs/final/` and `docs/history/final/`, but it lives
  in `docs/history/Continuous Gain/final/manifest_frozen.json`, so 4 tests
  skip ("frozen FC manifest not available").
- Values computed and then never used (ruff F841): `hybrid/services/local_llm.py` `model`, `base_url`;
  `routes/continuous_gain.py` `STAGES`.
- Already failing on master: `test_local_llm_teaches_mode_selection_from_signal_behaviour`.
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
