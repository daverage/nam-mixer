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

- [ ] Python code nothing uses: codebase-memory graph (no callers), `vulture`,
      `ruff --select F401,F841`. Code only called from tests counts as dead too.
- [ ] Scripts: for each file in `scripts/`, check for references (code, docs, CI,
      tests). Keep deliberately guarded ones, e.g. `cg_reproduce_fc.py`, which
      guards the frozen FC recipe.
- [ ] Files that probably shouldn't be tracked: a tracked file in the
      gitignored `work/`; decide on `deliverables/` and `.codebase-memory/`.
- [ ] Docs: duplicated docs or docs for dropped designs. Moving them to
      `docs/history/` is often better than deleting them.
- [ ] Frontend: JS/CSS/template code no route or page uses.

### Candidate list

| Path | Category | Evidence | Decision |
|------|----------|----------|----------|

---

## Phase 2: Tidy the structure (moves only, no behaviour change)

- [ ] `cg_routes.py` (repo root, next to `app.py`) → a proper routes package
- [ ] `hybrid/` (51 flat modules) → subpackages (core DSP, one per design mode,
      training, validation)
- [ ] Only move files and fix imports; run the tests after each move
- [ ] Update CLAUDE.md, README, AGENTS.md (they reference module paths heavily)
- [ ] Re-index codebase-memory after the structural commits

---

## Phase 3: Review each file (group by group, dependencies first)

For each group: `/code-review high <path>`, then fix what it finds in a
separate commit.

- [ ] 1. Core signal processing: `envelope`, `safety`, `level_match`, `align`, `cab_ir`
- [ ] 2. `pipeline`, `design`, and the three design modes (`blend`, `fixed_blend`, `character_blend`)
- [ ] 3. Training targets, `receptive_field`, validation, Kaggle, and the `cg_*` modules
- [ ] 4. The Flask layer (`app.py`, routes), `static/*.js`, `desktop/`, `cloud/`, `native/`
- [ ] Coverage report (`pytest --cov`): untested code is where review finds the most problems and where dead code hides
- [ ] Final check: `/code-review ultra` on the whole tidy branch before merging

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
