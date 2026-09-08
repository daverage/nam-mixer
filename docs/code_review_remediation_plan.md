# Code Review Remediation Plan

## Purpose

Resolve the verified defects from the 2026-09 code review, make the test suite
reliable from a clean checkout, and reduce the highest-cost duplication without
changing the audio/training behaviour.

## Baseline

- Review target: the current working tree, indexed after its uncommitted changes.
- Test command that resolves project imports: `python3 -m pytest -q`.
- Baseline result: 343 passed, 7 skipped, 1 failed.
- Do not overwrite unrelated, currently uncommitted work.

## Phase 1 — Correct the request-validation defect

### Problem

`POST /api/render_pair` converts `reference_input_level_dbu` before entering a
`TypeError`/`ValueError` handler. A non-numeric value therefore produces an
HTML 500 response instead of the API's normal JSON 400 validation response.

### Changes

1. In `app.py`, parse `reference_input_level_dbu` inside the same validation
   boundary as `test_gain_db` and the per-amp input gains.
2. Return a specific JSON 400 error naming the invalid field.
3. Keep valid defaults and the render/pair-cache semantics unchanged.
4. Add an application-route test that submits non-numeric
   `reference_input_level_dbu` with otherwise present required fields and
   asserts status 400 plus the JSON error contract.

### Acceptance criteria

- Invalid numeric input never reaches NAM loading or rendering.
- The response is JSON with HTTP 400.
- Existing valid render-pair tests remain green.

## Phase 2 — Make the test suite hermetic and easy to invoke

### Problem A: environment-dependent Kaggle test

`test_submit_async_persists_dataset_ref_before_long_upload_completes` creates
the default `KaggleCli`. On hosts where neither a `kaggle` executable nor the
Python package is available, the code returns before its mocked upload process
can start.

### Changes

1. Construct and inject a `KaggleCli(executable=...)` in that test, or reuse
   the test module's `make_cli()` helper while retaining its deliberately slow
   `Popen` implementation.
2. Assert that the mocked upload command is reached before checking persisted
   state.
3. Run the test in an environment without Kaggle installed to prove it has no
   host-tool dependency.

### Problem B: bare `pytest` cannot import the repository modules

The standalone `pytest` launcher does not put this repository on `sys.path`;
`python3 -m pytest` does. This is an avoidable contributor/CI trap.

### Changes

1. Add the smallest suitable test configuration (`pytest.ini` or
   `pyproject.toml`) that makes the repository root importable for pytest.
2. Document the canonical test command in `README.md`.
3. Avoid adding packaging metadata solely as a workaround unless the project
   is intentionally being packaged for distribution.

### Acceptance criteria

- Both `pytest -q` and `python3 -m pytest -q` collect the same test suite.
- Both commands pass without relying on an installed Kaggle CLI/package.
- The full suite is green, with skipped tests explicitly reported rather than
  silently deselected.

## Phase 3 — Consolidate safe duplicated primitives

### Scope

Create a small dependency-light internal utility module for stateless helpers.
Keep cloud-worker compatibility in mind: anything used by the Kaggle worker
must either be stdlib-only and staged with it, or remain intentionally local.

### Changes

1. Centralize the repeated chunked file-digest implementation used by:
   - `hybrid/training_target.py`
   - `hybrid/cab_ir.py`
   - `hybrid/character_analysis.py`
   - `hybrid/kaggle_training.py`
   - `scripts/train_a2.py`
   - `cloud/kaggle/train_a2_cloud.py`
2. Parameterize the digest algorithm rather than maintaining nearly identical
   SHA-256 and MD5 loops.
3. Centralize the identical `_rms_dbfs` implementation currently in
   `hybrid/fixed_blend.py` and `hybrid/level_match.py`.
4. Preserve public function names temporarily as thin wrappers if tests or
   external scripts may import the current private helpers.
5. Add focused equivalence tests for empty, small, and multi-chunk files, and
   for empty/non-empty RMS inputs.

### Acceptance criteria

- Hashes and RMS values are byte-for-byte/numerically identical to the
  pre-refactor behaviour.
- No production module carries a copied digest loop or the duplicate RMS body.
- Cloud training still runs from its staged files without importing the Flask
  application or unrelated local dependencies.

## Phase 4 — Remove confirmed dead artifacts and reduce test duplication

### Changes

1. Confirm no external workflow depends on tracked zero-byte files `pure` and
   `manifest-update`; then remove them in a dedicated cleanup commit.
2. Remove `CabDesign.to_dict()` if its lack of call sites is confirmed against
   external/plugin consumers; otherwise add an explicit test and document its
   API purpose.
3. Extract repeated test fixture writers (`_write_nam`,
   `_write_training_input`, and related synthetic bundle builders) into a
   shared test helper or fixtures module.
4. Do not delete ignored `work/`, `.venv-a2/`, or native build trees as part of
   this code change; they are local generated data and should have their own
   retention/cleanup policy.

### Acceptance criteria

- Cleanup removes only confirmed repository artifacts.
- Tests remain readable and retain their scenario-specific assertions.
- No undocumented external API is removed.

## Phase 5 — Reduce orchestration complexity without a behavioural rewrite

### Problem

`api_render_pair`, `api_preview`, and especially `api_generate` combine HTTP
parsing, validation, audio-domain orchestration, cache mutation, filesystem
writes, and response serialization. This makes changes risky and obscures
unit-level failure cases.

### Changes

1. Introduce typed request/parameter parsing helpers for render, hybrid,
   blend, character, and cabinet options.
2. Extract domain orchestration into service functions that return domain
   results rather than Flask responses.
3. Keep routes thin: translate request -> parameters, call service, translate
   expected domain failures -> JSON 400/409/500 responses.
4. Encapsulate the global rendered-pair cache behind a small interface. Define
   whether the application is single-user only; if not, make cache ownership
   session/job scoped before deployment beyond local use.
5. Preserve current endpoints and response fields; introduce contract tests
   before moving each route.

### Acceptance criteria

- Route functions are primarily transport adapters, not workflow engines.
- Unit tests cover parse failures and service outcomes without Flask setup.
- Existing UI/API contract tests remain green.
- Cache scope and concurrency expectations are documented.

## Phase 6 — De-risk local/cloud receptive-field policy drift

### Problem

The local trainer and Kaggle worker intentionally duplicate receptive-field
policy. A parity test exists, but the source has already diverged in details
such as manifest result fields and diagnostics.

### Changes

1. Define a compact, stdlib-only policy input/output contract based on the
   training manifest and installed A2 configuration.
2. Move shared calculations and result assembly into a module staged alongside
   `train_a2_cloud.py` for Kaggle jobs.
3. Leave environment-specific adapters local: source-NAM inspection belongs in
   the local trainer; manifest-only data belongs in the cloud worker.
4. Replace broad source-parity expectations with contract tests covering all
   modes, unavailable RF records, exact fits, over-capacity core targets, and
   baked-cab approximation.

### Acceptance criteria

- Local and cloud report the same policy fields for equivalent inputs.
- A policy change requires one implementation change plus shared tests.
- Cloud staging contains every module it imports.

## Recommended delivery order

1. Phase 1 (validation defect).
2. Phase 2 (green, reproducible suite).
3. Phase 4 cleanup, after explicit confirmation for deletions.
4. Phase 3 primitive deduplication.
5. Phase 5 route decomposition in small endpoint-by-endpoint commits.
6. Phase 6 receptive-field policy extraction after the route work is stable.

## Verification gate for every phase

1. Run focused tests for the changed behaviour.
2. Run `python3 -m pytest -q`.
3. Run bare `pytest -q` after Phase 2.
4. Exercise the affected Flask route with malformed and valid JSON payloads.
5. Review `git diff` to ensure generated model artifacts, `work/`, native
   build output, virtual environments, and unrelated user changes are absent.
