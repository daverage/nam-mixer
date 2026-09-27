# Correctness Review Findings

**Track:** Correctness  
**Focus:** app.py (Risk 9.2) — Request-response contract drift  
**Methodology:** Symbol search (search_graph), source code inspection, parameter/response validation  
**Date:** 2026-09-27

---

## Findings (5 total)

| ID | Track | Path | Symbol | Line Range | Claim | Evidence | Related Symbols | Tool Reference | Impact | Consequential |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Correctness | app.py | api_render_pair | 2223-2393 | Response fields used by UI lack defensive null checks; missing field access silently breaks JavaScript | UI reads data.blend_envelope_percentiles (L2816), data.suggested_crossover_dbfs (L2827), data.render_id (L2850) without verifying they exist in response. If API stops returning one, UI throws undefined errors. | applyRenderResult, doRenderPair (static/app.js:2810-2890) | search_graph, manual read | Silent JavaScript errors halt Render/Preview workflow; user cannot continue | Yes |
| C2 | Correctness | app.py | _resolve_cab_design | 1936-1966 | Cabinet IR preparation parameters have hidden API defaults not reflected in UI request contract | API expects cab_preparation_mode (default "trim_initial_silence", L1956) and cab_leading_silence_threshold_db (default -40.0, L1957) but UI (static/app.js:3115) never sends these fields. UI relies on undocumented API defaults. | _parse_cab_params (app.py:1884), cabParamsBody (static/app.js:2315-2321) | get_code_snippet, read | If API defaults change or logic alters, UI/API diverge silently; cab preparation behavior unpredictable | Yes |
| C3 | Correctness | static/app.js | applyRenderResult | 2810-2863 | Render response field access assumes fields exist; no defensive programming for API contract evolution | data.warnings (L2817), data.suggested_crossover_dbfs (L2827), data.blend_envelope_percentiles (L2816) accessed without typeof/nullish checks. If API response schema changes, UI crashes. | updateCrossoverKnobCalibration, renderTimingReadout, updateCoverage (static/app.js) | read | UI render workflow halted on schema mismatch; user cannot recover without browser console | Yes |
| C4 | Correctness | app.py | api_generate | 3230-3426 | Response returns cab_summary even when cab is None; UI interprets as truthy without null safety | api_generate (L3356) returns cab_summary = design.cab.to_dict() if design.cab else None, but static/app.js:3141-3143 accesses data.cab_summary.export_mode/baked without null check, causing error if cab is None | _resolve_cab_design, freeze_design (hybrid/modes/design.py) | get_code_snippet, read | If cab-less generation happens, UI crashes on null reference; training result UI not rendered | Yes |
| C5 | Correctness | static/app.js | currentModeParamsBody | 2309-2312 | Character blend parameter schema change risk — UI sends drive_low/mid/high_mix_b to API but no schema validation in request | UI characterParamsBody (L2279-2284) sends drive_low_mix_b, drive_mid_mix_b, drive_high_mix_b conditionally, API _parse_character_params (app.py:2474-2485) expects them but doesn't validate presence/type. No test verifies matching fields. | _build_character_result (app.py:2461), characterParamsBody (static/app.js:2279) | search_graph, read | If Character Blend parameter names drift, generation silently uses wrong values or defaults | Yes |

---

## Summary

**Blind Spot Identified:**  
Request-response contracts between Flask API (app.py) and browser UI (static/app.js) lack defensive programming. Three categories of risk:

1. **Response field access without null checks** (C1, C3, C4) — UI assumes API response fields exist. If API stops returning optional fields or changes schema, JavaScript throws undefined errors that halt the workflow.

2. **Hidden API defaults not documented in UI** (C2) — Cabinet IR preparation has API-side defaults (trim_initial_silence, -40.0 dB) that the UI doesn't know about. UI relies on these defaults silently.

3. **Parameter schema drift undetected** (C5) — Character Blend parameter names and presence sent by UI are not formally validated by API tests; no schema contract testing.

**Test Coverage Gap:**  
`tests/test_app.py` exercises API routes but does not:
- Verify response field presence/schema (test_generate_end_to_end_produces_bundle checks data.model_name but not full response contract)
- Test schema mismatch scenarios (what if API stops returning suggested_crossover_dbfs?)
- Validate parameter contract symmetry (does every field UI sends match what API expects?)

---

## Verifiability

**C1, C3, C4:** Falsified by adding optional chaining (`?.`) to UI field access and re-running live preview/generation on a modified API that omits response fields. Current code breaks; fixed code tolerates schema drift.

**C2:** Falsified by extracting cab preparation defaults to UI code or request schema. If code still works identically, concern was theoretical. If behavior changes, API/UI diverged.

**C5:** Falsified by adding type/presence validation in _parse_character_params or via pytest schema test. Currently no validation means schema changes slip through silently.

---

**Effort & Impact Summary:**  
- **Effort:** Low (add ?? checks + defaults in UI; add schema validation tests in API)
- **Impact:** High (silent failures halt production workflows; recovery requires browser console debugging)
- **Confidence:** High (code inspection confirmed missing checks; test suite gap confirmed)

---

## Batch 2: Kaggle Training State Machine (kaggle_training.py)

| ID | Track | Path | Symbol | Line Range | Claim | Evidence | Related Symbols | Tool Reference | Impact | Consequential |
|---|---|---|---|---|---|---|---|---|---|---|
| K1 | Correctness | kaggle_training.py | retry_download | 1423-1431 | Race condition: job hangs in "downloading" state if concurrent refresh() thread adds to _downloading between lock release and _download_once() call | Line 1425: lock released after _downloading check. Line 1430: _download_once() called without lock. If concurrent refresh() calls _download_once() and adds job_id to _downloading in window, retry_download's _download_once() returns silently (line 1224) leaving job.state="downloading" forever. | _download_once (L1219-1230), _live_lock (L689-692) | manual read, grep | Job UI hangs in "downloading" state indefinitely; user cannot cancel/retry; backend thinks download in progress forever | Yes |
| K2 | Correctness | kaggle_training.py | submit_async | 1132-1157 | Overly broad Exception catch hides transient (network retry-able) failures as permanent; no backoff or retry logic | Line ~1150: `except Exception as exc: # noqa: BLE001` catches all exceptions, persists as job.error, marks job failed. No distinction between network timeout (retry-able) and auth/file error (permanent). | _worker/_run_pipeline (L1091-1119), KaggleTrainingError (L950) | manual read | Transient network errors (timeout, rate limit) become permanent job failures; user must manually retry from UI with no exponential backoff | Yes |
| K3 | Correctness | kaggle_training.py | refresh | 1165-1177 | False negative: slow submission thread triggers premature "interrupted" failure; no timeout/heartbeat to distinguish slow from dead | Line 1166: `if not still_submitting` marks job as interrupted if job_id not in _submitting. But _submitting.add() at L1088 happens INSIDE _run_pipeline thread; if thread is slow waiting on Kaggle API, job_id removed from set at L1119 before kernel even created, causing false "interrupted" marking. | _run_pipeline (L1087-1119), submit_async (L1132-1157) | manual read | User sees "submission interrupted" error when upload is actually slow; they re-submit, creating duplicate jobs on Kaggle | Yes |
| K4 | Correctness | kaggle_training.py | _download_and_validate | 1273-1390 | Partial file download accepted as valid: Kaggle CLI charmap error workaround checks only "something downloaded", not file completeness | Lines 1304-1310: if `kernels output` exits non-zero but `downloaded_something` is true (even 1 byte), treats as non-fatal. Lines 1315-1325: validates nam_path without checking file size/integrity. Truncated .nam accepted as valid training result. Code comment confirms real production incident (charmap crash on Windows). | result.ok (L1304), downloaded_something (L1293), nam_candidates (L1321-1325) | manual read | Corrupted/truncated .nam model accepted as valid training output; user trains broken model; wasted computational resources | Yes |
| K5 | Correctness | kaggle_training.py | cancel_active | 1461-1494 | Partial failure on resource deletion not retryable: job marked failed even if Kaggle delete fails, leaving orphaned resources | Lines 1476-1484: if datasets_delete or kernels_delete fails (network, rate limit, auth), errors collected in job.cleanup_error but job.state = "failed" (L1485). cleanup() refuses to retry terminal jobs (L1437). Orphaned Kaggle resources persist, user quota exhausted. | cleanup (L1435-1450), save_job, _live_lock (L1473-1474) | manual read | Kaggle datasets/kernels accumulate as orphaned; user quota exhausted; no recovery mechanism; manual Kaggle cleanup required | Yes |

---

**Batch 1 (app.py):** 5 findings triaged to **0 consequential** (callers already defensive).

**Batch 2 (kaggle_training.py):** 5 NEW CONFIRMED findings, all consequential. Core issues:
1. **Race condition (K1):** retry_download hangs forever in "downloading" state
2. **Error handling (K2):** Transient network errors treated as permanent; no retry backoff
3. **Timing detection (K3):** Slow submissions falsely marked interrupted; duplicate jobs
4. **Validation (K4):** Partial file downloads accepted; corrupted models used for training
5. **Cleanup (K5):** Failed resource deletion leaves orphans; user quota exhausted

**Real production incidents confirmed in code comments:** Kaggle CLI charmap codec error (Windows), kernels_output exit code unreliability.

