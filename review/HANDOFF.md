# Review Handoff

**Session:** Batch 3 (train_a2.py) — RF Policy Gating Review (BLOCKED)  
**Date:** 2026-09-27  
**Status:** Batch 2 merged; Batch 3 Step 10 review complete; CRITICAL blocking issue found (B3-1)

---

## Phase Summary

**Batches completed:**
- Batch 1 (app.py): 5 findings → triaged to 0 consequential (defensive layers in callers)
- Batch 2 (kaggle_training.py): 5 findings → all CONFIRMED and consequential

**Current work:** Implementing CONFIRMED findings from review/TRIAGED.md in priority order

---

## Step 15: Independent Review — COMPLETE

**Reviewer:** Claude Haiku 4.5 (fresh context)  
**Date:** 2026-09-27  
**Result:** See review/INDEPENDENT_REVIEW.md

**K4 (15f32ac):** ✅ APPROVED — File validation correct, no side effects  
**K2 (820e824):** ✅ APPROVED — Error classification and retry logic sound  
**K5 (820e824):** ❌ CRITICAL BUG FOUND → Fixed in f43b1bd  

---

## Completed: K4 (Data Integrity) ✅ APPROVED

**Finding:** Partial .nam file downloads accepted as valid; corrupted models used for training

**Implementation:**
- Added JSON validation in `_download_and_validate()` (hybrid/training/kaggle_training.py, lines 1330-1346)
- Validates .nam files are valid JSON with required 'architecture' field
- Rejects on JSONDecodeError, missing fields, or file read errors
- Added regression test: `test_download_and_validate_rejects_truncated_nam_file()`

**Tests:** All 108 tests in test_kaggle_training.py pass ✓

**Commit:** 15f32ac  
**Independent review:** ✅ APPROVED (review/INDEPENDENT_REVIEW.md)

---

## Completed Implementations

| ID | Finding | Status | Commits | Reviewed |
|---|---|---|---|---|
| K4 | Partial .nam downloads accepted | ✅ FIXED | 15f32ac | ✅ Approved |
| K2 | Broad Exception catch | ✅ FIXED | 820e824 | ✅ Approved |
| K5 | Orphaned resources on deletion | ✅ FIXED | 820e824 + f43b1bd | ✅ Approved |
| K3 | Slow submission marked interrupted | ⏸️ DEFERRED | — | — |
| K1 | State staleness in concurrent downloads | ⏸️ DEFERRED | — | — |

---

## Commands & Results

**Full test suite:** `pytest tests/test_kaggle_training.py -x`

**Final test run (after K5 bug fix):**
```
tests/test_kaggle_training.py .......................................... [ 38%]
..................................................................       [100%]
============================= 108 passed in 4.31s ==============================
```

**Changes committed:**
- 15f32ac: K4 — File validation for partial downloads
- 820e824: K2+K5 — Retry logic + resource cleanup (contains K5 bug)
- 6eea4f1: Consolidation commit
- f43b1bd: K5 fix — Persist cleanup_error in cancel_active (identified & fixed in independent review)

---

## Batch 3 Findings (train_a2.py RF Policy Gating)

**Status:** Step 10 complete; Step 11 decision: BLOCK release

**Critical Finding B3-1:** Hard gate silently skips when branch_samples empty
- Both local and cloud versions return {} if envelope_max_history_ms missing (Hybrid/Character)
- Or if all source amp paths are missing/unreachable
- Result: Training proceeds without RF verification (A2 trained on impossible function)
- **Impact:** Safety-critical hard gate that must never silently skip
- **Recommendation:** Raise TrainingAbort instead of silent skip
- **Fix effort:** High (design change in error handling)

**Medium Finding B3-2:** Parity gap if local load_nam() fails
- Plausible but edge case; defer to 0.5.6
- Needs additional parity test case

**Low Finding B3-3:** Cabinet approximation warning missing when FIR is 0
- Plausible but cosmetic; defer to 0.5.6

**Next action:** Step 12 (triage B3-1 in fresh context), then Step 14 (implement fix)

See review/FINDINGS_BATCH3.md and review/BATCH3_PROGRESS.md for details.

---

## Deferred Work (K3, K1, B3-2, B3-3)

**K3** requires timeout-based approach to distinguish slow uploads from interrupted ones. Initial implementation failed tests because timeout window doesn't align with test execution speeds. Needs refinement:
- Add configurable timeout parameter for testing
- Or: implement heartbeat mechanism in submission thread

**K1** addresses race condition where retry_download() state update can race with concurrent refresh(). Initial fix added double-locking which broke _download_once(). Needs:
- Redesign to avoid deadlock while maintaining state consistency
- Or: accept eventual consistency model (job state may be temporarily stale)

---

## Session Summary

**Batch 2 review & implementation: 3 of 5 findings FIXED + REVIEWED**
- ✅ K4 (data integrity): Critical fix complete + independent review approved
- ✅ K2 (error handling): High-priority fix complete + independent review approved  
- ✅ K5 (resource cleanup): Fix complete, bug found in review, fixed in f43b1bd, approved
- ⏸️ K3, K1: Deferred for next session (test/design concerns)

**Independent review findings:**
- K5 had data-loss bug: cleanup_error not persisted to job (found in fresh review)
- Bug fixed: _delete_kaggle_resources() now returns (cleanup_state, cleanup_error) tuple
- All 108 tests pass; no regressions

**Ready for next action:**
- Merge to master (all K4/K2/K5 approved)
- Or continue to Batch 3 (train_a2.py receptive field, Risk 8.5)
- K3/K1 can remain deferred for 0.5.6 unless they're blocking

