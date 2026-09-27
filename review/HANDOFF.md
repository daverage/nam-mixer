# Review Handoff

**Session:** Batch 2 (kaggle_training.py) — K4 Implementation  
**Date:** 2026-09-27  
**Status:** In progress (K4 complete, K2/K5/K3/K1 pending)

---

## Phase Summary

**Batches completed:**
- Batch 1 (app.py): 5 findings → triaged to 0 consequential (defensive layers in callers)
- Batch 2 (kaggle_training.py): 5 findings → all CONFIRMED and consequential

**Current work:** Implementing CONFIRMED findings from review/TRIAGED.md in priority order

---

## Completed: K4 (Data Integrity)

**Finding:** Partial .nam file downloads accepted as valid; corrupted models used for training

**Implementation:**
- Added JSON validation in `_download_and_validate()` (hybrid/training/kaggle_training.py, lines 1330-1346)
- Validates .nam files are valid JSON with required 'architecture' field
- Rejects on JSONDecodeError, missing fields, or file read errors
- Added regression test: `test_download_and_validate_rejects_truncated_nam_file()`

**Tests:** All 108 tests in test_kaggle_training.py pass ✓

**Commit:** 15f32ac  
**Message:** "K4: Validate .nam file integrity; reject truncated/corrupted downloads"

**Review requirement:** K4 is consequential → next action: **independent review of commit 15f32ac**

---

## Remaining Work (Priority Order)

| ID | Finding | Priority | Effort | Next Action |
|---|---|---|---|---|
| K2 | Broad Exception catch in submit_async | 1 (High) | Low | Implement retry backoff + transient error handling |
| K5 | Orphaned resources on deletion failure | 2 (High) | Low | Add cleanup retry loop or webhook mechanism |
| K3 | Slow submission marked interrupted | 3 (Medium) | Medium | Add heartbeat/timeout to _submitting tracking |
| K1 | State staleness in concurrent downloads | 4 (Medium) | Medium | Hold lock through state update in retry_download |

---

## Commands & Results

**Build:** `python app.py` (Flask dev server)  
**Targeted test:** `pytest tests/test_kaggle_training.py::test_download_and_validate_rejects_truncated_nam_file -xvs`  
**Full test suite:** `pytest tests/test_kaggle_training.py -x`

**Last test run:**
```
tests/test_kaggle_training.py ........................... [ 38%]
..................................................................       [100%]
============================= 108 passed in 4.30s ==============================
```

---

## Open Questions

1. **K1 impact:** Does caller of retry_download() rely on returned state being current, or re-read from disk? If re-read, impact is low (eventual consistency).

2. **K3 timing:** In practice, how often do slow Kaggle uploads exceed typical refresh() poll intervals? (Affects likelihood of duplicate job creation.)

---

## Next Action

**Step 13 (Independent review):** Review commit 15f32ac for K4 fix before proceeding to K2 implementation.

Then continue with K2 → K5 → K3 → K1 in remaining review sessions.

