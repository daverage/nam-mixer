# Batch 3 Review Plan: Receptive Field Policy Gating (train_a2.py, Risk 8.5)

**Date:** 2026-09-27  
**Risk Score:** 8.5/10 (CRITICAL)  
**Target:** `scripts/train_a2.py` (631 LOC, 28 commits) + `cloud/kaggle/train_a2_cloud.py` (667 LOC)

---

## Why This Matters

The **receptive field (RF) policy** is safety-critical:

1. **CORE (hard gate):** Amp A/B RF + envelope MUST fit inside A2's actual receptive field. Exceeding this ABORTS training (cannot represent the function).
2. **Cabinet (advisory):** Baked cab FIR may require A2 to approximate within available capacity. Reported honestly, never gates training by itself.
3. **Character (advisory):** Processing filters may overflow A2 RF. Same policy: report, don't abort.

**Real bug from the past:** Kaggle worker was too strict about cabinet overflow, refusing training when the intent was to allow approximation. Fixed by dual-mode policies (hard vs. advisory).

---

## Correctness Questions for Batch 3

### Q1: Hard gate correctly aborts on CORE overflow?
**Risk:** If the gate fails, A2 trained on an impossible function; results invalid.

**Test coverage:** 
- `test_receptive_field_parity.py::test_both_reject_core_overflow_regardless_of_cab` ✓
- `test_train_a2.py::test_check_receptive_field_*` (multiple)

**Blind spot to check:** Does the hard gate work when:
- NAM file is missing/corrupted? (local version tries to compute RF; cloud version reads from manifest)
- Manifest has `receptive_field` but is missing `branch_samples`?
- Source amps are from a different NAM version with different actual RF?

### Q2: Parity: do local and cloud reach same conclusions?
**Risk:** Silent drift where one worker rejects and the other accepts training.

**Mechanism:**
- Local: `check_receptive_field()` loads `.nam` files, computes RF with `compute_source_nam_receptive_field()`
- Cloud: reads only manifest's pre-computed `receptive_field.branch_samples`
- Parity test: `test_receptive_field_parity.py` loads both modules, compares decisions

**Test coverage:**
- 4 parity tests: core overflow, cabinet overflow, character overflow, no approximation needed ✓

**Blind spot to check:** Does parity hold when:
- Manifest has no `receptive_field` record at all?
- Branch samples are incomplete (e.g., only "Amp A" but not "Amp B")?
- Continuous Gain mode (one amp, N captures)?

### Q3: Cabinet/Character advisory paths never silently truncate?
**Risk:** Code comment says "never disables training" but does it actually train A2 to approximate, or does it silently fail?

**Check:** 
- Are advisory-path "CABINET APPROXIMATION" and "CHARACTER APPROXIMATION" printed unconditionally?
- Do training logs record the approximation mode in the manifest?
- Is validation run against both Full and Lite exports when approximation is reported?

---

## Concrete Review Tasks

### Task B3-1: Edge case analysis
**Prompt:** Inspect `check_receptive_field()` in both `scripts/train_a2.py` (line 215) and `cloud/kaggle/train_a2_cloud.py` (line 262). For each error path and edge case, verify:
1. Does it abort with a clear error, or silently continue with incorrect state?
2. Is the error message actionable (user can understand and fix)?
3. Does local version match cloud version behavior?

**Paths to check:**
- Line 249-251 (local): missing envelope_max_history_ms → returns empty dict (skip)
- Line 257-258 (local): missing amp path → prints warning, continue (skip)
- Line 289 (cloud): missing receptive_field record → returns empty dict (skip)
- Line 302 (cloud): empty branch_samples → returns empty dict (skip)
- Lines 335-338 (cloud) + 294-298 (local): hard gate logic

### Task B3-2: Continuous Gain edge case
**Prompt:** The manifest comment at line 242-246 (local) says CG mode's RF is pre-computed at bundle generation. But what if:
1. Bundle was generated with old code that didn't compute RF?
2. RF record is present but `branch_samples` is empty?
3. The per-capture RFs are in the manifest but the bounds/envelope is missing?

Trace through how `check_receptive_field()` handles these for CG mode.

### Task B3-3: Cabinet formal overflow reporting
**Prompt:** Inspect lines 376-389 (local) and 405-425 (cloud) -- the "CABINET APPROXIMATION" path. Verify:
1. Is the advisory warning printed ONLY when formal_total exceeds A2 RF, or always?
2. Does the result dict accurately reflect whether approximation was needed?
3. When approximation IS needed, is there validation code that checks the output against the baked target?

---

## Expected Findings

**Confidence: Medium** — Parity tests pass, hard gate tested. Blind spots likely in:
1. Edge cases with incomplete manifests (missing RF record, empty branch_samples)
2. CG mode edge cases (pre-computed RF may be incomplete)
3. Advisory path consistency (cabinet/character approximation always reported, never silent)

**Effort to fix (if issues found):**
- Missing data validation: Low (add checks, raise clear errors)
- Parity gap: Medium (may need to sync manifest generation logic)
- Advisory reporting: Low (ensure warning is printed)

---

## Execution Plan

**Step 10 (Review):** Inspect tasks B3-1, B3-2, B3-3 above using MAP.md, receptive_field.py, and test cases.

**Step 11 (Progress):** Record findings in review/FINDINGS_BATCH3.md, recommend continue or triage.

**Step 12 (Triage, fresh context):** Classify each finding CONFIRMED/THEORETICAL/WRONG/NOT_WORTH_IT.

**Step 13 (Independent check):** Spot-check any CONFIRMED findings for correctness.

**Step 14+:** Implement fixes (if any).

---

## Reference

- CLAUDE.md: RF policy rationale under "Design modes"
- docs/history/blend-mode.md: Full cabinet-approximation-policy section
- `hybrid/core/receptive_field.py`: RF computation engine
- `tests/test_receptive_field_parity.py`: Parity verification (4 tests, all passing)
- `tests/test_train_a2.py`: Local RF gating tests (5 tests, all passing)

**Next:** Proceed to Step 10 (Review batch).
