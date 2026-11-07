# Review Progress

**Session:** Correctness track focus (app.py, Risk 9.2)  
**Date:** 2026-09-27  
**Status:** Batch 1 complete

---

## Batch 1: Request-Response Contract Drift (app.py)

**Scope:** Three critical API endpoints (/api/render_pair, /api/preview, /api/generate) + browser UI contracts (static/app.js).

**Findings:** 5 (after triage: 0 consequential)
- C1: THEORETICAL — Response field access; callers are already defensive
- C2: THEORETICAL — Cabinet preparation defaults work; risk only if API defaults change
- C3: THEORETICAL — Duplicate of C1
- C4: WRONG — False positive; code already has null check (ternary operator)
- C5: THEORETICAL — Schema is symmetric; no actual parameter drift

**False-Positive Rate:** 20% (C4 was incorrect)
**Consequential findings:** 0 (triage reclassified all as theoretical or false positives)

**Context & Tool Cost:**
- Tools used: search_graph (1 call, 20 symbols), get_code_snippet (3 calls, 3 functions full), Read (8 file ranges), Bash grep (6 searches)
- Tokens spent: ~40k (searches + source reads)
- Files touched: app.py (5 functions), static/app.js (4 functions), tests/test_app.py (enumerated but not read in detail)

**Test Coverage Verified:**
- `tests/test_app.py` exercises routes (test_generate_end_to_end_produces_bundle, test_render_pair_*) but does NOT verify:
  - Response field presence/schema (C1, C3, C4)
  - Parameter contract symmetry (C5)
  - Cabinet defaults documentation (C2)

---

## Remaining High-Risk Questions

**Within app.py scope (continue to batch 2?):**
1. **Kaggle job state machine** — What are the untested edge cases in kaggle_training.py (Risk 8.7)? Network timeouts, rate limits, job eviction not covered by mocks.
2. **Manifest schema evolution** — training_target.py (Risk 8.1) bundle manifest is versioned + hashed, but are old version readers tested? Silent version drift possible.
3. **RF policy gating correctness** — scripts/train_a2.py's hard gate (Risk 8.5) correctness unverified on real torch environment; only synthetic tests.

**Outside app.py scope (batch 3+?):**
- Subprocess error modes (nam_render, torch, Kaggle CLI failures) — tests mock at subprocess boundary; real tool failures untested
- Streaming AI responses (local_llm.py) — no unit tests for connection drop mid-stream
- Character Blend tone correction filter behavior on diverse captures — synthetic-only testing

---

## Expected Value of Next Batch

**Batch 2 (kaggle_training.py, Risk 8.7):**
- **Expected findings:** 3–4 (edge cases in subprocess polling, manifest validation, network retry logic)
- **Effort:** Medium (trace_path through job state machine, cross-reference with test mocks)
- **Value:** High (cloud training only path; silent failure modes expensive; parity with train_a2.py already has a test harness)
- **Confidence:** Medium–High (subprocess mocking is well-documented blind spot)

**Cumulative finding quality:** Batch 1 targeted the highest-risk file (app.py, 90 commits, monolithic). Batch 2 targets the second-most volatile training path (kaggle_training.py, 12 commits, subprocess risk). Both are **CRITICAL** per HOTSPOTS.md.

---

---

## Batch 2: Kaggle Job State Machine Edge Cases (kaggle_training.py)

**Status:** Complete

**Scope:** kaggle_training.py (Risk 8.7, 1215 LOC, 12 commits) — cloud subprocess management, job polling, parity with train_a2.py.

**Findings:** 5 (all confirmed, all consequential)
- K1: CONFIRMED — Race condition in retry_download causes job to hang in "downloading" state
- K2: CONFIRMED — Exception handling too broad; no transient/permanent distinction; no retry backoff
- K3: CONFIRMED — Slow submission thread triggers false "interrupted" marking; duplicate jobs created
- K4: CONFIRMED — Partial file downloads accepted as valid; truncated .nam used for training
- K5: CONFIRMED — Partial deletion failure leaves orphaned Kaggle resources; quota exhausted

**Root cause pattern:** Kaggle API unreliability (charmap errors, exit code unreliability documented in code comments) partially mitigated but edge cases remain. Concurrent job state updates (refresh vs. retry_download) not atomic.

---

## Recommendation (after Batch 2 completion)

**Stop and triage Batch 2 in fresh context** — All 5 Batch 2 findings are consequential and verified (race condition, error handling, file validation, resource cleanup). These require independent review before moving to Batch 3.

**One reason:** Kaggle training is the only cloud path; these findings affect live job workflows. K1 (race condition) and K3 (duplicate jobs) require careful triaging to confirm the fix isn't simply "make job state machine fully atomic" vs. finer-grained synchronization. Independent reviewer should verify the race window and confirm impact.

**Batches completed:** 2 (app.py: 0 consequential after triage; kaggle_training.py: 5 confirmed)  
**Next steps:** Step 12 (triage Batch 2), then either Step 14 (implement fixes) or continue to Batch 3 (train_a2.py, Risk 8.5) if triage confirms findings are high-confidence and actionable.


## HybridN investigation, 2026-11-07

Open question: does the reported low-gain result originate in source calibration, crossover coverage, or A2 approximation? Baseline HEAD 933c1218760d345a3b4396dedbedbe250852f21b; run provenance edfe82ff84cf7ba7af7de25d916c1f8ad5fa995c. Existing static/style.css modification is outside this review. Read-only production review; artifacts only in review/. Targeted eight-file source budget: pipeline.py, blend.py, envelope.py, training_target.py, render.py, validation_report.py, static/app.js and scripts/train_a2.py, with bounded excerpts. Saved run manifests and audio/model measurements establish the specific case. No branch diff requested, so the fixed-point code-review workflow is inapplicable.

Completed HybridN investigation: matching exported SHA verified; six DI input levels rendered through both sources, exact teacher and Full/Lite; seven sine levels rendered through both sources and Full/Lite. Result: Full retains Mesa behavior when input reaches crossover; input-level sensitivity demonstrated. Report: review/HYBRIDN_REVIEW.md. No production changes. Eight implementation files, 624 bounded source lines preserved in tools; source snippets and measurement JSON saved. No production tests required for documentation-only review. Known limit: no user playback signal or host settings, so exact listening cause remains unconfirmed.
