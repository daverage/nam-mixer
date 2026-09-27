# Batch 4 Findings: pipeline.py (Risk 8.3)

**Date:** 2026-09-27  
**Status:** Investigation in progress  
**Target:** hybrid/core/pipeline.py (277 LOC)

---

## Investigation Summary

Reviewed pipeline.py for correctness issues in:
1. Cost boundary between render_pair() (EXPENSIVE) and build_hybrid() (CHEAP)
2. Mode dispatch across three design modes (Dynamic Hybrid, Fixed Blend, Character Blend)
3. RenderedPair schema consistency
4. Alignment offset application
5. Edge cases (empty audio, calibration overflow, repeated calls)

---

## Code Flow Analysis

### render_pair() (lines 84-181)
- Takes dry input + two amp models
- Applies input_profile_gain_db + test_gain_db BEFORE NAM inference ✓
- Applies per-model calibration gain (per official NAM formula) ✓
- Applies per-amp input_gain_db (for limited headroom handling) ✓
- Computes envelope from profiled_dry (for crossover) + raw dry (for coverage analysis) ✓
- Returns RenderedPair with all metadata

**No issues found:** render_pair() correctly separates the expensive NAM inference from blending logic.

### build_hybrid() (lines 216-277)
- Takes already-rendered RenderedPair
- Applies dry_gain_db ONLY to envelope (test-only, deprecated) ✓
- Computes alignment offset, applies to amp_b BEFORE level matching ✓
- **CRITICAL:** Level matching uses ORIGINAL envelope, not dry_gain_db-shifted one ✓
  - Test `test_auto_trim_ignores_dry_gain_db_and_measures_the_real_input_level` verifies this
- Blends using shifted envelope (so test-gain DOES affect the crossfade) ✓
- Returns HybridResult with applied offset

**No issues found:** build_hybrid() correctly implements the cost/quality boundary and test parameter handling.

### RenderedPair Schema (lines 37-82)
- All fields have defaults (safe for all three modes to construct)
- No mode-specific fields (three modes share the same dataclass) ✓
- Tracks both profiled and raw dry for different analyses ✓

**No issues found:** Schema is uniform across modes.

### Alignment Offset Application (lines 248-249)
- resolve_alignment_request() validates: if disabled, must not pass offset ✓
- apply_fixed_offset() creates new array, safe for repeated calls ✓
- Offset applied to amp_b BEFORE level matching (correct for time-aligned comparison) ✓

**No issues found:** Alignment correctly applied.

---

## Edge Cases Checked

1. **Empty audio:** _peak_dbfs() handles zero-length arrays ✓
2. **NaN/Inf:** amp_input_peak_warnings() already flags out-of-range peaks ✓
3. **Repeated calls:** build_hybrid() is pure (no in-place modifications), safe for repeated calls ✓
4. **Alignment=0:** Correctly treated as "no offset" ✓
5. **Mode compatibility:** All three modes use the same RenderedPair, no conflicts ✓

---

## Routes Integration (from app.py)

### /api/render_pair (line 2223)
- Calls render_pair() correctly with all calibration/profile parameters ✓
- Validates sample rates match (line 2282) ✓
- Caches result for subsequent /api/preview calls ✓

### /api/preview (line 2888)
- Calls build_hybrid() for "hybrid" source ✓
- Calls build_fixed_blend() for "blend" source ✓
- Calls build_character_blend() for "character" source ✓
- All three pass RenderedPair (cached, no re-render) ✓

### /api/character/low_level_check (line 2535)
- Re-renders at different gain levels on a shorter excerpt ✓
- Recreates RenderedPair-like structure with SimpleNamespace ✓
- Does NOT use pipeline RenderedPair (intentional, uses trimmed reference) ✓

---

## Test Coverage Review (test_pipeline.py)

Tests present:
- build_hybrid() with/without auto-level ✓
- Manual trim combines with auto trim ✓
- dry_gain_db shifts envelope (test-only mode) ✓
- Auto-trim ignores dry_gain_db (correct behavior) ✓
- Repeated calls on same pair (cost isolation) ✓

Tests **absent:**
- render_pair() validation (does not auto-skip if .nam present)
- Alignment offset edge cases (0, max int, negative)
- Empty dry input to render_pair()
- Mode-specific field access after render_pair()

---

## Candidate Findings

### P4-1: No validation of render_pair() input parameters

**Claim:** render_pair() does not validate envelope_config parameter; malformed config silently passed to bounded_causal_envelope_db().

**Evidence:** Line 98 accepts BoundedEnvelopeConfig without type check. If caller passes wrong type, envelope computation might fail silently.

**Verdict:** PLAUSIBLE but LOW-RISK (Flask route validates via JSON schema; direct Python calls use type hints).

---

### P4-2: RenderedPair.source_envelope_db not used in build_hybrid()

**Claim:** RenderedPair includes source_envelope_db (raw dry envelope) but build_hybrid() never uses it; only pair.envelope_db (profiled) is used.

**Evidence:** Line 246 uses pair.envelope_db. Line 154 computes source_envelope_db but line 266 (blend call) uses envelope_db.

**Verdict:** NOT A FINDING. source_envelope_db is used by coverage.analyse_profile_coverage() (different module). Confirmed in coverage.py.

---

## Next Steps

1. Verify render_pair() behavior with edge cases (all-zero dry, very high gains)
2. Confirm mode compatibility with a cross-mode test
3. If no P4-1 issue confirmed, move to Batch 5 (training_target.py, Risk 8.1)

