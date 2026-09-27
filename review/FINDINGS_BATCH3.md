# Batch 3 Findings: Receptive Field Policy Gating

**Date:** 2026-09-27  
**Analysis:** Edge case correctness in check_receptive_field (local + cloud)  
**Methodology:** Symbol inspection, error path tracing, parity analysis

---

## Finding B3-1: Silent Skip of Hard Gate When Branch Samples Unavailable

**Risk Level:** 🔴 CRITICAL  
**Finding ID:** B3-1  
**Track:** Correctness (safety-critical hard gate)  
**Severity:** Training proceeds without RF verification

### Claim

Both `scripts/train_a2.py::check_receptive_field()` (line 276-278) and `cloud/kaggle/train_a2_cloud.py::check_receptive_field()` (line 301-303) silently return an empty dict and **skip the entire RF hard gate check** when `branch_samples` is empty.

This means training can proceed without verifying that CORE RF fits inside A2 RF if:
1. For Hybrid/Character: `envelope_max_history_ms` is missing from manifest
2. For any mode: All source amp paths are missing or unreachable
3. For any mode: `receptive_field.branch_samples` is empty/None in manifest

### Evidence (Local Version)

**Lines 248-251 (Hybrid/Character path):**
```python
elif mode in ("hybrid", "character"):
    max_history_ms = manifest.get("design", {}).get("envelope_max_history_ms")
    if max_history_ms is None:
        print("WARNING: manifest has no envelope_max_history_ms -- skipping receptive-field check.")
        return {}  # ← SKIP hard gate entirely
```

**Lines 256-259 (Amp source paths):**
```python
amp_path = manifest.get(key, {}).get("path")
if not amp_path:
    print(f"WARNING: manifest has no {key}.path -- skipping {label}'s receptive-field check.")
    continue  # ← Just skip this amp, continue to next
```

**Lines 276-278 (Hard gate gate):**
```python
if not branch_samples:
    print("WARNING: no branch dependency could be determined -- skipping receptive-field check.")
    return {}  # ← If all sources failed/missing, return {} → NO HARD GATE
```

**Example scenario:** Hybrid mode with:
- `design.envelope_max_history_ms` = None (missing)
- `amp_a.path` = None (missing) OR unreachable
- `amp_b.path` = None (missing) OR unreachable
- Result: `branch_samples` empty → returns {} → hard gate SKIPPED ❌

### Evidence (Cloud Version)

**Lines 287-290:**
```python
rf_record = manifest.get("receptive_field")
if not rf_record:
    print("WARNING: manifest has no receptive_field record -- skipping receptive-field check.")
    return {}  # ← SKIP hard gate
```

**Lines 301-303:**
```python
if not branch_samples:
    print("WARNING: receptive_field.branch_samples is empty/unavailable -- skipping receptive-field check.")
    return {}  # ← SKIP hard gate
```

### Root Cause

Both implementations treat missing/unavailable data as "skip the check" rather than "abort training with a clear error". The intent (per code comments) is to degrade gracefully when data is unavailable, but the effect is:

**Intended behavior:** "We can't verify RF, but that's OK if the user knows what they're doing"  
**Actual behavior:** "We can't verify RF, so we silently allow training that might use an impossible function"

### What Would Confirm This Bug

1. Create a manifest with missing `envelope_max_history_ms` (Hybrid mode)
2. Create a manifest with both `amp_a.path` and `amp_b.path` missing/unreachable
3. Call `check_receptive_field(manifest, 48000)`
4. Observe: Returns `{}` instead of raising `TrainingAbort`
5. Observe: Training would proceed without RF hard gate verification

**Current behavior:** ✗ Confirmed by code inspection (no dynamic test created yet)

### Impact

- **Local trainer:** If input bundle is malformed (missing envelope or amp paths), training proceeds without RF verification. A2 trained on an impossible function; results invalid.
- **Cloud trainer:** If manifest's `receptive_field` record is missing or empty `branch_samples`, training proceeds without hard gate. Same impact.
- **Severity:** Safety-critical. The hard gate is the ONLY thing preventing training on non-representable functions.

### Recommendation

**Option A (Conservative):** Raise `TrainingAbort` if `branch_samples` is empty, with clear message: "Cannot determine core RF requirements; manifest is malformed or incomplete. Refusing to train."

**Option B (Degrade to warning):** Only skip hard gate if EXPLICITLY stated in manifest (e.g., `"receptive_field": {"skip_core_check": true}`). Otherwise, error.

**Preferred:** Option A — the hard gate is not optional.

---

## Finding B3-2: Parity Gap in Fallback Cascade (Local vs Cloud)

**Risk Level:** 🟡 MEDIUM  
**Finding ID:** B3-2  
**Track:** Correctness (parity requirement)  
**Severity:** Local and cloud might reach different conclusions

### Claim

When source amp paths are missing/unreachable:

- **Local version** (lines 256-264): Tries to load from file, catches errors, prints warning, continues.
- **Cloud version** (lines 292): Reads ONLY from manifest. If `branch_samples` doesn't include "Amp A" or "Amp B" keys, they're silently omitted (lines 294-297).

Parity test at `test_receptive_field_parity.py` constructs identical `branch_samples` from manifest, so local and cloud see the same data. But if a REAL bundle is generated with incomplete manifest, the two might diverge.

### Evidence

**Local version (lines 256-264):**
```python
for label, key in source_keys if mode != "continuous_gain" else ():
    amp_path = manifest.get(key, {}).get("path")
    if not amp_path:
        print(f"WARNING: manifest has no {key}.path -- skipping {label}'s receptive-field check.")
        continue
    try:
        model = load_nam(amp_path)
        branch_samples[label] = compute_source_nam_receptive_field(model)
    except (OSError, ValueError, ReceptiveFieldUnavailable) as exc:
        print(f"WARNING: could not compute {label}'s receptive field ({amp_path}): {exc}")
```

If `amp_a.path` is missing, `branch_samples` won't have an "Amp A" key.

**Cloud version (lines 292-297):**
```python
all_branch_samples = {k: int(v) for k, v in (rf_record.get("branch_samples") or {}).items() if v is not None}
mode = manifest.get("mode", "hybrid")
branch_samples = {
    k: v for k, v in all_branch_samples.items()
    if mode != "character" or not str(k).startswith("character_")
}
```

Cloud just reads what's in manifest's `receptive_field.branch_samples`. If local trainer failed to compute "Amp A" RF and manifest only has "Amp B", cloud sees the same.

**BUT:** What if manifest has BOTH "Amp A" and "Amp B" recorded, but local's `load_nam()` fails on one? Local will skip it (warning), cloud will see the recorded value. Divergence possible.

### Parity Test Gap

`test_receptive_field_parity.py` uses `_matching_manifests()` to create identical `branch_samples` for both local and cloud (line 74: `"branch_samples": {"amp_a": 3, "amp_b": 3}`).

But it does NOT test the scenario: "Manifest has recorded branch_samples, but local load_nam() fails."

### Recommendation

Document or add test: "If local load_nam() fails, is the recorded manifest value trusted, or is training aborted?" Currently, local prints warning and continues with partial branch_samples (only the ones that succeeded).

---

## Finding B3-3: Cabinet Approximation Reporting Consistency

**Risk Level:** 🟢 LOW  
**Finding ID:** B3-3  
**Track:** Correctness (advisory path consistency)  
**Severity:** User may not see cabinet approximation warning

### Claim

The "CABINET APPROXIMATION" warning (local lines 376-389, cloud lines 405-423) is printed to stdout, but:

1. Is it ALWAYS printed when `cab_requires_approximation = True`?
2. Is it ONLY printed when approximation is needed, or also printed in other cases?
3. Is the flag correctly set in ALL code paths?

### Evidence

**Local version (lines 376-389):**
```python
# The formal total is NEVER a hard gate -- see function docstring. Only
# report/record whether A2 is being asked to approximate the cab.
if cab_formal > rf.receptive_field_samples:
    result["cab_requires_approximation"] = True
    print("\nCABINET APPROXIMATION:\n" + ...)
else:
    print(f"  Cabinet-adjusted core fits inside the A2 receptive field ...")
```

This is conditional on `cab_formal > rf.receptive_field_samples`. But `cab_formal` is only computed after `_resolve_baked_cab_fir_samples()` returns successfully (line 353-357).

**What if `_resolve_baked_cab_fir_samples()` returns 0 (no FIR samples)?**

Line 354-355:
```python
if not cab_fir_samples:
    return result
```

If `_resolve_baked_cab_fir_samples()` fails to find FIR data (both the IR file and the manifest fallback), it returns `(0, None)`. Then `if not cab_fir_samples` is True (0 is falsy), and the function returns early WITHOUT printing any cabinet message.

**Result:** If cab baking falls back to 0 samples (missing data), the function silently returns with `result["cab_baked"] = True` but no approximation message.

### Cloud Version Behavior

**Lines 389-391:**
```python
cab_fir_samples = _resolve_baked_cab_fir_samples(manifest)
if not cab_fir_samples:
    return result
```

Same pattern: if `_resolve_baked_cab_fir_samples()` returns 0, returns early with no message.

### Scenario

1. Bundle has `cab.baked = True`
2. IR file is missing or invalid
3. Manifest's `receptive_field.cab.fir_history_samples` is missing
4. Result: Both fallbacks fail, `_resolve_baked_cab_fir_samples()` returns `(0, None)`
5. User doesn't know the cabinet was NOT actually baked into training

### Recommendation

Either:
- (A) Ensure `_resolve_baked_cab_fir_samples()` always returns a valid FIR history (fail with clear error if both sources unavailable)
- (B) Print a warning if cabinet was marked baked but FIR data is 0 samples

Currently, both versions silently treat a 0-sample baked cab as "cabinet not actually applied" without warning.

---

## Summary Table

| ID | Finding | Risk | Status | Fix Effort |
|---|---|---|---|---|
| B3-1 | Hard gate silently skipped when branch_samples empty | 🔴 CRITICAL | CONFIRMED | High |
| B3-2 | Parity gap if local load_nam() fails | 🟡 MEDIUM | PLAUSIBLE | Low |
| B3-3 | Cabinet approximation reporting when FIR is 0 | 🟢 LOW | PLAUSIBLE | Low |

---

## Next Steps

**Step 11 (Progress):** Document findings, assess if high-risk issues warrant block/continue decision.

**Step 12 (Triage):** Verify each finding is CONFIRMED, then classify practical impact.

**Step 13 (Independent check):** Have a fresh reviewer spot-check B3-1 (critical).

**Step 14 (Implement):** Fix B3-1 first (safety-critical hard gate), then B3-2/B3-3.

