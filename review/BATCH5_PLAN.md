# Batch 5 Review Plan: training_target.py (Risk 8.1)

**Target:** hybrid/modes/training_target.py  
**Risk Score:** 8.1/10  
**Status:** Starting  
**Date:** 2026-09-27

---

## Hotspot Summary

| Metric | Value |
|--------|-------|
| LOC | 632 |
| Commits | 6 |
| Criticality | CRITICAL |
| Coverage | 427 test LOC |
| Last Modified | 2026-09-23 |

**Key risks:**
- Bundle manifest schema is versioned + hashed; any silent change breaks training reproducibility
- Export naming follows strict rules (per CLAUDE.md): `[Amp Only]` vs. `[Full Rig]` vs. `[Learned Cab]` — misnamed export breaks downstream
- `generate_training_bundle()` is single path to A2 training data; no fallback
- Shared by Fixed Blend and Dynamic Hybrid modes; schema break affects both
- Real training execution on generated bundles doesn't happen in unit tests; manifest roundtrip validation only

---

## Review Questions

1. **Manifest schema:** Is the manifest structure versioned correctly? Can old readers handle it?
2. **Export naming:** Are the naming rules (Amp Only, Full Rig, Learned Cab, Embedded Cab) correctly implemented per CLAUDE.md?
3. **Cabinet handling:** When baking a cabinet IR, is it always applied consistently? No silent skips?
4. **Receptive field policy:** RF gating is enforced here (hard_required_samples must fit A2's RF)?
5. **Safety/peak ceiling:** Is peak ceiling always applied to training targets, never to preview?
6. **Shared helpers:** Is blend_training_target.py consistent with training_target.py?

---

## Implementation Plan

**Step 1:** Review manifest schema and versioning  
**Step 2:** Verify export naming rules implementation  
**Step 3:** Trace cabinet IR baking logic  
**Step 4:** Check RF policy enforcement + safety operations  
**Step 5:** Cross-reference with blend_training_target.py  
**Step 6:** Identify and triage findings  
**Step 7:** Implement fixes (if any)  
**Step 8:** Independent review + commit  

---

## Expected Findings

- 2-3 findings (manifest schema evolution, export naming edge cases, cabinet/RF edge cases)
- Estimated effort: Medium-High
- Estimated value: High (manifest drives reproducibility; naming affects training quality)

