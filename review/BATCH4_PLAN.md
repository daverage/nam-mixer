# Batch 4 Review Plan: pipeline.py (Risk 8.3)

**Target:** hybrid/core/pipeline.py  
**Risk Score:** 8.3/10  
**Status:** Not started  
**Date:** 2026-09-27

---

## Hotspot Summary

| Metric | Value |
|--------|-------|
| LOC | 277 |
| Commits | 7 |
| Criticality | CRITICAL |
| Coverage | 142 + 188 test LOC |
| Last Modified | 2026-09-25 |

**Key risks:**
- `render_pair()` is EXPENSIVE (runs NAM inference twice); `build_hybrid()` is CHEAP (numpy reblend)
- Cost-isolation boundary: any breach here tanks performance or loses design state
- `RenderedPair` dataclass reused by all three design modes; schema changes affect all
- Alignment/offset application happens here; off-by-one errors silently corrupt audio
- Real `nam_render` invocation tests auto-skip unless native binary built + sample `.nam` present

---

## Review Questions

1. **Cost boundary:** Does `render_pair()` ever accidentally re-render after the first call? Can `build_hybrid()` trigger expensive operations?
2. **Mode dispatch:** Does each of the three design modes (Dynamic Hybrid, Fixed Blend, Character Blend) correctly route through the pipeline?
3. **RenderedPair schema:** Are all three modes using consistent fields? Can a mode-specific field silently break another mode's caller?
4. **Alignment/offset:** When applying `frozen_alignment_offset`, is the offset applied consistently across all audio branches?
5. **Null/empty handling:** What happens when DI is silent, amps produce no output, or envelope is flat?

---

## Implementation Plan

**Step 1:** Review pipeline.py for cost boundary and dispatch logic  
**Step 2:** Inspect RenderedPair schema usage across modes  
**Step 3:** Trace alignment offset application in render_pair/build_hybrid  
**Step 4:** Identify and triage findings  
**Step 5:** Implement fixes (high-priority first)  
**Step 6:** Independent review + commit  
**Step 7:** Move to Batch 5 (training_target.py)

---

## Expected Findings

- 2-4 findings (dispatch edge cases, schema inconsistency, alignment off-by-one, silent fallbacks)
- Estimated effort: Medium
- Estimated value: High (core DSP pipeline; affects all three design modes)

