# Batch 6 Review Plan: local_llm.py (Risk 7.9)

**Target:** hybrid/services/local_llm.py  
**Risk Score:** 7.9/10  
**Status:** Starting  
**Date:** 2026-09-27

---

## Key Risks

- **Ollama subprocess management:** fallback to public endpoint on local failure
- **Streaming response handling:** partial failures (connection drop mid-stream) not obviously caught
- **No unit tests:** integration-only if Ollama running locally
- **Optional feature:** leaks into UI if enabled; UX degradation if fails silently

---

## Quick Investigation

1. Check subprocess error handling (transient vs permanent)
2. Verify streaming error handling
3. Identify untested failure modes
4. Check fallback logic (local → public)

Expected: 2-3 findings (error handling, fallback robustness, streaming edge cases)

