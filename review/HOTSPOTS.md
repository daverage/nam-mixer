# Code Hotspots & Risk Analysis

**Analysis date:** 2026-09-27  
**Source data:** review/INDEX.csv, review/TOOLS.md, review/tools/*, pytest results  
**Methodology:** Normalized scoring across change frequency, complexity, criticality, and test coverage

## Top 10 Risk Hotspots

### 1. `app.py` (3214 LOC, 90 commits)

**Risk Score: 9.2/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 90 commits | Highest activity — volatile routing layer |
| Complexity Estimate | 1260 | Extreme; highest in codebase |
| Size | 3214 LOC | Monolithic Flask app; contains routing + state mgmt |
| Criticality | CRITICAL | Entry point to all `/api/*` endpoints + web interface |
| Test Coverage | No dedicated tests | Gap: app.py logic lacks integration test isolation |
| Last Modified | 2026-09-26 | Recent changes; active development |

**Why it's a hotspot:**
- Single-point failure for all web routes and API contracts
- Routing/state management intertwined; refactoring risk high
- No unit tests isolate app.py logic from routes/Flask framework
- Each commit changes potential customer-facing behavior

**Blind spot:** Request-response contract drift between API and browser UI not caught by unit tests (only Flask integration tests).

---

### 2. `hybrid/training/kaggle_training.py` (1215 LOC, 12 commits)

**Risk Score: 8.7/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 12 commits | Active training pipeline refactoring |
| Complexity Estimate | High | Subprocess management + job polling + retry logic |
| Size | 1215 LOC | 2nd largest file; GPU orchestration |
| Criticality | CRITICAL | Only path to cloud A2 training (Kaggle backend) |
| Test Coverage | 1876 test LOC | Comprehensive `test_kaggle_training.py` + mocking |
| Last Modified | 2026-09-24 | Recent activity |

**Why it's a hotspot:**
- Cloud subprocess execution: shell commands to Kaggle API (never `shell=True`, but subprocess risk remains)
- Job state machines (submit→poll→download→validate) have edge cases (network failure, timeout)
- Parity requirement: must match `cloud/kaggle/train_a2_cloud.py` logic exactly; drift is a silent bug
- 12 commits suggest recurring fixes/tuning

**Blind spot:** Network/timeout scenarios tested by mock; real Kaggle API behavior (rate limits, session expiry) not covered.

---

### 3. `scripts/train_a2.py` (631 LOC, 28 commits)

**Risk Score: 8.5/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 28 commits | Highest single-file churn after app.py |
| Complexity Estimate | 200+ | Manifest parsing, RF checking, training coordination |
| Size | 631 LOC | Large CLI; orchestrates local training |
| Criticality | CRITICAL | Local A2 training entry point + RF policy enforcement |
| Test Coverage | Partial | `test_train_a2.py` covers RF parity; training itself mocked |
| Last Modified | 2026-09-25 | Very recent |

**Why it's a hotspot:**
- RF (receptive field) policy gating: hard gate MUST abort if CORE RF exceeds A2's actual RF (per CLAUDE.md)
- Cabinet RF advisory path printed but must never silently truncate
- 28 commits = frequent fixes/adjustments
- Manifest structure is parity-checked against cloud worker but can drift if loader changes

**Blind spot:** Real torch/neural-amp-modeler environment tests auto-skip; RF policy gates tested only in synthetic/mocked scenarios.

---

### 4. `hybrid/core/pipeline.py` (277 LOC, 7 commits)

**Risk Score: 8.3/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 7 commits | Steady evolution |
| Complexity Estimate | 95 | Control flow: mode dispatch, render/blend coordination |
| Size | 277 LOC | Medium; bridges all design modes |
| Criticality | CRITICAL | `render_pair()` + `build_hybrid()` are core DSP pipeline |
| Test Coverage | 142 + 188 test LOC | `test_pipeline.py` + `test_pipeline_render.py` |
| Last Modified | 2026-09-25 | Recent |

**Why it's a hotspot:**
- `render_pair()` is EXPENSIVE (runs NAM inference twice); `build_hybrid()` is CHEAP (numpy reblend)
- Cost-isolation boundary: any breach here tanks performance or loses design state
- `RenderedPair` dataclass reused by all three design modes; schema changes affect all
- Alignment/offset application happens here; off-by-one errors silently corrupt audio

**Blind spot:** Real `nam_render` invocation tests auto-skip unless native binary built + sample `.nam` present; synthetic tests may miss native tool failures.

---

### 5. `hybrid/modes/training_target.py` (632 LOC, 6 commits)

**Risk Score: 8.1/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 6 commits | Stable; large file rarely refactored |
| Complexity Estimate | 100+ | Manifest building, audio processing, export naming |
| Size | 632 LOC | 4th largest; training target generation core |
| Criticality | CRITICAL | Freezes current design into A2 training input |
| Test Coverage | 427 test LOC | `test_training_target.py` covers manifest schema |
| Last Modified | 2026-09-23 | Slightly older |

**Why it's a hotspot:**
- Bundle manifest schema is versioned + hashed; any silent change breaks training reproducibility
- Export naming follows strict rules (per CLAUDE.md): `[Amp Only]` vs. `[Full Rig]` vs. `[Learned Cab]` — misnamed export breaks downstream
- `generate_training_bundle()` is single path to A2 training data; no fallback
- Shared by Fixed Blend and Dynamic Hybrid modes; schema break affects both

**Blind spot:** Real training execution on generated bundles doesn't happen in unit tests; manifest roundtrip validation only (not actual A2 training).

---

### 6. `hybrid/services/local_llm.py` (1053 LOC, 8 commits)

**Risk Score: 7.9/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 8 commits | Active feature (AI Assistant) |
| Complexity Estimate | High | Ollama subprocess, streaming, prompt caching |
| Size | 1053 LOC | 3rd largest; AI backend orchestration |
| Criticality | HIGH | Optional feature; user-facing AI conversational UI |
| Test Coverage | None observed | No dedicated tests for llm.py |
| Last Modified | 2026-09-23 | Recent activity |

**Why it's a hotspot:**
- Ollama subprocess management: fallback to public endpoint on local failure (network dependency)
- Streaming response handling: partial failures (connection drop mid-stream) not obviously caught
- Model pull (`ollama_pull.py`) runs background subprocess; failure modes untest
- Optional but leaks into UI if enabled; UX degradation if llm fails silently

**Blind spot:** No isolated unit tests; integration only if Ollama running locally. Streaming error handling untested.

---

### 7. `hybrid/core/receptive_field.py` (291 LOC, 2 commits)

**Risk Score: 7.8/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 2 commits | VERY LOW for critical DSP logic |
| Complexity Estimate | 80 | CORE RF policy gating (hard + advisory) |
| Size | 291 LOC | Large for narrow responsibility; RF calculations only |
| Criticality | CRITICAL | Receptive field must NEVER be wrong (data integrity gate) |
| Test Coverage | 498 test LOC | `test_receptive_field.py` + parity test |
| Last Modified | 2026-09-23 | Not recent |

**Why it's a hotspot:**
- **Low-change/high-consequence:** Only 2 commits across entire RF policy → likely "set it and forget it"
- RF policy has THREE tiers (CORE hard gate, Character advisory, Baked cabinet advisory per CLAUDE.md)
- Duplicated logic in `scripts/train_a2.py` + `cloud/kaggle/train_a2_cloud.py` (parity test catches drift)
- Hard gate abort is ONLY thing that stops training; soft warnings can be missed

**Blind spot:** Advisory-tier overflow (Character, baked cabinet) never gates training in tests; approximation quality in real training unknown.

---

### 8. `routes/continuous_gain.py` (485 LOC, 7 commits)

**Risk Score: 7.7/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 7 commits | Active CG tab feature development |
| Complexity Estimate | 240 | Stateful project management + audio I/O |
| Size | 485 LOC | Largest Flask route file; CG-specific |
| Criticality | HIGH | Continuous Gain tab entry point; 4th workflow mode |
| Test Coverage | 462 + 324 test LOC | `test_cg_project_routes.py` + routes validation |
| Last Modified | 2026-09-24 | Recent |

**Why it's a hotspot:**
- State machine for CG project lifecycle (create→configure→train→export)
- File upload/download boundary (user files); validation required
- Routes directly call `hybrid/continuous_gain/*` modules; contract breaks propagate to UI
- CG tab is "separate workflow" per CLAUDE.md; shares training backend with Hybrid but not design logic

**Blind spot:** Project persistence (`work/sessions/`) not mocked in tests; real session file corruption untested.

---

### 9. `cloud/kaggle/train_a2_cloud.py` (544 LOC, 16 commits)

**Risk Score: 7.6/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 16 commits | Steady churn (Kaggle API changes?) |
| Complexity Estimate | 190 | Self-contained trainer; manifest loading + torch training |
| Size | 544 LOC | Large; runs inside Kaggle kernel |
| Criticality | CRITICAL | Only path to cloud A2 training (code payload) |
| Test Coverage | Loaded as module in tests | `test_receptive_field_parity.py` asserts RF logic match |
| Last Modified | 2026-09-25 | Recent |

**Why it's a hotspot:**
- Runs in Kaggle environment (Python 3.10+, torch, NAM); different constraints than local
- Duplicated `check_receptive_field` with `scripts/train_a2.py`; parity test catches divergence but is manual
- Manifest validation on arrival; bad bundles can still reach kernel (network risk)
- Output validation on return (Full + Lite render comparison) is last gate

**Blind spot:** Kaggle environment specifics (memory limits, job timeout) not reproduc in tests; cold-start timing unknown.

---

### 10. `hybrid/modes/character_blend.py` (389 LOC, 7 commits)

**Risk Score: 7.5/10**

| Signal | Value | Assessment |
|--------|-------|------------|
| Change Frequency | 7 commits | Active Character Blend mode development |
| Complexity Estimate | 120+ | Drive carrier analysis + tone corrections + response sweep |
| Size | 389 LOC | Large; complex DSP mode |
| Criticality | HIGH | Character Blend design mode (1 of 3); newest mode |
| Test Coverage | 533 test LOC | Most comprehensive mode test (`test_character_blend.py`) |
| Last Modified | 2026-09-25 | Recent |

**Why it's a hotspot:**
- Character Blend is "residual-bounded drive carrier + measured tone/feel corrections" (per CLAUDE.md)
- Low-level response sweep is "hard preflight gate"; failure bakes gate into target (data quality issue)
- Newest of three modes; design still stabilizing (7 commits, 533 test lines suggest active tuning)
- Shares training target generation with Fixed Blend (`character_training_target.py`)

**Blind spot:** Real tone correction filter behavior on diverse amp captures untested (synthetic only); live mode switching behavior not full-path tested.

---

## Summary Table

| Rank | File | LOC | Commits | Criticality | Risk | Reason |
|------|------|-----|---------|-------------|------|--------|
| 1 | app.py | 3214 | 90 | CRITICAL | 9.2 | Monolithic routing; no unit test isolation |
| 2 | hybrid/training/kaggle_training.py | 1215 | 12 | CRITICAL | 8.7 | Cloud subprocess + parity requirement |
| 3 | scripts/train_a2.py | 631 | 28 | CRITICAL | 8.5 | RF policy gating + manifest parsing |
| 4 | hybrid/core/pipeline.py | 277 | 7 | CRITICAL | 8.3 | DSP cost boundary; mode dispatch |
| 5 | hybrid/modes/training_target.py | 632 | 6 | CRITICAL | 8.1 | Manifest schema + export naming |
| 6 | hybrid/services/local_llm.py | 1053 | 8 | HIGH | 7.9 | Ollama subprocess; no tests |
| 7 | hybrid/core/receptive_field.py | 291 | 2 | CRITICAL | 7.8 | Low change/high consequence; advisory tiers |
| 8 | routes/continuous_gain.py | 485 | 7 | HIGH | 7.7 | State machine + file I/O boundary |
| 9 | cloud/kaggle/train_a2_cloud.py | 544 | 16 | CRITICAL | 7.6 | Duplicated logic + Kaggle environment |
| 10 | hybrid/modes/character_blend.py | 389 | 7 | HIGH | 7.5 | Newest mode; complex DSP; active development |

## Blind Spots & Low-Change, High-Consequence Areas

### Critical but Rarely Touched

1. **`hybrid/core/receptive_field.py`** (2 commits) — RF policy is load-bearing; only changes if architecture shifts
2. **`hybrid/core/envelope.py`** (2 commits) — Causal envelope for crossover control; design-locked
3. **`hybrid/core/input_profiles.py`** (2 commits) — Pickup gain presets; research-backed; frozen
4. **`hybrid/core/parallel_phase.py`** (2 commits) — Phase alignment in parallel rendering; stable but critical

### Untested Scenarios

1. **Real NAM inference (`hybrid/core/render.py`)** — Tests auto-skip unless native binary built + `.nam` models present
2. **Training environment (`scripts/train_a2.py`)** — Torch/NAM training tests auto-skip unless training environment installed
3. **Streaming AI responses (`hybrid/services/local_llm.py`)** — No unit tests; integration-only if Ollama running
4. **Kaggle environment specifics** — Cold-start, memory limits, timeout behavior not reproduced locally

### Single-Commit High-LOC Files (Risky)

- **`hybrid/services/research.py`** (256 LOC, 1 commit) — Research utilities; frozen code path
- **`hybrid/services/nam_inspector.py`** (206 LOC, 1 commit) — Read-only NAM inspection; stable but one-shot

## Recommendations

1. **Unit test isolation for `app.py`** — Mock Flask/routes; test request handling + state separately
2. **Real-world Kaggle testing** — Run a test submission to catch environment-specific failures
3. **Streaming error tests for LLM** — Simulate connection drops mid-stream; test fallback paths
4. **Advisory RF tier validation** — Test Character Blend + baked cabinet approximation quality with diverse captures
5. **RF parity regression** — Automate drift detection between local + cloud RF logic (currently manual test)

---

**Next review:** After v0.5.5 release; reindex to track volatility trends.
