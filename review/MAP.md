# Architecture Map

**NAM Hybrid Builder** is a Flask web app + Python audio DSP library for blending two NAM (Neural Amp Modeler) captures into dynamic A2 training targets. Three design modes (Dynamic Hybrid, Parallel Blend, Character Blend) share core DSP; a fourth workflow (Continuous Gain) chains multiple fixed-gain captures into one model.

## Layers & Responsibilities

### Entry Point: Flask Web Layer (app.py, 3214 LOC, 90 commits)

Single monolithic app file; routes all `/api/*` endpoints + static/template serving. High volatility (90 commits). No dependency isolation; logic intertwined with Flask routing. Registers route modules from `routes/continuous_gain.py`. Key entry points:
- `/api/render_pair` (expensive: runs NAM inference via subprocess)
- `/api/generate` (training target generation via `hybrid/modes/training_target.py`)
- `/api/kaggle/*` (cloud training job submission/polling)

**State model:** Request-response; session state held in browser + `work/sessions/` files (project persistence for CG tab).

### Core DSP Library (hybrid/, ~11k LOC across subpackages)

Pure Python audio processing with no Flask dependencies. Five subpackages, each with clear responsibility:

#### hybrid/core/ (18 files, ~2.8k LOC)
Low-level audio DSP: NAM loading, rendering, envelope calculation, blending, alignment, calibration.

| Responsibility | File | LOC | Commits |
|---|---|---|---|
| NAM I/O | nam_loader.py | 115 | 2 |
| Rendering (subprocess to nam_render) | render.py | 143 | 5 |
| Envelope (causal, for crossfade control) | envelope.py | 179 | 2 |
| Level matching (trim auto-gain) | level_match.py | 59 | 2 |
| Alignment detection & application | align.py, align_diagnostic.py, align_verification.py | 474 | 9 |
| Blending (smoothstep crossfade) | blend.py | 58 | 1 |
| Calibration (input level compensation) | calibration.py | 69 | 2 |
| Cabinet IR (FIR convolution, baked or preview) | cab_ir.py | 360 | 4 |
| Receptive field gating (CORE hard gate, advisory tiers) | receptive_field.py | 291 | 2 |
| Pipeline coordinator (render_pair → build_hybrid) | pipeline.py | 225 | 7 |

**Design:** `render_pair()` is expensive (NAM inference); `build_hybrid()` is cheap (numpy). Cost boundary strictly enforced via `RenderedPair` dataclass.

#### hybrid/modes/ (12 files, ~2.6k LOC)
Three design modes + shared training target generators.

| Mode | Core | Training | Test LOC |
|---|---|---|---|
| Dynamic Hybrid (level-driven crossfade) | blend.py (58 LOC) | training_target.py (632 LOC) | 427 |
| Parallel Blend (fixed ratio) | fixed_blend.py (237 LOC) | blend_training_target.py (201 LOC) | 185 |
| Character Blend (drive carrier + corrections) | character_blend.py (389 LOC) | character_training_target.py (289 LOC) | 533 |

All share `design.py` (145 LOC) and `metadata.py` (63 LOC) for provenance records. Manifest schema is versioned + hashed.

#### hybrid/continuous_gain/ (12 files, ~1.2k LOC)
CG tab: one amp, N fixed-gain captures → one `.nam`. Separate workflow, shares training backend.

| Component | LOC | Purpose |
|---|---|---|
| project.py | 297 | State machine (create→configure→train→export) |
| selection.py | 203 | Capture selection (Phase 4D anchors, response-distance) |
| bundle.py | 169 | Training audio assembly |
| audit.py | 183 | Music lag detection (auto-align captures) |
| validation.py | 140 | CG result validation (Full/Lite render) |

**State:** Project files in `work/sessions/` (persistent); stateful file I/O boundaries at routes.

#### hybrid/training/ (11 files, ~3.2k LOC)
A2 training orchestration: local torch or Kaggle GPU.

| Component | LOC | Commits | Purpose |
|---|---|---|---|
| local_training.py | 341 | 7 | Torch training subprocess + epoch polling |
| kaggle_training.py | 1215 | 12 | Kaggle CLI wrapper + job state machine |
| validation.py | 166 | 6 | Full/Lite render + ESR comparison |
| nam_provenance.py | 147 | 6 | Export naming (applies per CLAUDE.md rules) |
| a2_training_settings.py | 117 | 6 | Hyperparameters + official V3 input MD5 |
| sequential_nam.py | 187 | 5 | Sequential export mode (experimental) |

**Concurrency:** Subprocess management (spawn, poll, wait). Kaggle uses background job polling (non-blocking Flask). Local training blocks CLI.

#### hybrid/services/ (8 files, ~1.9k LOC)
Auxiliary: settings, LLM, update check, NAM inspection.

| Service | LOC | Purpose |
|---|---|---|
| local_llm.py | 1053 | Ollama subprocess, streaming responses (AI Assistant feature) |
| settings.py | 439 | App config (merged from .env + UI-set values) |
| research.py | 256 | Analysis utilities (frozen) |
| nam_inspector.py | 206 | Read-only NAM metadata inspection |

**State:** local_llm.py spawns Ollama subprocess; fallback to public endpoint if local fails. No unit tests for streaming error handling.

### CLI & Cloud

| Entry Point | LOC | Commits | Purpose |
|---|---|---|---|
| scripts/train_a2.py | 631 | 28 | Local A2 training entry point; RF policy enforcement |
| cloud/kaggle/train_a2_cloud.py | 544 | 16 | Self-contained Kaggle kernel trainer (duplicates `check_receptive_field`) |
| scripts/validate_a2.py | 138 | 5 | Bundle validation (post-training audit) |

**Parity requirement:** `scripts/train_a2.py` + `cloud/kaggle/train_a2_cloud.py` must maintain identical RF policy logic (tested by `test_receptive_field_parity.py`).

## State & Concurrency Model

**Synchronous request-response:** Flask routes block on slow operations.
- `render_pair()` waits for `nam_render` subprocess (typical: 1-5s per render)
- Cabinet IR convolution (numpy, in-process)

**Background jobs:** Kaggle training (async via job polling in Flask route).

**File persistence:** Sessions and training bundles under `work/` (gitignored). Session files store `nam_mixer-autosave` and named project records.

**Subprocess boundaries:** 
- NAM inference: `hybrid/core/render.py` → `nam_render` binary (mono float32 in/out)
- Training: `hybrid/training/local_training.py` → torch training + NAM CLI validation
- Cloud: `hybrid/training/kaggle_training.py` → `kaggle` CLI (never shell=True)
- LLM: `hybrid/services/local_llm.py` → Ollama subprocess (streaming, fallback to public endpoint)

## Hot Paths & Key Contracts

### Critical Data Flows

1. **Render pair (expensive):** DI audio → `render_pair()` → (apply input profile) → run NAM twice → calibrate → envelope → return `RenderedPair`
2. **Build hybrid (cheap):** `RenderedPair` → align → blend (smoothstep) → level-match → apply cabinet IR → safety ceiling → return audio
3. **Training target (freezing):** Current design state → fold official NAM training excitation through design → freeze → bundle manifest (versioned, hashed)

### Key Invariants

- **RF policy (receptive_field.py):** CORE hard gate aborts if Amp A/B RF + bounded envelope exceeds A2 actual RF. Cabinet IR advisory (overflow reported, never gates).
- **Alignment:** Original/Corrected timing intent stored; Corrected only if next render re-verifies offset exactly.
- **Export naming:** Must state amp configuration per CLAUDE.md: `[Amp Only]`, `[Full Rig]`, `[Learned Cab]`, `[Embedded Cab]`.
- **Causal envelope:** `envelope.py` uses zero-padded windowed sum (never future samples); design-locked.

### High-Volatility Files (Risk)

- **app.py** (90 commits, 3214 LOC): Monolithic routing; any change affects all endpoints
- **scripts/train_a2.py** (28 commits, 631 LOC): RF gating + manifest parsing; frequent fixes
- **kaggle_training.py** (12 commits, 1215 LOC): Cloud subprocess + parity with cloud worker

## Test Coverage & Blind Spots

**1005 passing tests** (69.38s), organized by subsystem:
- Core DSP: ~2.2k test LOC (envelope, blend, align, CAB IR, receptive field)
- Training pipeline: ~2.5k test LOC (A2 settings, Kaggle routes, local training, RF parity)
- CG tab: ~600 test LOC (project routes, selection, validation)
- Character Blend: 533 test LOC (comprehensive mode testing)

**Blind spots:**
- Real NAM inference (`render.py`) tests skip without native binary + sample `.nam` files
- Torch training (`train_a2.py`) tests skip without training environment (mocked otherwise)
- Ollama streaming (`local_llm.py`) has no unit tests; integration-only if Ollama running
- Kaggle environment (memory, cold-start, timeout) not reproduced locally

## Conventions

**Python:** Four-space indent, type hints, snake_case, short focused functions. No comments unless WHY is non-obvious.

**Package structure:** One responsibility per module. Dataclasses for DSP (e.g., `RenderedPair`, `CabDesign`, `BlendDesign`).

**Testing:** `tests/test_<subsystem>.py` beside `hybrid/<subsystem>/`. Prefer synthetic fixtures; skip real tools gracefully (no hard fails).

**Flask:** Routes thin; all logic in `hybrid/`. No business logic in `app.py` route handlers beyond request/response marshalling.

---

**Mapped subsystems:** 64 production files, 20k+ LOC | **Tests:** 60 test files, 14.7k LOC | **Uncertainty:** ⚠ Concurrent/threaded behavior not explicitly modeled in code review; subprocess error modes (Kaggle API, Ollama network) partially untested.

