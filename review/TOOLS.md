# Source Inspection Summary

## Build & Test Status

**✅ Test Suite: PASSING**
- **1005 tests passed** | 12 skipped | 3 warnings
- Execution time: 69.38s (0:01:09)
- Platform: darwin -- Python 3.11.9, pytest-9.1.1
- Exit code: 0 (success)
- Configuration: pytest.ini, .pytest_cache

**Deprecation Warnings** (non-fatal, from audioread dependencies):
- `aifc`, `audioop`, `sunau` modules flagged for Python 3.13 removal in three test contexts

## Project Metrics

| Metric | Count |
|--------|-------|
| **Python source files** | 136 |
| **Production lines of code** | 20,565 |
| **Test files** | 61 |
| **Test lines of code** | 14,714 |
| **HTML templates** | 1 |
| **JavaScript/TypeScript files** | 2 |
| **npm packages** | 6 (no vulnerabilities) |
| **Python dependencies** | 43 unique (14 base + 29 training) |

## Dependency Audit

**npm audit:** `found 0 vulnerabilities` ✅

**Python dependencies (base):**
- requirements.txt: 14 packages
- requirements-training.txt: 29 packages (torch, neural-amp-modeler, etc. for A2 training only)
- Managed in: .venv-a2 virtual environment (isolated training environment)

**Combined dependency list:** See `review/tools/requirements-combined.txt`

## Architecture & Entry Points

### Primary Entry Points

1. **Flask Web Application** (`app.py`)
   - Serves `/api/*` endpoints and Flask templates
   - Development server on `http://127.0.0.1:5001/`
   - Routes in `routes/continuous_gain.py`

2. **CLI Scripts** (`scripts/`)
   - `train_a2.py`: Local A2 model training (28 commits, 794 lines)
   - `validate_a2.py`: Training bundle validation
   - `analyze_di.py`: DI signal analysis/fixture generation

3. **Cloud Worker** (`cloud/kaggle/train_a2_cloud.py`)
   - Self-contained Kaggle kernel runner
   - Parity-tested against local trainer

### Source Organization

| Module | Purpose | LOC | Key Files |
|--------|---------|-----|-----------|
| `hybrid/core/` | Audio DSP & inference | ~3000 | render.py, pipeline.py, blend.py, envelope.py, calibration.py, align.py, cab_ir.py, receptive_field.py |
| `hybrid/modes/` | Design modes & training targets | ~2400 | training_target.py (801 lines, 6 commits), character_blend.py (495 lines, 7 commits), fixed_blend.py (296 lines) |
| `hybrid/continuous_gain/` | CG tab (multi-amp gain) | ~1200 | project.py (336 lines), bundle.py (206 lines), selection.py (228 lines) |
| `hybrid/training/` | A2 training pipeline | ~3700 | kaggle_training.py (1561 lines, 12 commits), local_training.py (447 lines), validation.py |
| `hybrid/services/` | Auxiliary services | ~2500 | local_llm.py (1274 lines, 8 commits), settings.py (503 lines), research.py (309 lines) |
| `routes/` | Flask API layer | ~535 | continuous_gain.py only |
| `tests/` | Test suite | ~14,714 | 61 test files, 1005 passing tests |

## Static Analysis Results

### Python Compilation Check
- Status: ✅ All Python files compile successfully
- Scope: `hybrid/**/*.py`, `routes/*.py`, `app.py`, `scripts/*.py`
- Result: No syntax errors detected

### Test Coverage
- **Unit tests**: 1005 passing
- **Integration tests**: Real NAM inference tests auto-skip unless native `nam_render` built + `.nam` model present
- **Training tests**: Auto-skip unless training environment (torch) installed
- **Parity tests**: `tests/test_receptive_field_parity.py` asserts local/cloud receptive-field logic equivalence

## Dependency Edges (Key Imports)

### Core Dependencies
- **Flask**: Web framework (routes, templates, JSON API)
- **numpy, scipy**: Array operations, signal processing (audio DSP core)
- **soundfile**: WAV I/O for audio files
- **audioread**: Metadata/format detection for audio files
- **requests**: HTTP client (update checks, LLM services)
- **pytest**: Test framework

### Optional/Training
- **torch**: PyTorch for A2 training (requirements-training.txt only)
- **neural-amp-modeler**: NAM package for training
- **kaggle**: Kaggle API for cloud training submission
- **ollama**: Local LLM backend (opt-in via services/local_llm.py)

### Native C++ Dependency
- **nam_render** (C++ binary, built from `native/nam_render/`)
  - Links NeuralAmpModelerCore
  - Invoked via subprocess from `hybrid/core/render.py`
  - Required for any NAM inference (render, training)

## File Inventory (Top Contributors by Commit History)

### Highest-Activity Production Files
1. `scripts/train_a2.py` - 28 commits, 794 lines
2. `hybrid/modes/training_target.py` - 6 commits, 801 lines
3. `hybrid/training/kaggle_training.py` - 12 commits, 1561 lines
4. `hybrid/services/local_llm.py` - 8 commits, 1274 lines
5. `hybrid/modes/character_blend.py` - 7 commits, 495 lines
6. `hybrid/core/pipeline.py` - 7 commits, 277 lines
7. `hybrid/training/local_training.py` - 7 commits, 447 lines

### Highest-Activity Test Files
1. `tests/test_kaggle_training.py` - 28 commits, 1876 lines
2. `tests/test_character_blend.py` - 15 commits, 533 lines
3. `tests/test_local_training.py` - 13 commits, 333 lines
4. `tests/test_cg_project_routes.py` - 13 commits, 462 lines

## Configuration Files

| File | Status | Purpose |
|------|--------|---------|
| `requirements.txt` | 14 lines | Base dependencies |
| `requirements-training.txt` | 29 lines | A2 training-only dependencies |
| `package.json` | 6 packages | npm/Node.js (browser build, if any) |
| `package-lock.json` | Present | npm lock file |
| `pytest.ini` | Present | Test configuration |
| `.codebase-memory/` | Indexed | MCP codebase-memory graph (separate index) |

## Recent Changes (HEAD~5 to HEAD)

See `review/tools/git-history-recent.txt` for last 5 commits and affected files.

**Recent commits:**
- `90a4e2a` - Accessibility: quiet result panels, journey text alternative, forced colors
- `5744f9e` - Add chart tables and automated accessibility checks for v0.5.4
- `c533d3b` - Improve keyboard and screen-reader access for v0.5.3
- `6507036` - General Updates
- `d2e0fe3` - Docs: fix stale AI setup, CG Sessions contradiction, README docs index; document Wizard/AI Assistant/TONE3000

## Tool Versions & Scope

| Tool | Version | Command | Notes |
|------|---------|---------|-------|
| pytest | 9.1.1 | `python -m pytest tests/ -v` | 1005 tests, 69.38s execution |
| Python | 3.11.9 | `python --version` | Via .venv-a2 |
| npm audit | latest | `npm audit` | 0 vulnerabilities found |
| py_compile | stdlib | `python -m py_compile *.py` | Syntax validation only |
| git | local | `git log`, `git diff` | Commit history & change tracking |

## Coverage Gaps & Notes

- **mypy/type checking**: Not run in this inspection (type hints not enforced in CI currently)
- **linting (pylint/flake8)**: Not run (not in tooling baseline; use as needed for style review)
- **Real NAM rendering tests**: Auto-skipped unless `native/nam_render` built + sample `.nam` models present
- **Training validation tests**: Auto-skipped unless torch/neural-amp-modeler installed (separate environment)
- **Frontend testing**: JavaScript/browser tests not covered (static/js not linted/tested in this run)

## Generated Tool Outputs

All tool outputs saved under `review/tools/`:

- `pytest-results.txt` - Full test output (1005 tests)
- `npm-audit.txt` - npm security audit (0 vulnerabilities)
- `python-compile.txt` - Python syntax check results
- `requirements-combined.txt` - Deduplicated dependencies across both requirements files
- `git-history-recent.txt` - Recent commits and file changes
- `source-inventory.txt` - Detailed file-by-file line counts and commit history
- `project-stats.txt` - Summary statistics (languages, LOC, test coverage)
- `architecture-summary.txt` - Entry points and module organization
- `file-inventory.sh` - Script used to generate file inventory (reusable)

## Next Steps (Optional Deeper Review)

1. **Type Safety**: Run `mypy hybrid/ routes/ app.py` if adding strict type checking
2. **Linting**: Run `pylint` or `flake8` for style/convention enforcement (not currently baseline)
3. **Frontend**: ESLint for `static/js/` if JavaScript refactoring planned
4. **Dependency Audit**: Use `pip-audit` for Python security scanning (current: npm only)
5. **Native Build**: Verify `native/nam_render/` C++ build status if not already built
6. **Live App**: Run `python app.py` and test Flask routes if changes touch routing

---

**Inspection date:** 2026-09-27  
**Branch:** review/0.5.5ReleaseReview  
**Test result:** ✅ All critical checks pass (1005 tests)  
