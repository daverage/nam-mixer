# Repository Guidelines

## Project Structure & Module Organization

The Flask entry point is `app.py`; browser assets live in `templates/` and
`static/`. Core DSP, cabinet-IR, design, validation, training, and export
logic is organized under `hybrid/` in subpackages (`core/`, `modes/`,
`continuous_gain/`, `training/`, `services/`); `routes/` holds Flask route
modules registered by `app.py`. Local and Kaggle training helpers are in
`scripts/` and `cloud/kaggle/`. The native `nam_render` CLI is built from
`native/nam_render/`. Tests are under `tests/`, with reusable audio/model
fixtures in `assets/`. Generated sessions, training bundles, and build
outputs belong under `work/` or ignored build directories.

## Build, Test, and Development Commands

- `python3 app.py` starts the local Flask app and browser UI.
- `python3 -m pytest -q` runs the complete Python test suite.
- `node --check static/app.js` checks frontend JavaScript syntax.
- `cmake -B native/nam_render/build -S native/nam_render -DCMAKE_BUILD_TYPE=Release`
  configures the pinned NAMCore renderer.
- `cmake --build native/nam_render/build --config Release --target nam_render`
  builds the single renderer used for conventional and Sequential NAM files.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, descriptive snake_case Python names,
and short focused functions. Keep Flask routes thin; put DSP and provenance
rules in `hybrid/`. JavaScript uses two-space indentation, camelCase
variables/functions, and explicit DOM element names. Avoid committing generated
files, local paths, credentials, or build directories.

## Testing Guidelines

Add pytest tests beside the relevant subsystem (for example,
`tests/test_cab_export_modes.py`). Name tests `test_<behavior>`, prefer
deterministic fixtures, and cover both success and validation/error paths.
Native Sequential compatibility tests require the built renderer and may skip
when optional external prerequisites are unavailable.

## Commit & Pull Request Guidelines

Use concise imperative commit subjects, such as `Fix ...` or `Add ...`.
Pull requests should explain user-visible behavior, list verification commands,
call out renderer or training-environment changes, and include screenshots for
meaningful UI changes. Release tags use `v<major>.<minor>.<patch>`; renderer
asset tags use `nam-render-v<version>`.

## Security & Configuration

Keep API keys and local paths in environment variables or the app’s local
settings file; never commit `.env` files or downloaded credentials. Review
upload, subprocess, and file-serving changes for path validation and secret
leakage before merging.

## Codebase memory usage

Use `codebase-memory-mcp` as the primary source of repository context before searching or inferring from scratch.

When working on an existing codebase:

- Query `codebase-memory-mcp` first for relevant architecture, files, symbols, prior decisions, and known relationships.
- Use it to locate likely implementation areas before performing broad repository searches.
- Reuse established codebase knowledge rather than repeatedly rediscovering the same structure.
- Verify important details against the actual source files before making changes.
- If memory results are incomplete, stale, or conflict with the source code, treat the source code as authoritative.
- After significant architectural discoveries or changes, update codebase memory when the MCP supports doing so.
- Do not invent repository structure, APIs, symbols, or implementation details when they can be retrieved from codebase memory or source.

DONT USE em dashes!

# Project constraints
- [Public contracts, safety or real-time constraints, protected paths]

# Commands
- Build: [command]
- Targeted test: [command]
- Full required checks: [commands]
- Baseline comparison: [filled in at Step 4]

# Definitions
- Consequential: affects a project constraint, a public contract, security,
  data integrity, or output people see or hear.
- Fresh context: a new session or subagent that has not seen the work.
- Independent reviewer: a different model from the one that did the work,
  or a person.

# Retrieval
- Index: review/INDEX.md
- Exclude vendor, generated and build output, media, model weights and large data.
- Save raw tool output to review/tools/; read summaries and relevant excerpts.
- Search symbols before opening source. Read line ranges before whole files.

# Retrieval budget
- Routine question: at most 3 files, 300 source lines, 5 short tool excerpts.
- Cross-module question: at most 8 files, 800 source lines.
- Before exceeding a budget, record the open question and why the next
  read answers it in review/PROGRESS.md.

# Output limits
- Chat replies: at most 400 words and 10 findings. Full detail goes in review/.

# Review sessions
- Write only to review/. Never edit production source during a review session.
- Commit review/ at the end of every session.

# Stop and hand off when
- the same check fails twice after your changes
- the next step needs more than the cross-module retrieval budget
- you cannot state what would confirm or falsify a finding
- a change would touch a project constraint or protected path
Before stopping: commit work in progress to the current branch with a
message starting "WIP:" (or use git stash if committing is not allowed).
Set review/HANDOFF.md status to "blocked" with the reason, the last check
result and the commit hash or stash reference. Never discard uncommitted work.
