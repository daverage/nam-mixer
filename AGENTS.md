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
