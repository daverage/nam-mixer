# Settings page and Advanced controls refactor plan

## Goal

Make the Settings page the complete, understandable UI for supported user-facing
environment variables while preserving the existing `.env` storage, provider
scoping, security rules, legacy variable compatibility, and application APIs.
The page should be consistent from top to bottom and require one deliberate
save action at most; values that can safely take effect immediately should do so
without a separate reload action.

## Current-state findings

- The current registry exposes the core rendering, provider, AI connection,
  temperature, timeout, token-limit, experimental-architecture, and TONE3000
  settings.
- The AI implementation still supports additional tuning variables that are
  not currently represented in the UI:
  - `NAM_MIXER_AI_HISTORY_MESSAGES`
  - `NAM_MIXER_AI_HISTORY_MESSAGE_CHARS`
  - `NAM_MIXER_AI_RESEARCH_CHARS`
  - `NAM_MIXER_LOCAL_LLM_MAX_EXPLANATION_CHARS`
  - `NAM_MIXER_LOCAL_LLM_MAX_REPLY_CHARS`
  - legacy `NAM_MIXER_LOCAL_LLM_HISTORY_MESSAGES`
  - legacy `NAM_MIXER_LOCAL_LLM_HISTORY_MESSAGE_CHARS`
  - legacy `NAM_MIXER_LOCAL_LLM_RESEARCH_CHARS`
  - legacy `NAM_MIXER_LOCAL_LLM_MAX_EXPLANATION_CHARS`
  - legacy `NAM_MIXER_LOCAL_LLM_MAX_REPLY_CHARS`
  - legacy `NAM_MIXER_LOCAL_LLM_MAX_TOKENS`
- `NAM_RENDER_SEQUENTIAL_EXE` is intentionally a developer/compatibility
  override and should remain environment-only unless a later architecture
  decision explicitly makes it user-facing.
- Settings currently mix top-level utility blocks, provider controls, dynamic
  connection actions, and footer save/reload actions. The refactor should give
  every group the same label, description, field-row, validation, and action
  treatment.

## Proposed information architecture

1. **Getting started / Updates** remain utility sections, visually distinct from
   editable configuration.
2. **Rendering** contains the render executable and any other supported normal
   rendering controls.
3. **AI Assistant** contains provider selection and provider-scoped connection
   fields (base URL, account ID, model, API token), followed by general runtime
   controls (temperature and timeout).
4. **Advanced** is one collapsed, clearly labelled section containing:
   - AI response token limit;
   - history message count and per-message character cap;
   - research-note character cap;
   - maximum reply and explanation character caps;
   - experimental NAM architecture enablement.
5. **TONE3000** contains its credential and any future TONE3000-only options.

Advanced fields should expose plain-language descriptions, safe ranges, units,
and an indication that changing caps can affect latency, truncation, or model
context. Internal/legacy aliases should not appear as duplicate rows; the UI
should edit the canonical `NAM_MIXER_AI_*` names while the server continues to
read and migrate legacy `NAM_MIXER_LOCAL_LLM_*` values.

## UX and layout work

- Establish one reusable settings-group layout: heading/summary, short help
  text, aligned label/input/description rows, inline validation, and a consistent
  action area.
- Keep Advanced collapsed by default, but preserve its open state while editing
  and after validation errors.
- Use a two-column label/control layout on wide screens and a single-column
  layout on narrow screens. Keep control widths and spacing consistent for text,
  number, select, checkbox, secret, and path fields.
- Put dependent actions after all related fields. The AI **Test connection**
  action must be the final action in the AI Assistant group, after provider,
  endpoint, model, credentials, timeout, token limit, and related caps are
  visible. Any future action requiring a text field follows that field group.
- Keep secrets write-only; never echo their values into the page, debug output,
  or browser responses.
- Show the effective behavior clearly: values are stored locally, provider
  switching is provider-scoped, and restart-required settings are marked.

## Save, autosave, and reload behavior

- Replace the current “Save settings” plus “Reload” workflow with a single
  **Save settings** action as the durable commit point.
- Maintain a client-side draft while editing. Do not write on every keystroke.
- On save, submit the complete editable settings payload once, validate and
  persist it server-side, then refresh the returned canonical values and
  provider visibility. This keeps one save sufficient and prevents partial
  provider configuration.
- Automatically refresh dependent UI and runtime status after a successful
  save. No manual reload should be needed for settings that are read per
  request; retain the existing restart warning for startup-only settings.
- Keep a small **Discard changes** action only if needed to restore the last
  server-loaded draft; do not retain a separate reload button unless testing
  demonstrates a real recovery use case.
- Provider changes may continue to save immediately because they change which
  scoped fields are shown, but they should use the same save/status pattern and
  never erase another provider’s stored values.

## Server and compatibility changes

- Add canonical `SettingField` entries and metadata for every user-facing AI
  tuning variable, including bounded number ranges and steps.
- Preserve `_setting()` fallback behavior and legacy aliases. Existing `.env`
  files must continue to work without manual migration.
- Keep localhost-only HTTP rules, HTTPS/public-host validation, credential
  isolation, secret-clearing behavior, and response contracts unchanged.
- Return field metadata (`min`, `max`, `step`, restart requirement, provider
  visibility, and secret state) from `/api/settings` so the browser does not
  duplicate validation rules.
- Validate submitted numeric values server-side with the same bounds used by the
  runtime; reject or clamp only according to the existing setting semantics and
  document the behavior.

## Verification plan

### Settings/API tests

- Every canonical AI environment variable appears exactly once in the settings
  response and is assigned to the intended group.
- Advanced fields expose correct ranges, steps, descriptions, and collapsed
  grouping metadata.
- Legacy `.env` names populate canonical fields and survive provider switching.
- Complete-save behavior preserves unrelated fields and provider-scoped secrets.
- Blank secrets remain unchanged unless explicitly cleared.
- Invalid numeric values cannot bypass server-side bounds.

### Browser/UI checks

- Render settings with every field type and verify aligned wide/narrow layouts.
- Confirm AI Test connection is last in its group and cannot run while required
  fields are incomplete.
- Confirm save refreshes values and provider visibility without a manual reload.
- Confirm restart-required notices remain visible and autosafe fields update
  immediately.
- Run `node --check static/app.js` and, where available, a browser smoke test.

### Regression checks

- Run the focused settings, AI-provider, app-route, and security tests.
- Run the full Python suite and report any unrelated collection failures
  separately.
- Run `git diff --check` before implementation is handed off.

## Implementation order

1. Expand the canonical settings registry and server metadata/tests.
2. Refactor settings-group rendering and validation metadata consumption.
3. Reorder AI actions and replace save/reload with draft-save-refresh behavior.
4. Add responsive styling and accessibility checks.
5. Run focused and full regression tests, then update user documentation.

