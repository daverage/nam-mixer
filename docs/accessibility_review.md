# NAM Mixer: blind accessibility review

Reviewed the public `daverage/nam-mixer` repository on 26 September 2026, from a shallow clone of its default branch. This is a source review of the browser interface used by Flask and the Tauri desktop shell. It is not a claim of WCAG conformance or a completed NVDA, JAWS, or VoiceOver user test.

## Changes in v0.5.3

- Added a skip link with an active-panel destination for the Builder and utilities.
- Moved keyboard focus into the first-run dialog, contained Tab within it, and restored focus when dismissed.
- Connected Amp A and Amp B file controls to real labels, and gave the cabinet picker a button role while retaining Enter and Space activation.
- Made the global status line a screen-reader status/alert region.
- Made Continuous Gain's capture picker a visible native file input; named per-file position and Remove controls; named the series selector; exposed the current step; restored focus after its dynamic body is rebuilt.
- Added capture point descriptions to Continuous Gain chart names where point data exists.

## Remaining work, ordered by impact

| Priority | Location | Issue and concrete change |
| --- | --- | --- |
| P0 | `static/cg.js` stage 2 charts | A series of plotted measurements is still exposed mostly as a single image with axes. Add a compact data table for each selected series with gain position, measurement and unit. The mapping chart should likewise expose each position and input gain in a table. Avoid a long ARIA label as a substitute for structured data. |
| P0 | `static/app.js` dynamic controls and results | Test the full path with NVDA/Firefox or Chrome on Windows and VoiceOver/Safari on macOS: choose captures, render, compare, adjust sliders, create, train, export. Check focus after stage changes, results, failures and disabled controls. Source inspection cannot prove the spoken experience. |
| P1 | `templates/index.html` and `static/app.js` Builder workflow | A step switch hides and replaces sections but leaves focus on the step button. Move focus to the new section heading when a keyboard user changes steps; announce a blocked step at the workflow hint. Do the same for utility-tab changes while preserving normal Back behavior. |
| P1 | `static/app.js` and `static/cg.js` live regions | Many large result containers have `aria-live=polite` and are rebuilt, and training logs update repeatedly. Announce a concise status or completion sentence; keep detailed results and logs navigable without reading the whole replacement. Debounce progress announcements to milestones. |
| P1 | `templates/index.html` range controls | Audit all 19 sliders in actual screen readers. Native arrow keys work, but percentages such as `50% B`, dB values, guitar-volume positions and soft/medium/hard context need consistent `aria-valuetext` or native output associations. Do not repeat unchanged values on every unrelated update. |
| P1 | `templates/index.html` modal and generated confirmations | The welcome dialog now traps focus, but any `confirm()` and dynamic modal flow needs keyboard and focus-return testing in Tauri on Windows and macOS. Native dialogs may behave differently in each WebView. |
| P2 | `static/app.js` journey visualization | The canvas has a useful average and threshold in its accessible name, and the coverage table has numeric data. Add a text summary of when the signal reaches Amp A, transition, and Amp B, plus a time-based data table if exact curve exploration matters. The visual playhead is not keyboard inspectable. |
| P2 | `templates/index.html` radio groups | Group the Wizard behavior choices and training presets/backends with `fieldset`/`legend`, or a correctly associated group label. Nearby text alone does not reliably announce the question on each radio. |
| P2 | `static/style.css`, desktop WebViews | Verify focus indicators, zoom to 200–400%, reflow and high-contrast/forced-color modes on Windows. Current focus CSS is promising; visual inspection in each platform is still required. |
| P2 | `docs/user_guide.md` | Add a screen-reader workflow: supported browser/desktop combinations, keyboard navigation, how to use file pickers and comparison audio, meanings of Amp A/B, and where training results and downloads appear. |

## Acceptance checks for a blind user

1. Keyboard and screen reader can start the app, dismiss the welcome guide, skip navigation and identify the current utility and workflow step.
2. Both NAM files and a DI can be selected without pointer input. File, readiness and rendering errors are announced once with a way to recover.
3. Each mode, slider, selected value and audio source has a unique spoken name. The user can compare A, result and B and know which source is playing.
4. Continuous Gain captures can be uploaded, positioned, removed, analysed and selected without drag and drop or chart interpretation.
5. Training progress announces meaningful milestones, does not flood speech with log updates, and finishes with a clearly named export action.
6. Focus remains visible and in the active interface after every panel switch, rerender, modal and error.

## Verification performed

`node --check static/app.js`, `node --check static/cg.js`, `git diff --check`, and `node --test tests/frontend_state.test.cjs` passed (45 tests). No live screen-reader or packaged desktop test was performed.
