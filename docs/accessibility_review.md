# NAM Mixer: blind accessibility review

Reviewed the public `daverage/nam-mixer` repository on 26 September 2026, from a shallow clone of its default branch. This is a source review of the browser interface used by Flask and the Tauri desktop shell. It is not a claim of WCAG conformance or a completed NVDA, JAWS, or VoiceOver user test.

## Changes in v0.5.3

- Added a skip link with an active-panel destination for the Builder and utilities.
- Moved keyboard focus into the first-run dialog, contained Tab within it, and restored focus when dismissed.
- Connected Amp A and Amp B file controls to real labels, and gave the cabinet picker a button role while retaining Enter and Space activation.
- Made the global status line a screen-reader status/alert region.
- Made Continuous Gain's capture picker a visible native file input; named per-file position and Remove controls; named the series selector; exposed the current step; restored focus after its dynamic body is rebuilt.
- Added capture point descriptions to Continuous Gain chart names where point data exists.

## Follow-up implementation in v0.5.4

- Added expandable measurement and Input-gain mapping tables for the Continuous Gain charts. The chart now points readers to exact tabular values.
- Keyboard activation of Builder steps and utility panels moves focus to the newly active heading; blocked steps explain the reason in a status region.
- Range controls with visible readouts expose the same value and units to assistive technology. Training presets and backend choices have group legends.
- Removed high-frequency live announcements from elapsed-time displays and raw training logs. Stage changes and errors use concise status messages.
- Added a keyboard guide and browser-level keyboard/ARIA checks in CI. Browser automation cannot verify the quality of spoken output.

## Remaining work, ordered by impact

| Priority | Location | Issue and concrete change |
| --- | --- | --- |
| P0 | `static/app.js` dynamic controls and results | Test the full path with NVDA/Firefox or Chrome on Windows and VoiceOver/Safari on macOS: choose captures, render, compare, adjust sliders, create, train, export. Check focus after stage changes, results, failures and disabled controls. Source inspection cannot prove the spoken experience. |
| P1 | `static/app.js` result panels | Review large AI, search, and validation result containers that still use `aria-live=polite`; announce a short result summary without reading an entire replacement. |
| P1 | `templates/index.html` range controls | Verify the spoken values of all sliders in NVDA and VoiceOver, including percentages, dB, and guitar-volume positions. |
| P1 | `templates/index.html` modal and generated confirmations | The welcome dialog now traps focus, but any `confirm()` and dynamic modal flow needs keyboard and focus-return testing in Tauri on Windows and macOS. Native dialogs may behave differently in each WebView. |
| P2 | `static/app.js` journey visualization | The canvas has a useful average and threshold in its accessible name, and the coverage table has numeric data. Add a text summary of when the signal reaches Amp A, transition, and Amp B, plus a time-based data table if exact curve exploration matters. The visual playhead is not keyboard inspectable. |
| P2 | `static/style.css`, desktop WebViews | Verify focus indicators, zoom to 200–400%, reflow and high-contrast/forced-color modes on Windows. Current focus CSS is promising; visual inspection in each platform is still required. |

## Acceptance checks for a blind user

1. Keyboard and screen reader can start the app, dismiss the welcome guide, skip navigation and identify the current utility and workflow step.
2. Both NAM files and a DI can be selected without pointer input. File, readiness and rendering errors are announced once with a way to recover.
3. Each mode, slider, selected value and audio source has a unique spoken name. The user can compare A, result and B and know which source is playing.
4. Continuous Gain captures can be uploaded, positioned, removed, analysed and selected without drag and drop or chart interpretation.
5. Training progress announces meaningful milestones, does not flood speech with log updates, and finishes with a clearly named export action.
6. Focus remains visible and in the active interface after every panel switch, rerender, modal and error.

## Verification performed

`node --check static/app.js`, `node --check static/cg.js`, `git diff --check`, and `node --test tests/frontend_state.test.cjs` passed (45 tests). No live screen-reader or packaged desktop test was performed.
