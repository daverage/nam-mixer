# UI/UX review — findings and fix plan

Reviewed 2026-09-06 against `templates/index.html`, `static/style.css`, `static/app.js`.

## Consistency
1. Step numbering skips 4 and leaves three main-column cards unnumbered — renumber all cards 1-7 in reading order (Amps&Input, Crossover&Transition, Level match, Listen, Journey, Coverage, Create A2).
2. "Generate Training Bundle" is `btn-secondary` despite being the card's primary action — promote to `btn-primary`.
3. Trim readout duplicates the value chip's manual-tweak number — trim it to auto match + effective trim only.
4. "Create A2" card is visually muted (`card-muted`) despite being the end goal — remove muting.
5. Inconsistent gain/trim terminology across slider labels — standardize on "gain"/"trim" wording.

## Accessibility
6. Journey canvas has no text/ARIA equivalent for screen readers — add `role="img"` + dynamic `aria-label` summary.
7. Status/result panels lack `aria-live` — add `polite` (status/result) or `assertive` (warning boxes) to render-status, render-warnings, generate-status, generate-result, coverage-warning, kaggle-status, kaggle-progress-state, kaggle-result.
8. Color-only Amp A/B distinction — already backed by text labels; no code change needed beyond confirming canvas draws with paired text/shape (deferred, no action here).
9. `reference-dbu-input` not linked to its explanatory text (`aria-describedby`); transition preset buttons have no active/pressed state — add both.
10. Focus-ring clipping risk on tightly-packed preset/link buttons — spot-checked, no clipping found; no change needed.
11. `--text-faint` token is defined but unused and fails contrast if it were ever used for text — remove the dead token.

## Intuitiveness / IA
12. No visual separation between "diagnostics" (Listen/Journey/Coverage) and "export" (Create A2) — add small group headings.
13. Test-gain slider triggers real re-inference from within the "Listen" card without an in-the-moment busy indicator — dim/disable the player and show a re-rendering hint while it's in flight.
14. Calibration warning text is duplicated verbatim between render-warnings and the generate-result bundle box — shorten the second occurrence to a cross-reference.
15. Coverage table has no link back to the currently-selected input profile — highlight the matching row.
16. Empty-state idiom for canvas/table vs. disabled-button idiom elsewhere — intentional/acceptable difference; no change.

## Fix order
1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15 — all except 8, 10, 16 (documented as no-op).
