# sanitize.css (vendored)

Vendored locally instead of loaded from a CDN, so the app has no runtime
dependency on an external host for its CSS reset/normalize layer -- matters
for the standalone desktop app (`desktop/`), which should work fully
offline.

- Source: https://csstools.github.io/sanitize.css/
- Version: 13.0.0
- Files: `sanitize.css`, `typography.css`, `forms.css` (unmodified, fetched
  from `unpkg.com/sanitize.css@13.0.0`)
- License: CC0 1.0 (see https://github.com/csstools/sanitize.css/blob/main/LICENSE.md)

To update: re-download the three files above at the desired version and
replace them here; `templates/index.html` references them by local path
and needs no changes.
