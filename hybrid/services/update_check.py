"""Checks GitHub for a newer app release -- manual/opt-in only (see app.py's /api/update/check).

No telemetry: nothing is reported TO GitHub about this install, and this is never called automatically -- only in
response to the Settings page's "Check for updates" button, matching the app's local-first/no-background-network
policy (README's "Local-first & private"). It shares the same unauthenticated GitHub releases API and error style
as hybrid/core/render_bootstrap.py's nam_render downloader, but looks at the app's own `v<major>.<minor>.<patch>` tags
rather than that module's separate `nam-render-v*` tag namespace.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen

_REPO = "daverage/nam-mixer"
_RELEASES_API = f"https://api.github.com/repos/{_REPO}/releases"
_REQUEST_TIMEOUT_SECONDS = 15
_VERSION_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

# The fixed installer names the release workflow publishes (.github/workflows/build-desktop.yml's
# "Give the installers stable names" step) -- kept here as plain data, one per platform this app runs on;
# each is a direct-download URL for a specific release once the tag is known.
_ASSET_BY_PLATFORM = {
    "darwin": "NAM-Mixer-macOS-arm64.dmg",
    "linux": "NAM-Mixer-Linux-x64.AppImage",
    "win32": "NAM-Mixer-Windows-x64-setup.exe",
}


class UpdateCheckError(RuntimeError):
    """Raised when GitHub couldn't be reached or returned nothing usable -- message is safe to show in the UI."""


@dataclass(frozen=True)
class UpdateCheckResult:
    current_version: str
    latest_version: str
    update_available: bool
    release_url: str
    asset_url: "str | None"  # a direct installer download for THIS platform, if the release publishes one


def _parse_version(tag: str) -> "tuple[int, int, int] | None":
    match = _VERSION_TAG_RE.match(tag.strip())
    return tuple(int(part) for part in match.groups()) if match else None  # type: ignore[return-value]


def _fetch_releases() -> list[dict]:
    request = Request(_RELEASES_API, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise UpdateCheckError(f"Could not reach GitHub to check for updates: {exc}") from exc


def check_for_update(current_version: str, *, platform: str = sys.platform) -> UpdateCheckResult:
    """Compare `current_version` (an app APP_VERSION string, "v0.3.2") against the newest `v<major>.<minor>.<patch>`
    tag on GitHub. Releases (like nam_render's own) may include OTHER tag namespaces in the same list -- anything
    not matching that exact shape is ignored rather than misread as a version.

    Raises UpdateCheckError if GitHub can't be reached, or if no app-version release exists at all.
    """
    current = _parse_version(current_version)
    if current is None:
        raise UpdateCheckError(f"Could not parse this app's own version string ({current_version!r}).")

    best: "tuple[int, int, int] | None" = None
    best_release: dict | None = None
    for release in _fetch_releases():
        parsed = _parse_version(str(release.get("tag_name", "")))
        if parsed is None or release.get("draft") or release.get("prerelease"):
            continue
        if best is None or parsed > best:
            best, best_release = parsed, release

    if best is None or best_release is None:
        raise UpdateCheckError("No published app release was found on GitHub.")

    asset_name = _ASSET_BY_PLATFORM.get(platform)
    asset_url = None
    if asset_name:
        for asset in best_release.get("assets") or []:
            if asset.get("name") == asset_name:
                asset_url = asset.get("browser_download_url")
                break

    return UpdateCheckResult(
        current_version=current_version,
        latest_version=best_release.get("tag_name", ""),
        update_available=best > current,
        release_url=best_release.get("html_url") or f"https://github.com/{_REPO}/releases/tag/{best_release.get('tag_name', '')}",
        asset_url=asset_url,
    )
