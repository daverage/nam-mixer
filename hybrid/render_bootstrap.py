"""Fetches a prebuilt `nam_render` binary for the current platform.

This is the in-app equivalent of scripts/download_nam_render.sh /
scripts/download_nam_render.ps1 (see native/nam_render/README.md) -- those
require a shell; this lets the Settings page's "Download nam_render" button
do the same thing with one click, so `nam_render` stops being a separate,
manually-fetched install/path the user has to go find. It writes into the
exact `native/nam_render/build/` location `hybrid.render.find_nam_render_exe`
already auto-detects, so NAM_RENDER_EXE never needs to be set by hand for
this path.

The standalone desktop app never needs this: its own `nam_render` is already
bundled inside the packaged app (see desktop/build.spec) at build time.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_REPO = "daverage/nam-mixer"
_RELEASES_API = f"https://api.github.com/repos/{_REPO}/releases"
_DOWNLOAD_TIMEOUT_SECONDS = 60

_ASSET_BY_PLATFORM = {
    "darwin": "nam_render-macos-arm64",
    "linux": "nam_render-linux-x64",
    "win32": "nam_render-windows-x64.exe",
}


class NamRenderDownloadError(RuntimeError):
    """Raised when no prebuilt nam_render binary could be fetched."""


def _asset_name() -> str:
    asset = _ASSET_BY_PLATFORM.get(sys.platform)
    if not asset:
        raise NamRenderDownloadError(
            f"No prebuilt nam_render binary is published for this platform ({sys.platform}). "
            "Build it from source instead -- see native/nam_render/README.md."
        )
    return asset


def _latest_tag() -> str:
    request = Request(_RELEASES_API, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            releases = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError) as exc:
        raise NamRenderDownloadError(f"Could not reach GitHub to look up nam_render releases: {exc}") from exc
    for release in releases:
        tag = release.get("tag_name", "")
        if tag.startswith("nam-render-v"):
            return tag
    raise NamRenderDownloadError("No nam_render release found on GitHub.")


def default_dest_path() -> Path:
    """The exact location hybrid.render.find_nam_render_exe already checks first."""
    repo_root = Path(__file__).resolve().parent.parent
    filename = "nam_render.exe" if sys.platform == "win32" else "nam_render"
    return repo_root / "native" / "nam_render" / "build" / filename


def download_prebuilt_nam_render(dest_path: Path | None = None) -> Path:
    """Download the matching prebuilt binary for this OS to `dest_path`.

    Raises NamRenderDownloadError with a message safe to show directly in
    the UI (no stack traces, no credentials -- this hits only GitHub's
    public releases API/CDN, unauthenticated).
    """
    asset = _asset_name()
    tag = _latest_tag()
    dest = dest_path or default_dest_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://github.com/{_REPO}/releases/download/{tag}/{asset}"
    try:
        with urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            data = response.read()
    except (URLError, OSError) as exc:
        raise NamRenderDownloadError(f"Download failed: {exc}") from exc

    dest.write_bytes(data)
    if sys.platform != "win32":
        mode = dest.stat().st_mode
        os.chmod(dest, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dest
