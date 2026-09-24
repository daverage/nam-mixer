"""Fetches a prebuilt `nam_render` binary for the current platform.

This is the in-app equivalent of scripts/download_nam_render.sh /
scripts/download_nam_render.ps1 (see native/nam_render/README.md) -- those
require a shell; this lets the Settings page's "Download nam_render" button
do the same thing with one click, so `nam_render` stops being a separate,
manually-fetched install/path the user has to go find. It writes into the
exact `native/nam_render/build/` location `hybrid.core.render.find_nam_render_exe`
already auto-detects, so NAM_RENDER_EXE never needs to be set by hand for
this path.
"""
from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..paths import REPO_ROOT

_REPO = "daverage/nam-mixer"
_RELEASES_API = f"https://api.github.com/repos/{_REPO}/releases"
_DOWNLOAD_TIMEOUT_SECONDS = 60

# (sys.platform, normalised CPU) -> published asset. Windows on ARM runs the
# x64 build under emulation; there is no Intel-Mac or Linux-ARM build.
_ASSET_BY_PLATFORM = {
    ("darwin", "arm64"): "nam_render-macos-arm64",
    ("linux", "x86_64"): "nam_render-linux-x64",
    ("win32", "x86_64"): "nam_render-windows-x64.exe",
    ("win32", "arm64"): "nam_render-windows-x64.exe",
}


class NamRenderDownloadError(RuntimeError):
    """Raised when no prebuilt nam_render binary could be fetched."""


def _machine() -> str:
    """This host's CPU as "arm64"/"x86_64". An x86_64 Python running under
    Rosetta on Apple Silicon still counts as arm64: the native build runs there."""
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    if sys.platform == "darwin" and machine == "x86_64":
        try:
            translated = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                                        capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            translated = ""
        if translated == "1":
            return "arm64"
    return machine


def _asset_name() -> str:
    machine = _machine()
    asset = _ASSET_BY_PLATFORM.get((sys.platform, machine))
    if not asset:
        raise NamRenderDownloadError(
            f"No prebuilt nam_render binary is published for this platform ({sys.platform}, {machine}). "
            "Build it from source instead -- see native/nam_render/README.md."
        )
    return asset


def _latest_tag() -> str:
    # One page of 100: desktop vX.Y.Z releases share this repo and would push
    # the last nam-render-v* tag off the default 30-entry page.
    request = Request(f"{_RELEASES_API}?per_page=100", headers={"Accept": "application/vnd.github+json"})
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
    """The exact location hybrid.core.render.find_nam_render_exe already checks first."""
    filename = "nam_render.exe" if sys.platform == "win32" else "nam_render"
    return REPO_ROOT / "native" / "nam_render" / "build" / filename


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

    if not data:
        raise NamRenderDownloadError("Download failed: GitHub returned an empty file.")

    # Write beside the destination and swap it in, never over the existing
    # binary in place: macOS kills a signed executable that was modified in
    # place, and a render running mid-download must never exec a partial file.
    tmp = dest.with_name(dest.name + ".download")
    tmp.write_bytes(data)
    if sys.platform != "win32":
        os.chmod(tmp, tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, dest)
    return dest
