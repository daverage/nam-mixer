#!/usr/bin/env bash
# Downloads a prebuilt nam_render binary (macOS/Linux) from this repo's
# GitHub Releases instead of building from source -- see "Quick start" in
# README.md. Requires no C++ compiler or CMake; just curl.
#
# Usage: scripts/download_nam_render.sh [tag]
#   tag defaults to the latest release whose tag starts with "nam-render-v".
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPO="daverage/nam-mixer"
TAG="${1:-}"

case "$(uname -s)" in
    Darwin) ASSET="nam_render-macos-arm64" ;;
    Linux)  ASSET="nam_render-linux-x64" ;;
    *)
        echo "No prebuilt binary for '$(uname -s)'. Build from source instead -- see native/nam_render/README.md." >&2
        exit 1
        ;;
esac

if [ -z "$TAG" ]; then
    echo "Looking up the latest nam_render release..."
    TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases" \
        | grep -o '"tag_name": *"nam-render-v[^"]*"' \
        | head -1 \
        | sed -E 's/.*"(nam-render-v[^"]+)"/\1/')
    if [ -z "$TAG" ]; then
        echo "No nam_render release found. Build from source instead -- see native/nam_render/README.md." >&2
        exit 1
    fi
fi

URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"
DEST_DIR="native/nam_render/build"
DEST="$DEST_DIR/nam_render"

echo "Downloading $ASSET from release $TAG..."
mkdir -p "$DEST_DIR"
curl -fL --progress-bar -o "$DEST" "$URL"
chmod +x "$DEST"

echo ""
echo "Installed: $DEST"
echo "Run 'python3 app.py' -- the app finds this automatically."
