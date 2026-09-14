#!/usr/bin/env bash
# Quick launch for macOS/Linux: activates .venv if present, installs
# dependencies on first run, and starts the Flask dev server -- see "Quick
# start" in README.md. Does not fetch nam_render; run
# scripts/download_nam_render.sh once beforehand if you haven't already.
#
# Usage: scripts/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.requirements_installed" ] || [ "requirements.txt" -nt ".venv/.requirements_installed" ]; then
    python3 -m pip install -r requirements.txt
    touch ".venv/.requirements_installed"
fi

exec python3 app.py
