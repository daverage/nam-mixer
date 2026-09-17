#!/usr/bin/env bash
# Quick launch for macOS/Linux: activates .venv if present, installs
# dependencies on first run, opens the browser, and starts the Flask dev
# server -- see "Quick start" in README.md. Does not fetch nam_render; run
# scripts/download_nam_render.sh once beforehand if you haven't already.
#
# Usage: scripts/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load machine-local settings (such as the optional Ollama recipe assistant)
# without committing them. Values already exported by the shell take priority.
if [ -f ".env" ]; then
    while IFS='=' read -r name value; do
        case "$name" in
            ""|\#*) continue ;;
        esac
        if [ -z "${!name+x}" ]; then
            export "$name=$value"
        fi
    done < ".env"
fi

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

# Default to 5001, not 5000 -- macOS's AirPlay Receiver squats on 5000 and
# silently 403s every request instead of refusing the connection.
PORT="${PORT:-5001}"
export PORT

url="http://127.0.0.1:${PORT}/"
echo "Opening ${url}"

if [ "${HYBRID_NAM_NO_BROWSER:-}" != "1" ]; then
    (
        sleep 1
        if command -v open >/dev/null 2>&1; then
            open "$url"
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" >/dev/null 2>&1 || true
        fi
    ) &
fi

exec python3 app.py
