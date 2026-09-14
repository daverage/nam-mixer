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

if [ -z "${PORT:-}" ]; then
    PORT="$(
        python3 - <<'PY'
import socket

for port in range(5000, 5011):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("no free port found from 5000 to 5010")
PY
    )"
    export PORT
fi

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
