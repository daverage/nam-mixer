#!/usr/bin/env bash
# Create a dedicated virtual environment for A2 training -- docs/phase3.md
# sections 13-14. Does NOT touch the main app's Python environment; the
# runtime app (app.py, hybrid/) must keep working without Torch installed.
#
# macOS/Linux counterpart to scripts/setup_a2_env.ps1 -- same behavior,
# same requirements-training.txt, same deliberately-unpinned Torch install
# step (platform/CUDA vary too much to pick one wheel for everyone).
#
# Usage:
#   scripts/setup_a2_env.sh [-p python3.12] [-d .venv-a2]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_EXE=""
VENV_DIR=".venv-a2"

while getopts "p:d:" opt; do
    case "$opt" in
        p) PYTHON_EXE="$OPTARG" ;;
        d) VENV_DIR="$OPTARG" ;;
        *) echo "Usage: $0 [-p python3.12] [-d .venv-a2]" >&2; exit 1 ;;
    esac
done

if [ -z "$PYTHON_EXE" ]; then
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_EXE="python3.12"
    else
        PYTHON_EXE="python3"
        echo "Warning: python3.12 not found on PATH. Falling back to '$PYTHON_EXE' -- Torch wheel availability for its version is not guaranteed. Pass -p to use a specific interpreter (e.g. -p python3.12)." >&2
    fi
fi

echo "Using interpreter: $PYTHON_EXE"
echo "Creating venv at: $VENV_DIR"

"$PYTHON_EXE" -m venv "$VENV_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements-training.txt

echo ""
echo "Base training package installed. Torch is NOT pinned in requirements-training.txt --"
echo "install the build matching your platform/CUDA, e.g.:"
echo "  $VENV_PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo "(see https://pytorch.org/get-started/locally/ for the current command for your machine,"
echo " including the Apple Silicon/MPS build on macOS)"
echo ""
echo "Then run training with:"
echo "  $VENV_PYTHON scripts/train_a2.py work/a2/<design_id>/training_manifest.json"
