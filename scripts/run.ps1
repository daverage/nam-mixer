# Quick launch for Windows: activates .venv if present, installs
# dependencies on first run, and starts the Flask dev server -- see "Quick
# start" in README.md. Does not fetch nam_render; run
# scripts/download_nam_render.ps1 once beforehand if you haven't already.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}

$MarkerFile = ".venv\.requirements_installed"
& ".venv\Scripts\Activate.ps1"

if ((-not (Test-Path $MarkerFile)) -or ((Get-Item "requirements.txt").LastWriteTime -gt (Get-Item $MarkerFile).LastWriteTime)) {
    python -m pip install -r requirements.txt
    New-Item -ItemType File -Path $MarkerFile -Force | Out-Null
}

python app.py
