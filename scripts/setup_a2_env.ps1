# Create a dedicated virtual environment for A2 training -- docs/phase3.md
# sections 13-14. Does NOT touch the main app's Python environment; the
# runtime app (app.py, hybrid/) must keep working without Torch installed.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_a2_env.ps1 [-PythonExe python3.12] [-VenvDir .venv-a2]
#
# Picks whatever Python interpreter you point it at (default: tries
# `py -3.12`, falling back to `python`) -- it does NOT hard-code a
# machine-specific install path. If your machine's main Python is too new for
# published Torch wheels (this repo's dev machine is on Python 3.14, which
# is), install Python 3.12 separately and pass -PythonExe explicitly.

param(
    [string]$PythonExe = "",
    [string]$VenvDir = ".venv-a2"
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrEmpty($PythonExe)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.12 -c "print(1)" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $PythonExe = "py -3.12"
        }
    }
    if ([string]::IsNullOrEmpty($PythonExe)) {
        $PythonExe = "python"
        Write-Warning "Could not find Python 3.12 via the 'py' launcher. Falling back to '$PythonExe' -- Torch wheel availability for its version is not guaranteed. Pass -PythonExe explicitly to use a specific interpreter."
    }
}

Write-Host "Using interpreter: $PythonExe"
Write-Host "Creating venv at: $VenvDir"

Invoke-Expression "$PythonExe -m venv `"$VenvDir`""
if ($LASTEXITCODE -ne 0) {
    Write-Error "venv creation failed."
    exit 1
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements-training.txt

Write-Host ""
Write-Host "Base training package installed. Torch is NOT pinned in requirements-training.txt --"
Write-Host "install the build matching your platform/CUDA, e.g.:"
Write-Host "  $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
Write-Host "(see https://pytorch.org/get-started/locally/ for the current command for your machine)"
Write-Host ""
Write-Host "Then run training with:"
Write-Host "  $VenvPython scripts/train_a2.py work/a2/<design_id>/training_manifest.json"
