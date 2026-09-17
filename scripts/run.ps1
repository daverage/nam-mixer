# Quick launch for Windows: activates .venv if present, installs
# dependencies on first run, opens the browser, and starts the Flask dev
# server -- see "Quick start" in README.md. Does not fetch nam_render; run
# scripts/download_nam_render.ps1 once beforehand if you haven't already.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Load machine-local settings (such as the optional Ollama recipe assistant).
# Existing PowerShell environment variables take priority over .env values.
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#\s][^=\s]*)\s*=\s*(.*?)\s*$') {
            $Name = $Matches[1]
            $Value = $Matches[2].Trim('"').Trim("'")
            if (-not (Test-Path "Env:$Name")) {
                Set-Item -Path "Env:$Name" -Value $Value
            }
        }
    }
}

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

if (-not $env:PORT) {
    $SelectedPort = $null
    foreach ($CandidatePort in 5000..5010) {
        $Listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse("127.0.0.1"), $CandidatePort)
        try {
            $Listener.Start()
            $SelectedPort = $CandidatePort
            break
        } catch {
            continue
        } finally {
            $Listener.Stop()
        }
    }

    if (-not $SelectedPort) {
        throw "no free port found from 5000 to 5010"
    }

    $env:PORT = "$SelectedPort"
}

$Url = "http://127.0.0.1:$env:PORT/"
Write-Host "Opening $Url"

if ($env:HYBRID_NAM_NO_BROWSER -ne "1") {
    Start-Job -ScriptBlock {
        param($BrowserUrl)
        Start-Sleep -Seconds 1
        Start-Process $BrowserUrl
    } -ArgumentList $Url | Out-Null
}

python app.py
