# Downloads a prebuilt nam_render.exe (Windows) from this repo's GitHub
# Releases instead of building from source -- see "Quick start" in
# README.md. Requires no Visual Studio Build Tools or CMake.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/download_nam_render.ps1 [-Tag nam-render-v1]

param(
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Repo = "daverage/nam-mixer"
$Asset = "nam_render-windows-x64.exe"

if ([string]::IsNullOrEmpty($Tag)) {
    Write-Host "Looking up the latest nam_render release..."
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases"
    $release = $releases | Where-Object { $_.tag_name -like "nam-render-v*" } | Select-Object -First 1
    if (-not $release) {
        Write-Error "No nam_render release found. Build from source instead -- see native/nam_render/README.md."
        exit 1
    }
    $Tag = $release.tag_name
}

$Url = "https://github.com/$Repo/releases/download/$Tag/$Asset"
$DestDir = "native/nam_render/build/Release"
$Dest = Join-Path $DestDir "nam_render.exe"

Write-Host "Downloading $Asset from release $Tag..."
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
Invoke-WebRequest -Uri $Url -OutFile $Dest

Write-Host ""
Write-Host "Installed: $Dest"
Write-Host "Run 'python app.py' -- the app finds this automatically."
