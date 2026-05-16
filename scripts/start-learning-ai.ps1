[CmdletBinding()]
param(
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command make -ErrorAction SilentlyContinue)) {
  throw "Could not find 'make' on PATH. Install GNU Make, then run Learning AI again."
}

if ($DryRun) {
  Write-Host "Learning AI launcher is ready."
  Write-Host "Repository: $repoRoot"
  Write-Host "Command: make start"
  return
}

Write-Host "Starting Learning AI"
Write-Host "Repository: $repoRoot"
Write-Host "Command: make start"

& make start

if ($LASTEXITCODE -ne 0) {
  throw "'make start' exited with code $LASTEXITCODE."
}
