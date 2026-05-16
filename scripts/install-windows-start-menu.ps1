[CmdletBinding(SupportsShouldProcess = $true)]
param(
  [string]$AppName = "Learning AI",
  [switch]$Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
  throw "The Learning AI Start Menu installer only supports Windows."
}

if ([string]::IsNullOrWhiteSpace($env:APPDATA)) {
  throw "APPDATA is not set, so the user Start Menu folder cannot be located."
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$launcherPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "start-learning-ai.ps1")).Path
$startMenuPrograms = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$shortcutPath = Join-Path $startMenuPrograms "$AppName.lnk"

if ($Uninstall) {
  if (-not (Test-Path -LiteralPath $shortcutPath)) {
    Write-Host "No Start Menu shortcut found at: $shortcutPath"
    return
  }

  if ($PSCmdlet.ShouldProcess($shortcutPath, "Remove Start Menu shortcut")) {
    Remove-Item -LiteralPath $shortcutPath -Force
    Write-Host "Removed Start Menu shortcut: $shortcutPath"
  }

  return
}

if ($PSCmdlet.ShouldProcess($shortcutPath, "Install Start Menu shortcut")) {
  New-Item -ItemType Directory -Path $startMenuPrograms -Force | Out-Null

  $powerShellPath = (Get-Command powershell.exe -ErrorAction Stop).Source
  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($shortcutPath)
  $shortcut.TargetPath = $powerShellPath
  $shortcut.Arguments = "-NoExit -ExecutionPolicy Bypass -File `"$launcherPath`""
  $shortcut.WorkingDirectory = $repoRoot
  $shortcut.Description = "Start Learning AI by running make start."

  $iconPath = Join-Path $repoRoot "app\favicon.ico"
  if (Test-Path -LiteralPath $iconPath) {
    $shortcut.IconLocation = $iconPath
  }

  $shortcut.Save()

  Write-Host "Installed Start Menu shortcut: $shortcutPath"
  Write-Host "Launch it by pressing Windows, typing '$AppName', and pressing Enter."
}
