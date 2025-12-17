$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Builds a GUI app folder + optional installer. CLI entrypoint is `run_cli.py`.
if (!(Test-Path ".\\.venv\\Scripts\\python.exe")) {
  throw "Virtualenv not found at .venv. Create it first (python -m venv .venv) and install requirements."
}

& .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt | Out-Host
& .\\.venv\\Scripts\\python.exe -m pip install pyinstaller | Out-Host

$work = ".\\build_exe"
$dist = ".\\dist_exe"

if (Test-Path $work) {
  try { Remove-Item -Recurse -Force $work } catch { Write-Host "Warning: could not remove $work ($_)"; }
}
if (Test-Path $dist) {
  try { Remove-Item -Recurse -Force $dist } catch { Write-Host "Warning: could not remove $dist ($_)"; }
}

# Build an onedir distribution (folder) - better for packaging DLL-heavy deps.
& .\\.venv\\Scripts\\pyinstaller.exe `
  --noconfirm `
  --onedir `
  --windowed `
  --name slice2solid `
  --workpath $work `
  --distpath $dist `
  --paths src `
  --add-data "src/slice2solid/gui/assets/herb.svg;slice2solid/gui/assets" `
  --collect-all skimage `
  --collect-all trimesh `
  --collect-all shapely `
  --collect-all rtree `
  --collect-all pymeshlab `
  run_gui.py

Write-Host "Built app folder: $PSScriptRoot\\dist_exe\\slice2solid\\"

# Build installer if Inno Setup is installed (ISCC.exe in PATH).
$cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
if ($cmd) {
  $iscc = $cmd.Source
} else {
  $iscc = $null
  $candidates = @(
    "${env:ProgramFiles(x86)}\\Inno Setup 6\\ISCC.exe",
    "$env:ProgramFiles\\Inno Setup 6\\ISCC.exe"
  )
  foreach ($c in $candidates) {
    if (Test-Path $c) { $iscc = $c; break }
  }
}
if (-not $iscc) {
  Write-Host "Inno Setup not found (iscc.exe missing)."
  Write-Host "Install Inno Setup, then run:"
  Write-Host "  iscc.exe tools\\installer\\slice2solid.iss"
  exit 0
}

& $iscc "tools\\installer\\slice2solid.iss" | Out-Host
Write-Host "Installer built (see OutputBaseFilename in tools\\installer\\slice2solid.iss)."
