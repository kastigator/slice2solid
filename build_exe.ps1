$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Builds a standalone GUI executable (PyInstaller). CLI entrypoint is `run_cli.py`.
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

& .\\.venv\\Scripts\\pyinstaller.exe `
  --noconfirm `
  --onefile `
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

Write-Host "Built: $PSScriptRoot\\dist_exe\\slice2solid.exe"
