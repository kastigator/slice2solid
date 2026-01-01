$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
  throw "Python venv not found: $python. Create it first (see README.md)."
}

& $python (Join-Path $PSScriptRoot "run_gui.py")
