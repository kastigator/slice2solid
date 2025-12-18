$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Invoke-External {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed (exit code $LASTEXITCODE): $FilePath $($Arguments -join ' ')"
  }
}

function Remove-TreeWithRetry {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Retries = 6,
    [int]$SleepMs = 500
  )
  if (!(Test-Path $Path)) { return }
  for ($i = 1; $i -le $Retries; $i++) {
    try {
      Remove-Item -Recurse -Force $Path -ErrorAction Stop
      break
    } catch {
      if ($i -eq $Retries) { throw }
      Start-Sleep -Milliseconds $SleepMs
    }
  }
  if (Test-Path $Path) {
    throw "Could not remove '$Path' (likely locked by another process). Close slice2solid / Explorer preview / antivirus scan and retry."
  }
}

# Builds a GUI app folder + optional installer. CLI entrypoint is `run_cli.py`.
if (!(Test-Path ".\\.venv\\Scripts\\python.exe")) {
  Write-Host "Virtualenv not found at .venv; creating it..."
  Invoke-External python -m venv .venv
  if (!(Test-Path ".\\.venv\\Scripts\\python.exe")) {
    throw "Failed to create virtualenv at .venv"
  }
}

& .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt | Out-Host
& .\\.venv\\Scripts\\python.exe -m pip install pyinstaller | Out-Host

$work = ".\\build_exe"
$dist = ".\\dist_exe"
$versionFile = ".\\src\\slice2solid\\__init__.py"
$appVersion = $null
if (Test-Path $versionFile) {
  $matchInfo = Select-String -Path $versionFile -Pattern '^__version__\s*=\s*"([^"]+)"' -AllMatches | Select-Object -First 1
  if ($matchInfo -and $matchInfo.Matches.Count -gt 0) { $appVersion = $matchInfo.Matches[0].Groups[1].Value.Trim() }
}

Remove-TreeWithRetry -Path $work
Remove-TreeWithRetry -Path $dist

# Build an onedir distribution (folder) - better for packaging DLL-heavy deps.
Invoke-External .\\.venv\\Scripts\\pyinstaller.exe `
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

if ($appVersion) {
  Write-Host "Building installer with AppVersion=$appVersion"
  $outDir = "tools\\installer\\Output"
  $outExe = Join-Path $outDir "slice2solid-setup-v$appVersion.exe"
  if (Test-Path $outDir) {
    try {
      Get-ChildItem -Path $outDir -Filter "slice2solid-setup*.exe" -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction Stop
    } catch {
      throw "Cannot clean '$outDir' (locked). Close any running installer/app, then retry."
    }
  }
  Invoke-External $iscc "/DAppVersion=$appVersion" "tools\\installer\\slice2solid.iss" | Out-Host
  if (!(Test-Path $outExe)) { throw "Installer not found at expected path: $outExe" }
} else {
  Invoke-External $iscc "tools\\installer\\slice2solid.iss" | Out-Host
}
Write-Host "Installer built (see OutputBaseFilename in tools\\installer\\slice2solid.iss)."
