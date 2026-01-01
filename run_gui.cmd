@echo off
setlocal
set PYTHON=%~dp0.venv\Scripts\python.exe
if not exist "%PYTHON%" (
  echo Python venv not found: %PYTHON%
  echo Create it first (see README.md).
  exit /b 1
)
"%PYTHON%" "%~dp0run_gui.py"
