@echo off
rem --- stop the running sokuhen instance ---------------------------
rem Double-click to gracefully terminate sokuhen. Uses the pid file
rem written by the app on startup to find the right process, then
rem falls back to taskkill /F if the process ignores the shutdown
rem signal. Safe to run even when nothing is running.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py -3"
if "!PY_CMD!"=="" (
    where python >nul 2>&1 && set "PY_CMD=python"
)
if "!PY_CMD!"=="" (
    echo [error] Python 3 is not on PATH; cannot locate the pid file location.
    pause
    exit /b 1
)

set "PYTHONPATH=%~dp0src"
!PY_CMD! -m sokuhen_llm --stop
set "EXITCODE=!ERRORLEVEL!"

echo.
pause
endlocal
exit /b !EXITCODE!
