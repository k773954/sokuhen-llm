@echo off
rem --- sokuhen foreground / debug launcher --------------------------
rem Runs the IME tied to this CMD window. Closing the window stops the
rem IME. Use this when you need to read log output directly or attach
rem a debugger. For normal use, run sokuhen-llm.bat (which detaches).

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py -3"
if "!PY_CMD!"=="" (
    where python >nul 2>&1 && set "PY_CMD=python"
)
if "!PY_CMD!"=="" (
    echo [error] Python 3 is not on PATH.
    pause
    exit /b 1
)

!PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing dependencies...
    !PY_CMD! -m pip install --upgrade pip >nul
    !PY_CMD! -m pip install PyQt6 pynput requests platformdirs || goto :fail
)

if not exist "data\SKK-JISYO.L" (
    set "PYTHONPATH=%~dp0src"
    !PY_CMD! -m sokuhen_llm.scripts.download_dict || goto :fail
)

rem Kill any detached prior instance to avoid fighting over the pid file.
set "PYTHONPATH=%~dp0src"
!PY_CMD! -m sokuhen_llm --stop >nul 2>&1

echo --------------------------------------------------
echo  sokuhen (foreground / debug mode)
echo  Closing this window STOPS the IME.
echo --------------------------------------------------
echo.

!PY_CMD! -m sokuhen_llm
set "EXITCODE=!ERRORLEVEL!"
echo.
echo --------------------------------------------------
echo  sokuhen-llm stopped (exit code !EXITCODE!).
echo --------------------------------------------------
pause
endlocal
exit /b !EXITCODE!

:fail
echo [error] Setup failed.
pause
endlocal
exit /b 1
