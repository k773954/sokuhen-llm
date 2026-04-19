@echo off
rem --- sokuhen-llm launcher (detached) v2 ----------------------------
rem Requires Windows (CRLF line endings). See .gitattributes.
rem Launches the IME via pythonw.exe and detaches from this window.
rem Closing the terminal does NOT stop the IME. To stop it, run
rem stop-sokuhen-llm.bat.
rem
rem This window stays open until you press a key -- so if anything
rem goes wrong you can read the error. Closing it does not affect
rem the detached IME process.

setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo [trace] cwd: %CD%

rem -- Python locator (console build: used for setup + selftest) ----
set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py -3"
if "!PY_CMD!"=="" (
    where python >nul 2>&1 && set "PY_CMD=python"
)
if "!PY_CMD!"=="" (
    echo [error] Python 3 is not on PATH.
    echo         Install it from https://www.python.org/downloads/windows/
    echo         Be sure to tick "Add Python to PATH" during install.
    goto :pause_and_exit
)
echo [trace] PY_CMD=!PY_CMD!

rem -- Windowless Python (used for the detached run) ---------------
set "PYW_CMD="
where pyw >nul 2>&1 && set "PYW_CMD=pyw -3"
if "!PYW_CMD!"=="" (
    where pythonw >nul 2>&1 && set "PYW_CMD=pythonw"
)
echo [trace] PYW_CMD=!PYW_CMD!

echo.
echo ==================================================
echo  sokuhen-llm launcher
echo ==================================================
echo.

rem -- One-time setup: deps + dict -------------------------------
!PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing dependencies ^(one-time; about 100 MB^)...
    !PY_CMD! -m pip install --upgrade pip
    if errorlevel 1 goto :pause_and_exit
    !PY_CMD! -m pip install PyQt6 pynput requests platformdirs
    if errorlevel 1 goto :pause_and_exit
    echo [setup] Dependencies installed.
    echo.
)

if not exist "data\SKK-JISYO.L" (
    echo [setup] Downloading SKK-JISYO.L ^(about 4 MB, GPLv2+^)...
    set "PYTHONPATH=%~dp0src"
    !PY_CMD! -m sokuhen_llm.scripts.download_dict
    if errorlevel 1 goto :pause_and_exit
    echo [setup] Dictionary ready.
    echo.
)

rem -- One-time setup: LLM deps + model weights --------------------
rem transformers/torch add ~1.5 GB but that's the whole point of this
rem flavor of sokuhen. Skip the install if the user ran sokuhen-llm
rem already (first check) or set SOKUHEN_LLM_DISABLE=1.
if "!SOKUHEN_LLM_DISABLE!"=="1" (
    echo [setup] SOKUHEN_LLM_DISABLE=1 -- skipping LLM install.
) else (
    !PY_CMD! -c "import transformers, torch" >nul 2>&1
    if errorlevel 1 (
        echo [setup] Installing LLM dependencies ^(one-time; about 1.5 GB^)...
        !PY_CMD! -m pip install "transformers>=4.40" "torch>=2.2" sentencepiece accelerate
        if errorlevel 1 (
            echo [warn] LLM deps failed to install. sokuhen-llm will run WITHOUT
            echo        LLM rescoring -- set SOKUHEN_LLM_DISABLE=1 to silence this.
            echo.
        ) else (
            echo [setup] LLM dependencies installed.
            echo.
        )
    )

    rem Pre-fetch the model into models/ so first conversion isn't
    rem blocked by a download. Skipped silently if already cached.
    if not exist "models\models--rinna--japanese-gpt2-small" (
        echo [setup] Downloading japanese-gpt2-small ^(about 440 MB, MIT^)...
        set "PYTHONPATH=%~dp0src"
        !PY_CMD! -m sokuhen_llm.scripts.download_model
        if errorlevel 1 (
            echo [warn] Model download failed. You can retry later with:
            echo        python -m sokuhen_llm.scripts.download_model
            echo.
        )
    )
)

rem -- Stop any prior instance so we don't fight over the pid file -
set "PYTHONPATH=%~dp0src"
echo [launch] Reaping any prior detached instance...
!PY_CMD! -m sokuhen_llm --stop

rem -- Launch the detached IME ------------------------------------
if "!PYW_CMD!"=="" (
    echo [warn] pythonw.exe not found. Running in foreground instead.
    echo        Closing this window WILL stop the IME.
    echo.
    !PY_CMD! -m sokuhen_llm
    goto :pause_and_exit
)

echo [launch] Starting sokuhen-llm in the background via: !PYW_CMD! -m sokuhen_llm
rem "start" with an empty title detaches the child from this cmd so
rem closing the terminal does NOT propagate to pythonw.
start "" !PYW_CMD! -m sokuhen_llm

rem Poll for the pid file as proof-of-life. platformdirs puts it under
rem %LOCALAPPDATA%\sokuhen\sokuhen\sokuhen-llm.pid.
set "PIDFILE=%LOCALAPPDATA%\sokuhen\sokuhen\sokuhen-llm.pid"
set "LOGFILE=%LOCALAPPDATA%\sokuhen\sokuhen\Logs\sokuhen-llm.log"
echo [launch] Waiting up to ~20s for startup...
set "LAUNCHED="
for /l %%i in (1,1,20) do (
    if exist "!PIDFILE!" (
        set "LAUNCHED=1"
        goto :after_poll
    )
    timeout /t 1 /nobreak >nul
)

:after_poll
echo.
echo --------------------------------------------------
if defined LAUNCHED (
    set /p PID=<"!PIDFILE!"
    echo  [OK] sokuhen is running in the background ^(PID !PID!^).
    echo.
    echo   * Toggle IME ON/OFF:  Alt + `
    echo   * Stop the IME:       stop-sokuhen-llm.bat
    echo   * Log file:
    echo       !LOGFILE!
    echo.
    echo  You can close this window -- the IME will keep running.
) else (
    echo  [ERROR] sokuhen did not start within 20 seconds.
    echo.
    echo  Possible causes:
    echo    1. Missing dependency -- try: !PY_CMD! -m pip install PyQt6 pynput requests platformdirs
    echo    2. Hook install failed -- try running sokuhen-llm-debug.bat to see the error.
    echo.
    echo  Log file ^(tail^):
    if exist "!LOGFILE!" (
        powershell -nop -c "Get-Content '!LOGFILE!' -Tail 20" 2>nul
    ) else (
        echo    ^(no log file at !LOGFILE!^)
    )
)
echo --------------------------------------------------

:pause_and_exit
echo.
echo Press any key to close this window ^(the IME keeps running in the background^)...
pause >nul
endlocal
exit /b 0
