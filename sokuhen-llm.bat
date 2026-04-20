@echo off
rem --- sokuhen-llm launcher (detached) ------------------------------
rem Requires Windows (CRLF line endings). See .gitattributes.
rem Launches the IME via pythonw.exe and detaches from this window.
rem Closing the terminal does NOT stop the IME.
rem
rem On every run this script RE-VERIFIES each dependency and download
rem artefact so that an interrupted install (Ctrl+C, power loss,
rem network drop) doesn't leave the environment half-configured.
rem Existing, complete installs are detected by actually importing
rem the packages / calling the integrity check, not just by file
rem existence -- a partial download that creates a directory still
rem fails the import and triggers a reinstall.

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

rem -- 1. Core IME dependencies -----------------------------------
rem We import each package separately so a failed install of one
rem doesn't hide a missing other. ``pip install`` is idempotent and
rem cheap when packages are already present, so we run it
rem unconditionally whenever an import fails -- that way interrupted
rem installs get retried on the next launch.
echo [check] Core IME dependencies ^(PyQt6, pynput, requests, platformdirs^)...
!PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing/repairing core dependencies ^(~100 MB^)...
    !PY_CMD! -m pip install --upgrade pip
    !PY_CMD! -m pip install PyQt6 pynput requests platformdirs
    if errorlevel 1 (
        echo [error] Core dependency install failed. Check internet + try again.
        goto :pause_and_exit
    )
    rem Re-verify after install so a corrupted wheel doesn't slip through.
    !PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
    if errorlevel 1 (
        echo [error] Core dependencies still don't import after install.
        echo         Try running this script from an Administrator cmd.
        goto :pause_and_exit
    )
    echo [setup] Core dependencies OK.
) else (
    echo [check] Core dependencies OK.
)

rem -- 2. Dictionary ----------------------------------------------
rem The download_dict script is itself idempotent -- it skips files
rem that already exist. We also verify file size is non-trivial so a
rem truncated download gets re-fetched.
set "PYTHONPATH=%~dp0src"
set "DICT_OK=1"
if not exist "data\SKK-JISYO.L" set "DICT_OK="
if defined DICT_OK (
    for %%F in ("data\SKK-JISYO.L") do if %%~zF LSS 1000000 set "DICT_OK="
)
if defined DICT_OK (
    echo [check] Dictionary OK ^(data\SKK-JISYO.L^).
) else (
    echo [setup] Downloading dictionaries ^(SKK-JISYO.L + edict2 + jinmei^)...
    !PY_CMD! -m sokuhen_llm.scripts.download_dict
    if errorlevel 1 (
        echo [error] Dictionary download failed. Check internet + try again.
        goto :pause_and_exit
    )
    echo [setup] Dictionary ready.
)

rem -- 3. LLM layer (skippable via SOKUHEN_LLM_DISABLE=1) ----------
if "!SOKUHEN_LLM_DISABLE!"=="1" (
    echo [check] SOKUHEN_LLM_DISABLE=1 -- LLM layer skipped.
    goto :skip_llm
)

echo [check] LLM dependencies ^(transformers, torch^)...
!PY_CMD! -c "import transformers, torch" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing/repairing LLM dependencies ^(~1.5 GB, one-time^)...
    !PY_CMD! -m pip install "transformers>=4.40" "torch>=2.2" sentencepiece accelerate
    if errorlevel 1 (
        echo [warn] LLM dependency install failed. Launching WITHOUT LLM
        echo        rescoring. Set SOKUHEN_LLM_DISABLE=1 to silence this.
        echo.
        goto :skip_llm
    )
    !PY_CMD! -c "import transformers, torch" >nul 2>&1
    if errorlevel 1 (
        echo [warn] LLM deps still don't import after install. Launching
        echo        WITHOUT LLM rescoring.
        echo.
        goto :skip_llm
    )
    echo [setup] LLM dependencies OK.
) else (
    echo [check] LLM dependencies OK.
)

rem Verify the model actually loads -- not just that the cache dir
rem exists. A Ctrl+C during ``download_model`` leaves a half-filled
rem directory that passes a simple ``exist`` check but fails to
rem instantiate. ``from_pretrained`` does the right integrity check
rem (config present, safetensors present, tokenizer files present).
echo [check] LLM model ^(rinna/japanese-gpt2-small^)...
set "MODEL_ID=%SOKUHEN_LLM_MODEL%"
if "!MODEL_ID!"=="" set "MODEL_ID=rinna/japanese-gpt2-small"
!PY_CMD! -c "import os, sys; sys.path.insert(0,'src'); from sokuhen_llm.paths import models_dir; from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained(os.environ.get('SOKUHEN_LLM_MODEL','rinna/japanese-gpt2-small'), cache_dir=str(models_dir()), local_files_only=True); AutoModelForCausalLM.from_pretrained(os.environ.get('SOKUHEN_LLM_MODEL','rinna/japanese-gpt2-small'), cache_dir=str(models_dir()), local_files_only=True)" >nul 2>&1
if errorlevel 1 (
    echo [setup] Downloading ^/ repairing model !MODEL_ID! ^(~440 MB^)...
    !PY_CMD! -m sokuhen_llm.scripts.download_model --model "!MODEL_ID!"
    if errorlevel 1 (
        echo [warn] Model download failed. Launching WITHOUT LLM rescoring.
        echo        Retry later:  python -m sokuhen_llm.scripts.download_model
        echo.
        goto :skip_llm
    )
    echo [setup] Model ready.
) else (
    echo [check] LLM model OK.
)

:skip_llm

echo.

rem -- 4. Stop any prior instance ---------------------------------
rem ``--stop`` is intentionally lightweight: it only reads the PID
rem file and sends a signal, without loading the engine or LLM.
echo [launch] Reaping any prior detached instance...
!PY_CMD! -m sokuhen_llm --stop

rem -- 5. Launch the detached IME ---------------------------------
if "!PYW_CMD!"=="" (
    echo [warn] pythonw.exe not found. Running in foreground instead.
    echo        Closing this window WILL stop the IME.
    echo.
    !PY_CMD! -m sokuhen_llm
    goto :pause_and_exit
)

echo [launch] Starting sokuhen-llm in the background via: !PYW_CMD! -m sokuhen_llm
start "" !PYW_CMD! -m sokuhen_llm

rem Poll for the pid file as proof-of-life. platformdirs lays out
rem the user data dir as %LOCALAPPDATA%\<APP_AUTHOR>\<APP_NAME>\.
rem For this app both are "sokuhen-llm".
set "PIDFILE=%LOCALAPPDATA%\sokuhen-llm\sokuhen-llm\sokuhen-llm.pid"
set "LOGFILE=%LOCALAPPDATA%\sokuhen-llm\sokuhen-llm\Logs\sokuhen-llm.log"
rem LLM model loading on a cold boot takes 5-15s on CPU. Give it up
rem to 60s before giving up.
echo [launch] Waiting up to 60s for startup ^(LLM model load is ~5-15s on CPU^)...
set "LAUNCHED="
for /l %%i in (1,1,60) do (
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
    echo  [OK] sokuhen-llm is running in the background ^(PID !PID!^).
    echo.
    echo   * Toggle IME ON/OFF:  Alt + `
    echo   * Stop the IME:       stop-sokuhen-llm.bat
    echo   * Log file:
    echo       !LOGFILE!
    echo.
    echo  You can close this window -- the IME will keep running.
) else (
    echo  [ERROR] sokuhen-llm did not start within 60 seconds.
    echo.
    echo  Possible causes:
    echo    1. The LLM is still loading ^(very slow disk / first run^). Try
    echo       running sokuhen-llm-debug.bat to see progress live.
    echo    2. Missing dependency. Re-run this script; it repairs on retry.
    echo    3. A pending Alt+` / AutoHotkey conflict blocked the hook install.
    echo.
    echo  Log file ^(tail^):
    if exist "!LOGFILE!" (
        powershell -nop -c "Get-Content '!LOGFILE!' -Tail 30" 2>nul
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
