@echo off
rem --- sokuhen-llm launcher (detached, phased progress) -------------
rem Closing this window does NOT stop the IME. On every run we
rem RE-VERIFY each dependency and download artefact so an interrupted
rem install (Ctrl+C, power loss, etc.) doesn't leave the environment
rem half-configured; existing complete installs are detected by
rem actually importing / loading them, not by file existence.
rem
rem The window also stays open until the LLM finishes loading (the
rem one really slow step). Press a key only AFTER you see [5/5] OK.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem -- Python locator ---------------------------------------------
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

set "PYW_CMD="
where pyw >nul 2>&1 && set "PYW_CMD=pyw -3"
if "!PYW_CMD!"=="" (
    where pythonw >nul 2>&1 && set "PYW_CMD=pythonw"
)

echo.
echo ==============================================================
echo  sokuhen-llm launcher
echo ==============================================================
echo.

rem -- [1/5] Core IME deps ----------------------------------------
echo [1/5] Core IME dependencies ^(PyQt6, pynput, requests, platformdirs^)
!PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
if errorlevel 1 (
    echo       installing / repairing ^(about 100 MB^)...
    !PY_CMD! -m pip install --upgrade pip >nul
    !PY_CMD! -m pip install PyQt6 pynput requests platformdirs
    if errorlevel 1 (
        echo       FAILED: check internet and rerun.
        goto :pause_and_exit
    )
    !PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
    if errorlevel 1 (
        echo       FAILED: packages install but don't import. Run as Administrator.
        goto :pause_and_exit
    )
)
echo       OK

rem -- [2/5] Dictionary -------------------------------------------
echo [2/5] Japanese dictionary ^(SKK-JISYO.L + edict2 + jinmei^)
set "PYTHONPATH=%~dp0src"
set "DICT_OK=1"
if not exist "data\SKK-JISYO.L" set "DICT_OK="
if defined DICT_OK (
    for %%F in ("data\SKK-JISYO.L") do if %%~zF LSS 1000000 set "DICT_OK="
)
if defined DICT_OK (
    echo       OK
) else (
    echo       downloading ^(about 5 MB^)...
    !PY_CMD! -m sokuhen_llm.scripts.download_dict
    if errorlevel 1 (
        echo       FAILED: dictionary download didn't complete.
        goto :pause_and_exit
    )
    echo       OK
)

rem -- [3/5] LLM deps ---------------------------------------------
if "!SOKUHEN_LLM_DISABLE!"=="1" (
    echo [3/5] LLM dependencies -- SKIPPED ^(SOKUHEN_LLM_DISABLE=1^)
    set "LLM_DISABLED=1"
    goto :skip_llm_all
)
echo [3/5] LLM dependencies ^(transformers, torch, sentencepiece, protobuf, tiktoken^)
!PY_CMD! -c "import transformers, torch, sentencepiece, google.protobuf, tiktoken" >nul 2>&1
if errorlevel 1 (
    echo       installing / repairing ^(about 1.5 GB, one-time^)...
    !PY_CMD! -m pip install "transformers>=4.40" "torch>=2.2" sentencepiece protobuf tiktoken accelerate
    if errorlevel 1 (
        echo       FAILED -- launching WITHOUT LLM. Retry this script later to repair.
        set "LLM_DISABLED=1"
        goto :skip_llm_all
    )
    !PY_CMD! -c "import transformers, torch, sentencepiece, google.protobuf, tiktoken" >nul 2>&1
    if errorlevel 1 (
        echo       FAILED post-install -- launching WITHOUT LLM.
        set "LLM_DISABLED=1"
        goto :skip_llm_all
    )
)
echo       OK

rem -- [4/5] LLM model weights ------------------------------------
echo [4/5] LLM model weights ^(japanese-gpt2-small or fallback^)
set "MODEL_ID=%SOKUHEN_LLM_MODEL%"
if "!MODEL_ID!"=="" set "MODEL_ID=rinna/japanese-gpt2-small"
!PY_CMD! -c "import os, sys; sys.path.insert(0,'src'); from sokuhen_llm.paths import models_dir; from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained(os.environ.get('SOKUHEN_LLM_MODEL','rinna/japanese-gpt2-small'), cache_dir=str(models_dir()), local_files_only=True); AutoModelForCausalLM.from_pretrained(os.environ.get('SOKUHEN_LLM_MODEL','rinna/japanese-gpt2-small'), cache_dir=str(models_dir()), local_files_only=True)" >nul 2>&1
if errorlevel 1 (
    echo       downloading !MODEL_ID! ^(about 440 MB^)...
    !PY_CMD! -m sokuhen_llm.scripts.download_model --model "!MODEL_ID!"
    if errorlevel 1 (
        echo       FAILED -- launching WITHOUT LLM. Retry later.
        set "LLM_DISABLED=1"
        goto :skip_llm_all
    )
)
echo       OK

:skip_llm_all

echo.

rem -- Stop any prior instance ------------------------------------
!PY_CMD! -m sokuhen_llm --stop >nul 2>&1

rem Clear stale status marker so our wait loop doesn't see an
rem "already ready" left over from a previous session.
set "STATUSFILE=%LOCALAPPDATA%\sokuhen-llm\sokuhen-llm\sokuhen-llm.status"
set "PIDFILE=%LOCALAPPDATA%\sokuhen-llm\sokuhen-llm\sokuhen-llm.pid"
set "LOGFILE=%LOCALAPPDATA%\sokuhen-llm\sokuhen-llm\Logs\sokuhen-llm.log"
del /q "!STATUSFILE!" 2>nul

rem -- [5/5] Launch + wait for ready -------------------------------
echo [5/5] Starting sokuhen-llm and waiting for LLM to warm up
if "!PYW_CMD!"=="" (
    echo       no pythonw.exe -- running in foreground ^(closing window stops IME^).
    !PY_CMD! -m sokuhen_llm
    goto :pause_and_exit
)

start "" !PYW_CMD! -m sokuhen_llm

rem Wait for PID file first (proof IME process is alive).
set "LAUNCHED="
for /l %%i in (1,1,60) do (
    if exist "!PIDFILE!" (
        set "LAUNCHED=1"
        goto :pid_ready
    )
    timeout /t 1 /nobreak >nul
)
echo       FAILED: no PID file in 60s. Check the log:
echo         !LOGFILE!
goto :after_poll

:pid_ready
<nul set /p"=      IME process alive -- waiting for LLM"

rem If LLM is disabled, skip the model wait entirely.
if defined LLM_DISABLED (
    echo .
    echo       LLM disabled -- done.
    goto :after_poll
)

rem Now poll the status file. It goes ``starting`` -> ``loading`` ->
rem ``ready`` / ``failed:<msg>`` / ``disabled``. We wait up to 120 s.
set "LLM_STATE="
for /l %%i in (1,1,120) do (
    if exist "!STATUSFILE!" (
        set /p LLM_STATE=<"!STATUSFILE!"
        if "!LLM_STATE!"=="ready"    goto :llm_done
        if "!LLM_STATE!"=="disabled" goto :llm_done
        echo !LLM_STATE! | findstr /b "failed" >nul && goto :llm_done
    )
    <nul set /p"=."
    timeout /t 1 /nobreak >nul
)
echo .
echo       WARNING: LLM did not report ready within 120 s.
echo       The IME is running; LLM may still be loading. Check the log.
goto :after_poll

:llm_done
echo .
if "!LLM_STATE!"=="ready" (
    echo       LLM ready.
) else if "!LLM_STATE!"=="disabled" (
    echo       LLM disabled.
) else (
    echo       LLM status: !LLM_STATE!
)

:after_poll
echo.
echo --------------------------------------------------------------
if defined LAUNCHED (
    set /p PID=<"!PIDFILE!"
    echo  [OK] sokuhen-llm is running ^(PID !PID!^).
    echo.
    echo    * Toggle IME ON/OFF:  Alt + `
    echo    * Stop the IME:       stop-sokuhen-llm.bat
    echo    * Log file:
    echo        !LOGFILE!
    echo.
    echo  You can close this window -- the IME will keep running.
) else (
    echo  [ERROR] sokuhen-llm did not start within 60 seconds.
    echo.
    echo  Log file ^(tail^):
    if exist "!LOGFILE!" (
        powershell -nop -c "Get-Content '!LOGFILE!' -Tail 30" 2>nul
    )
)
echo --------------------------------------------------------------

:pause_and_exit
echo.
echo Press any key to close this window ^(the IME keeps running^)...
pause >nul
endlocal
exit /b 0
