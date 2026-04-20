@echo off
rem --- sokuhen-llm foreground / debug launcher ----------------------
rem Runs the IME tied to this CMD window. Closing the window stops the
rem IME. Use this when you need to read log output directly or attach
rem a debugger. For normal use, run sokuhen-llm.bat (which detaches).
rem
rem Like the detached launcher, this script RE-VERIFIES dependencies
rem on every run so a half-finished install doesn't silently slip by.

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

rem -- 1. Core deps (re-verify each run) --------------------------
echo [check] Core IME dependencies...
!PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing core dependencies...
    !PY_CMD! -m pip install --upgrade pip >nul
    !PY_CMD! -m pip install PyQt6 pynput requests platformdirs || goto :fail
    !PY_CMD! -c "import PyQt6, pynput, platformdirs, requests" || goto :fail
)

rem -- 2. Dictionary ---------------------------------------------
set "PYTHONPATH=%~dp0src"
set "DICT_OK=1"
if not exist "data\SKK-JISYO.L" set "DICT_OK="
if defined DICT_OK (
    for %%F in ("data\SKK-JISYO.L") do if %%~zF LSS 1000000 set "DICT_OK="
)
if not defined DICT_OK (
    echo [setup] Downloading dictionaries...
    !PY_CMD! -m sokuhen_llm.scripts.download_dict || goto :fail
)

rem -- 3. LLM deps + model (skippable) ---------------------------
if "!SOKUHEN_LLM_DISABLE!"=="1" goto :skip_llm
echo [check] LLM dependencies...
!PY_CMD! -c "import transformers, torch" >nul 2>&1
if errorlevel 1 (
    echo [setup] Installing LLM dependencies ^(~1.5 GB, one-time^)...
    !PY_CMD! -m pip install "transformers>=4.40" "torch>=2.2" sentencepiece accelerate
    if errorlevel 1 (
        echo [warn] LLM deps failed to install. Continuing without LLM.
        goto :skip_llm
    )
)
set "MODEL_ID=%SOKUHEN_LLM_MODEL%"
if "!MODEL_ID!"=="" set "MODEL_ID=rinna/japanese-gpt2-small"
!PY_CMD! -c "import os, sys; sys.path.insert(0,'src'); from sokuhen_llm.paths import models_dir; from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained(os.environ.get('SOKUHEN_LLM_MODEL','rinna/japanese-gpt2-small'), cache_dir=str(models_dir()), local_files_only=True); AutoModelForCausalLM.from_pretrained(os.environ.get('SOKUHEN_LLM_MODEL','rinna/japanese-gpt2-small'), cache_dir=str(models_dir()), local_files_only=True)" >nul 2>&1
if errorlevel 1 (
    echo [setup] Downloading ^/ repairing model !MODEL_ID! ^(~440 MB^)...
    !PY_CMD! -m sokuhen_llm.scripts.download_model --model "!MODEL_ID!"
    if errorlevel 1 (
        echo [warn] Model download failed. Continuing without LLM.
        goto :skip_llm
    )
)
:skip_llm

rem Kill any detached prior instance to avoid fighting over the pid file.
set "PYTHONPATH=%~dp0src"
!PY_CMD! -m sokuhen_llm --stop >nul 2>&1

echo --------------------------------------------------
echo  sokuhen-llm (foreground / debug mode)
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
