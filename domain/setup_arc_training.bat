@echo off
REM ============================================================
REM  NML Training Environment Setup — Intel Arc Pro B50
REM  Run this script in a regular CMD window (not PowerShell).
REM
REM  Steps performed:
REM    1. Check for Intel oneAPI Base Toolkit
REM    2. Activate oneAPI environment
REM    3. Create a Python venv
REM    4. Install Intel XPU wheels + training stack
REM    5. Print verification command
REM ============================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo  NML Training Setup for Intel Arc Pro B50
echo ============================================================
echo.

REM ----------------------------------------------------------
REM Step 1 — Check oneAPI
REM ----------------------------------------------------------
set ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
set SETVARS=%ONEAPI_ROOT%\setvars.bat

if not exist "%SETVARS%" (
    echo [!] Intel oneAPI Base Toolkit NOT found at:
    echo     %SETVARS%
    echo.
    echo     Please install it first:
    echo     https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
    echo.
    echo     Select "Intel oneAPI Base Toolkit 2025.2.1" (or later).
    echo     A ~3 GB download. Accept defaults during installation.
    echo.
    echo     Also enable Resizable BAR in BIOS before training:
    echo       BIOS > Advanced > PCI Configuration > Resizable BAR = Enabled
    echo.
    pause
    exit /b 1
)

echo [1/5] Found Intel oneAPI at: %ONEAPI_ROOT%

REM ----------------------------------------------------------
REM Step 2 — Activate oneAPI environment (for this session)
REM ----------------------------------------------------------
echo [2/5] Activating oneAPI environment...
call "%SETVARS%" --force >nul 2>&1
if errorlevel 1 (
    echo [!] Failed to activate oneAPI. Try running this script
    echo     from a fresh CMD window (not inside VS Code terminal).
    pause
    exit /b 1
)
echo       Done.

REM ----------------------------------------------------------
REM Step 3 — Create Python venv at nml\venv_arc
REM ----------------------------------------------------------
set VENV_DIR=%~dp0venv_arc

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [3/5] venv_arc already exists — skipping creation.
) else (
    echo [3/5] Creating Python venv at %VENV_DIR% ...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [!] Failed to create venv. Is Python 3.10+ installed?
        pause
        exit /b 1
    )
    echo       Done.
)

call "%VENV_DIR%\Scripts\activate.bat"
echo       Venv activated: %VIRTUAL_ENV%

REM ----------------------------------------------------------
REM Step 4 — Install wheels
REM ----------------------------------------------------------
echo [4/5] Installing training stack (this may take 5-10 minutes)...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip --quiet

REM PyTorch with Intel XPU support (nightly 2.7+)
echo   Installing PyTorch with XPU backend...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --quiet
if errorlevel 1 (
    echo [!] PyTorch XPU install failed. Trying intel extension wheels...
)

REM IPEX-LLM — Intel's optimized LLM library (replaces standard transformers loading)
echo   Installing IPEX-LLM...
pip install ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --quiet
if errorlevel 1 (
    echo [WARN] ipex-llm[xpu] failed. Will fall back to standard transformers.
    pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --quiet
)

REM Core training stack
echo   Installing transformers, peft, trl, datasets...
pip install transformers>=4.40.0 peft>=0.10.0 trl>=0.8.0 datasets>=2.18.0 accelerate>=0.28.0 --quiet

REM bitsandbytes with Intel support
echo   Installing bitsandbytes (Intel variant)...
pip install bitsandbytes --quiet
REM Try Intel bitsandbytes if standard one doesn't support XPU
pip install bitsandbytes-intel --quiet 2>nul

REM Utilities
pip install sentencepiece protobuf scipy numpy --quiet

echo.
echo [5/5] Setup complete!
echo.
echo ============================================================
echo  NEXT STEPS
echo ============================================================
echo.
echo  1. Verify the environment (run in this window):
echo.
echo       python verify_arc_env.py
echo.
echo  2. Quick test run (1.5B model, 200 steps, no eval):
echo.
echo       python train_nml.py ^
echo           --model Qwen/Qwen2.5-Coder-1.5B-Instruct ^
echo           --steps 200 ^
echo           --batch 2 ^
echo           --grad-acc 2 ^
echo           --save-steps 100 ^
echo           --no-eval ^
echo           --output domain/output/model/test-run
echo.
echo     To RESUME a crashed test run from the last checkpoint:
echo.
echo       python train_nml.py ^
echo           --model Qwen/Qwen2.5-Coder-1.5B-Instruct ^
echo           --steps 200 ^
echo           --batch 2 ^
echo           --grad-acc 2 ^
echo           --save-steps 100 ^
echo           --no-eval ^
echo           --resume last ^
echo           --output domain/output/model/test-run
echo.
echo  3. Full Stage-1 training (7B model, 30K steps):
echo.
echo       python train_nml.py
echo.
echo     To resume a crashed Stage-1 run:
echo.
echo       python train_nml.py --resume last
echo.
echo  IMPORTANT: Run all training from a CMD window where
echo  you have called:
echo       "%SETVARS%"
echo  (or run setup_arc_training.bat again — it does this).
echo.
echo  OPTIONAL ENV VARS (set before running train_nml.py):
echo.
echo    Skip HF Hub network check (model already downloaded):
echo       set HF_HUB_OFFLINE=1
echo    Or pass --offline flag directly:
echo       python train_nml.py --offline ...
echo.
echo    Authenticate with HuggingFace (higher rate limits):
echo       set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
echo       (get a token at https://huggingface.co/settings/tokens)
echo.
echo ============================================================

pause
