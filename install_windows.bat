@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem AI Engineering Assessment Studio Windows bootstrapper.
rem Run this file from Command Prompt, PowerShell, or an Anaconda Prompt.

pushd "%~dp0"
if errorlevel 1 goto :fail
set "REPO_ROOT=%CD%"

echo.
echo ================================================
echo AI Engineering Assessment Studio setup
echo ================================================
echo.

set "ENV_NAME="
set /p "ENV_NAME=Enter a Conda environment name [ai-assessment-studio]: "
if not defined ENV_NAME set "ENV_NAME=ai-assessment-studio"

rem Keep the name portable and safe for activation commands.
echo(%ENV_NAME%| %SystemRoot%\System32\findstr.exe /r /x "[A-Za-z0-9][A-Za-z0-9._-]*" >nul
if errorlevel 1 (
    echo.
    echo Invalid environment name.
    echo Use only letters, numbers, dots, underscores, and hyphens.
    goto :fail
)

set "CONDA_CMD="
for /f "delims=" %%G in ('where conda.bat 2^>nul') do if not defined CONDA_CMD set "CONDA_CMD=%%G"
if not defined CONDA_CMD if defined CONDA_EXE if exist "%CONDA_EXE%" set "CONDA_CMD=%CONDA_EXE%"

if not defined CONDA_CMD (
    for %%G in (
        "%USERPROFILE%\anaconda3\condabin\conda.bat"
        "%USERPROFILE%\miniconda3\condabin\conda.bat"
        "%USERPROFILE%\miniforge3\condabin\conda.bat"
        "%USERPROFILE%\mambaforge\condabin\conda.bat"
        "%ProgramData%\anaconda3\condabin\conda.bat"
        "%ProgramData%\miniconda3\condabin\conda.bat"
        "%ProgramData%\miniforge3\condabin\conda.bat"
        "%ProgramData%\mambaforge\condabin\conda.bat"
    ) do if not defined CONDA_CMD if exist "%%~G" set "CONDA_CMD=%%~G"
)

if not defined CONDA_CMD (
    echo.
    echo Conda was not found from this script.
    echo.
    echo Open an Anaconda Prompt, or initialize Conda for the shell you use:
    echo   conda init cmd.exe
    echo   conda init powershell
    echo Then close and reopen the prompt and run this file again.
    goto :fail
)

echo Using Conda command: %CONDA_CMD%
echo Target environment: %ENV_NAME%
echo.

set "ENV_LIST=%TEMP%\ai_assessment_conda_envs_%RANDOM%.txt"
call "%CONDA_CMD%" env list > "%ENV_LIST%" 2>nul
if errorlevel 1 (
    echo Conda could not list environments.
    del /q "%ENV_LIST%" >nul 2>nul
    goto :fail
)

findstr.exe /l /b /c:"%ENV_NAME% " "%ENV_LIST%" >nul
if errorlevel 1 (
    echo Creating Conda environment "%ENV_NAME%" with Python 3.12...
    call "%CONDA_CMD%" create -n "%ENV_NAME%" python=3.12 pip -y
    if errorlevel 1 (
        del /q "%ENV_LIST%" >nul 2>nul
        echo Conda environment creation failed.
        goto :fail
    )
) else (
    echo Reusing existing environment "%ENV_NAME%".
    call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import sys; print('Python ' + sys.version.split()[0]); raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if errorlevel 1 (
        del /q "%ENV_LIST%" >nul 2>nul
        echo The existing environment must use Python 3.12.
        echo Choose another environment name or recreate this environment manually.
        goto :fail
    )
)
del /q "%ENV_LIST%" >nul 2>nul

echo.
echo Updating pip...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 goto :install_fail

rem Ultralytics uses PyTorch. Install an official CUDA wheel before the GUI extra
rem so a fresh or previously CPU-only environment can train on an NVIDIA GPU.
set "CUDA_EXPECTED="
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :cpu_setup
nvidia-smi -L >nul 2>nul
if errorlevel 1 goto :cpu_setup
set "CUDA_EXPECTED=1"
echo.
echo NVIDIA GPU detected. Checking CUDA PyTorch in "%ENV_NAME%"...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import torch, torchvision; assert torch.cuda.is_available(); torch.zeros(1, device='cuda:0'); print('CUDA PyTorch already works:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if not errorlevel 1 goto :install_gui
echo Installing PyTorch 2.9.1 with CUDA 12.8 from the official PyTorch wheel index...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -m pip install --force-reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :cuda_fail
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import torch, torchvision; assert torch.version.cuda == '12.8' and torch.cuda.is_available(); torch.zeros(1, device='cuda:0'); print('CUDA ready:', torch.__version__, torch.cuda.get_device_name(0))"
if errorlevel 1 goto :cuda_fail
goto :install_gui

:cpu_setup
echo.
echo No working NVIDIA driver/GPU detected by nvidia-smi. Installing for CPU use.
echo To enable GPU training, install or update the NVIDIA driver and rerun this script.

:install_gui
echo.
echo Installing the GUI and training-server dependencies with the editable project package...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -m pip install -e ".[gui,training-server]"
if errorlevel 1 goto :install_fail

echo.
echo Verifying the installation...
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import PySide6, ai_assessment, ultralytics, torch; print('Application and YOLO imports OK; PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
if errorlevel 1 goto :install_fail
if not defined CUDA_EXPECTED goto :setup_complete
call "%CONDA_CMD%" run --no-capture-output -n "%ENV_NAME%" python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda:0'); print('YOLO training can use:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :cuda_fail

:setup_complete

echo.
echo ================================================
echo Setup complete
echo ================================================
echo.
echo This script cannot activate the parent CMD or PowerShell session.
echo In a new Command Prompt or Anaconda Prompt, run:
echo   call conda activate %ENV_NAME%
echo   python -m ai_assessment.app --headless-check --workspace candidate_workspaces\C014
echo   python -m ai_assessment.app --workspace candidate_workspaces\C014
echo.
echo In PowerShell, run:
echo   conda activate %ENV_NAME%
echo   python -m ai_assessment.app --headless-check --workspace candidate_workspaces\C014
echo   python -m ai_assessment.app --workspace candidate_workspaces\C014
echo.
echo To host the LAN training server from this environment, run from the repository root:
echo   python -m server.run_training_server
echo.
echo No model weights or datasets were downloaded by this setup script.
popd
exit /b 0

:cuda_fail
echo.
echo CUDA setup failed: the NVIDIA driver, GPU, or PyTorch wheel may be incompatible.
echo Check nvidia-smi and the PyTorch Windows install selector for a matching CUDA build.
echo This installer will not silently use CPU on a detected NVIDIA GPU.
goto :fail

:install_fail
echo.
echo Dependency installation failed. Review the Conda or pip output above.
goto :fail

:fail
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" set "EXIT_CODE=1"
popd >nul 2>nul
exit /b %EXIT_CODE%
