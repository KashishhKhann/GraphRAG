@echo off
REM Graph RAG - Conda Environment Setup Script for Windows
REM Usage: setup_conda.bat

set ENV_NAME=graphrag
set CUDA_VERSION=11.8

echo ============================================================
echo Graph RAG - Conda Environment Setup (Windows)
echo ============================================================

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda not found. Please install Anaconda or Miniconda first.
    exit /b 1
)

REM Check for GPU
nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo GPU detected.
    nvidia-smi --query-gpu=name --format=csv,noheader
) else (
    echo WARNING: No GPU detected. Installing CPU-only version.
    set CUDA_VERSION=cpu
)

echo.
echo Creating conda environment '%ENV_NAME%'...
echo.

REM Remove existing environment
call conda env remove -n %ENV_NAME% -y 2>nul

REM Create new environment
call conda create -n %ENV_NAME% python=3.10 -y

REM Activate environment
call conda activate %ENV_NAME%

echo Installing PyTorch...
if "%CUDA_VERSION%"=="cpu" (
    call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
) else (
    call conda install pytorch torchvision torchaudio pytorch-cuda=%CUDA_VERSION% -c pytorch -c nvidia -y
)

echo Installing core dependencies...
call conda install numpy=1.26.4 scipy scikit-learn -c conda-forge -y

echo Installing spaCy 3.4.4...
call conda install spacy=3.4.4 -c conda-forge -y

echo Installing other conda packages...
call conda install pymongo tqdm pyyaml requests python-dotenv -c conda-forge -y

echo Installing pip packages...
pip install --upgrade pip
pip install sentence-transformers>=2.2.0 transformers>=4.30.0 huggingface-hub>=0.14.0

if "%CUDA_VERSION%"=="cpu" (
    pip install faiss-cpu
) else (
    pip install faiss-gpu
)

pip install neo4j>=5.0.0
pip install scispacy==0.5.4

echo.
echo Installing SciSpaCy model...
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

if %ERRORLEVEL% neq 0 (
    echo WARNING: SciSpaCy model failed. Installing fallback...
    python -m spacy download en_core_web_sm
)

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To activate: conda activate %ENV_NAME%
echo.
echo Verify GPU: python -c "import torch; print(torch.cuda.is_available())"
echo.
echo See RUNBOOK.md for next steps.
echo ============================================================

pause
