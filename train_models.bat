@echo off
REM ============================================================================
REM Enhanced Model Training Script for Trading Bot
REM ============================================================================
REM This script trains models and packages them for easy transfer to another computer
REM Run this on your powerful training computer
REM
REM Usage:
REM   train_models.bat                    - Train all models with default settings
REM   train_models.bat --symbols BTCEUR   - Train only BTCEUR models
REM   train_models.bat --models lightgbm  - Train only LightGBM models
REM
REM Checkpoint Options (for 6-hour runtime limits):
REM   train_models.bat --resume           - Resume from last checkpoint
REM   train_models.bat --checkpoint-dir checkpoints\custom - Use custom checkpoint directory
REM   train_models.bat --auto-checkpoint  - Enable automatic checkpointing (default: enabled)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Trading Bot Model Training
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "src\config\config.yaml" (
    echo ERROR: config.yaml not found. Are you in the Bot_kilo directory?
    echo Current directory: %CD%
    echo Please navigate to the Bot_kilo directory and run this script again
    pause
    exit /b 1
)

REM Check if required directories exist
if not exist "data" (
    echo ERROR: data directory not found
    echo Please ensure you have market data in the 'data' directory
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "models" mkdir models
if not exist "models\exports" mkdir models\exports
if not exist "mlruns" mkdir mlruns
if not exist "checkpoints" mkdir checkpoints

echo Checking Python dependencies...
python -c "import numpy, pandas, yaml, lightgbm, torch" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some Python dependencies may be missing
    echo Installing required packages...
    pip install numpy pandas pyyaml lightgbm torch scikit-learn
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting enhanced model training...
echo Training will include automatic model packaging for transfer
echo Checkpoint system enabled - training can be resumed if interrupted
echo.

REM Check if this is a resume operation
echo %* | findstr /C:"--resume" >nul
if not errorlevel 1 (
    echo Resuming training from checkpoint...
)

REM Run the enhanced trainer with packaging and checkpoint support enabled
python scripts\enhanced_trainer.py --package-models --create-transfer-bundle --auto-checkpoint %*

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo Check the logs above for details
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Training Completed Successfully!
echo ========================================
echo.
echo Your trained models have been packaged and are ready for transfer.
echo Look for the export directory in models\exports\
echo.
echo Next steps:
echo 1. Copy the entire export folder to your trading computer
echo 2. Run the import_models.py script on your trading computer
echo 3. Start paper trading with deploy_trading.bat
echo.
echo Note: If training was interrupted, you can resume with: train_models.bat --resume
echo.
echo Press any key to open the exports folder...
pause >nul
explorer models\exports

echo.
echo Training session complete!
pause