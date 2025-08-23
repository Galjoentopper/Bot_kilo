@echo off
REM ============================================================================
REM Trading Bot Environment Setup Script
REM ============================================================================
REM This script helps set up the trading bot environment on a new computer
REM Run this first when setting up the bot on your trading computer
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   Trading Bot Environment Setup
echo ========================================
echo.

REM Check if Python is available
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or later from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
) else (
    echo Python found: 
    python --version
)

echo.
echo Checking pip installation...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
) else (
    echo pip found: 
    pip --version
)

echo.
echo ========================================
echo   Installing Python Dependencies
echo ========================================
echo.

echo Installing core dependencies...
pip install numpy pandas pyyaml scikit-learn requests python-telegram-bot
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    pause
    exit /b 1
)

echo.
echo Installing optional ML dependencies...
echo (These may take a while to download and install)
echo.

REM Try to install ML libraries, but don't fail if they don't work
echo Installing LightGBM...
pip install lightgbm
if errorlevel 1 (
    echo WARNING: LightGBM installation failed - you may need Visual Studio Build Tools
    echo You can still use other models, but LightGBM models won't work
)

echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: PyTorch installation failed
    echo You can still use other models, but GRU models won't work
)

echo Installing additional ML libraries...
pip install stable-baselines3 gymnasium mlflow optuna
if errorlevel 1 (
    echo WARNING: Some ML libraries failed to install
    echo Basic functionality should still work
)

echo.
echo ========================================
echo   Creating Directory Structure
echo ========================================
echo.

REM Create necessary directories
if not exist "data" (
    mkdir data
    echo Created: data\
)
if not exist "models" (
    mkdir models
    echo Created: models\
)
if not exist "models\exports" (
    mkdir models\exports
    echo Created: models\exports\
)
if not exist "models\metadata" (
    mkdir models\metadata
    echo Created: models\metadata\
)
if not exist "logs" (
    mkdir logs
    echo Created: logs\
)
if not exist "mlruns" (
    mkdir mlruns
    echo Created: mlruns\
)

echo.
echo ========================================
echo   Configuration Check
echo ========================================
echo.

if exist "src\config\config.yaml" (
    echo Configuration file found: src\config\config.yaml
    echo.
    echo IMPORTANT: Please review your configuration file and update:
    echo - API keys for your exchange
    echo - Telegram bot settings (if using notifications)
    echo - Trading parameters (symbols, intervals, etc.)
    echo.
) else (
    echo WARNING: Configuration file not found!
    echo Please ensure src\config\config.yaml exists and is properly configured
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Your trading bot environment is now set up.
echo.
echo Next steps:
echo.
echo FOR TRAINING COMPUTER:
echo 1. Run train_models.bat to train models
echo 2. Copy the export folder to your trading computer
echo.
echo FOR TRADING COMPUTER:
echo 1. Copy the export folder from your training computer
echo 2. Run the import_models.py script from the export folder
echo 3. Run deploy_trading.bat to start paper trading
echo.
echo CONFIGURATION:
echo - Edit src\config\config.yaml for your exchange API and settings
echo - Test with paper trading before using real money
echo.
echo Press any key to continue...
pause >nul

echo.
echo Would you like to open the configuration file now? (y/n)
set /p "open_config="
if /i "!open_config!" == "y" (
    if exist "src\config\config.yaml" (
        notepad src\config\config.yaml
    ) else (
        echo Configuration file not found
    )
)

echo.
echo Setup complete! You can now use:
echo - train_models.bat (on training computer)
echo - deploy_trading.bat (on trading computer)
echo.
pause