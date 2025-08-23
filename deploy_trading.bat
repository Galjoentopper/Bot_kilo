@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Enhanced Paper Trading Deployment
echo ========================================
echo.
echo This script will set up and run the paper trading bot
echo on this Windows computer using models imported from
echo your Linux training computer.
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python is installed.
echo.

REM Check if we're in the correct directory
if not exist "scripts" (
    echo ERROR: Please run this script from the Bot_kilo root directory
    echo Current directory: %CD%
    echo Expected to find 'scripts' folder here.
    pause
    exit /b 1
)

echo Directory check passed.
echo.

REM Check if models directory exists and has content
if not exist "models" (
    echo ERROR: Models directory not found!
    echo.
    echo Please import models first:
    echo 1. Copy your transfer package (*.zip) from Linux training computer to this directory
    echo 2. Run 'import_models.bat'
    echo.
    echo Or use the cross-platform transfer:
    echo   python scripts/cross_platform_transfer.py import --package your_package.zip --destination models
    echo.
    pause
    exit /b 1
)

dir /b models 2>nul | findstr . >nul
if errorlevel 1 (
    echo ERROR: Models directory is empty!
    echo.
    echo Please import models first:
    echo 1. Copy your transfer package (*.zip) from Linux training computer to this directory
    echo 2. Run 'import_models.bat'
    echo.
    pause
    exit /b 1
)

echo Models directory found with content.
echo.

REM Check if requirements.txt exists and install from it
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo WARNING: Some dependencies from requirements.txt failed to install
        echo Continuing with manual dependency check...
    ) else (
        echo Dependencies installed from requirements.txt
        goto :skip_manual_deps
    )
)

REM Manual dependency check if requirements.txt failed or doesn't exist
echo Checking Python dependencies manually...

REM Essential packages for trading
set "PACKAGES=numpy pandas PyYAML scikit-learn requests python-telegram-bot ccxt"

for %%p in (%PACKAGES%) do (
    echo Checking %%p...
    python -c "import %%p" 2>nul
    if errorlevel 1 (
        echo Installing %%p...
        pip install %%p
        if errorlevel 1 (
            echo WARNING: Failed to install %%p
            echo You may need to install it manually: pip install %%p
        )
    )
)

:skip_manual_deps
echo Dependencies check completed.
echo.

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "config" mkdir config

echo Created necessary directories.
echo.

REM Check for trading configuration
if not exist "config\trading_config.yaml" (
    if exist "trading_config.yaml" (
        echo Moving trading_config.yaml to config directory...
        move trading_config.yaml config\
    ) else (
        echo WARNING: trading_config.yaml not found!
        echo The bot will use default settings.
        echo You may want to create a configuration file for optimal performance.
    )
)

REM Check if enhanced trader script exists, fallback to regular trader
set "TRADER_SCRIPT="
if exist "scripts\enhanced_trader.py" (
    set "TRADER_SCRIPT=scripts\enhanced_trader.py"
    echo Using enhanced trader script.
else if exist "scripts\trader.py" (
    set "TRADER_SCRIPT=scripts\trader.py"
    echo Using standard trader script.
else (
    echo ERROR: No trader script found!
    echo Expected: scripts\enhanced_trader.py or scripts\trader.py
    pause
    exit /b 1
)

echo.

REM Run model validation
echo Running model validation...
if exist "validate_models.bat" (
    call validate_models.bat
    if errorlevel 1 (
        echo Model validation failed or was cancelled.
        pause
        exit /b 1
    )
else (
    echo validate_models.bat not found, skipping validation...
)

echo.
echo ========================================
echo     Starting Paper Trading Bot
echo ========================================
echo.
echo Configuration:
echo - Trader Script: !TRADER_SCRIPT!
echo - Models Directory: models
echo - Logs Directory: logs
echo - Mode: Paper Trading
echo.
echo The bot will start in paper trading mode.
echo Press Ctrl+C to stop the bot.
echo.
echo Logs will be saved to the 'logs' directory.
echo Monitor the logs for trading activity and performance.
echo.
set /p "START=Press Enter to start trading or Ctrl+C to cancel..."

REM Start the trading bot with appropriate arguments
echo Starting trader...
if "!TRADER_SCRIPT!"=="scripts\enhanced_trader.py" (
    python !TRADER_SCRIPT! --mode paper --config config\trading_config.yaml
else (
    python !TRADER_SCRIPT! --paper-trading
)

if errorlevel 1 (
    echo.
    echo ERROR: Trading bot encountered an error!
    echo Check the logs directory for more information.
    echo.
    echo Common issues:
    echo - Missing or corrupted model files
    echo - Network connectivity problems
    echo - Configuration errors
    echo.
    echo Try running 'validate_models.bat' to check your models.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Trading bot stopped successfully.
echo ========================================
echo.
echo Check the following for results:
echo - logs\ directory for trading logs
echo - data\ directory for market data
echo - Any generated reports or performance files
echo.
pause