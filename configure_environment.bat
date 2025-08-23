@echo off
REM Environment Configuration Script
REM This script helps configure the trading bot for different environments

setlocal enabledelayedexpansion

echo ========================================
echo   Trading Bot Environment Manager
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "src\config\environment_manager.py" (
    echo ERROR: Please run this script from the Bot_kilo directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Show current environment status
echo Checking current environment...
echo.
python src\config\environment_manager.py --status
echo.

:MENU
echo ========================================
echo   Configuration Options
echo ========================================
echo 1. Setup for Training Computer (High Performance)
echo 2. Setup for Trading Computer (Lightweight)
echo 3. Show Environment Status
echo 4. Validate Current Configuration
echo 5. Generate Environment Report
echo 6. Exit
echo.
set /p choice="Please select an option (1-6): "

if "%choice%"=="1" goto SETUP_TRAINING
if "%choice%"=="2" goto SETUP_TRADING
if "%choice%"=="3" goto SHOW_STATUS
if "%choice%"=="4" goto VALIDATE_CONFIG
if "%choice%"=="5" goto GENERATE_REPORT
if "%choice%"=="6" goto EXIT

echo Invalid choice. Please select 1-6.
echo.
goto MENU

:SETUP_TRAINING
echo.
echo ========================================
echo   Setting up Training Environment
echo ========================================
echo.
echo This will configure the system for model training.
echo Training configuration includes:
echo - High resource usage settings
echo - Advanced feature engineering
echo - Model packaging and export
echo - MLflow tracking
echo - Optuna optimization
echo.
set /p confirm="Continue with training setup? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

echo.
echo Configuring training environment...
python src\config\environment_manager.py --setup training
if errorlevel 1 (
    echo.
    echo ERROR: Failed to configure training environment
    pause
    goto MENU
)

echo.
echo ========================================
echo   Training Environment Setup Complete
echo ========================================
echo.
echo Next steps:
echo 1. Review and customize config.yaml if needed
echo 2. Run setup_environment.bat to install dependencies
echo 3. Use train_models.bat to start training
echo.
echo Configuration file: config.yaml
echo Training script: train_models.bat
echo.
pause
goto MENU

:SETUP_TRADING
echo.
echo ========================================
echo   Setting up Trading Environment
echo ========================================
echo.
echo This will configure the system for paper trading.
echo Trading configuration includes:
echo - Lightweight resource usage
echo - Optimized for real-time trading
echo - Model loading from imports
echo - Minimal feature engineering
echo.
set /p confirm="Continue with trading setup? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

echo.
echo Configuring trading environment...
python src\config\environment_manager.py --setup trading
if errorlevel 1 (
    echo.
    echo ERROR: Failed to configure trading environment
    pause
    goto MENU
)

echo.
echo ========================================
echo   Trading Environment Setup Complete
echo ========================================
echo.
echo Next steps:
echo 1. Import models from your training computer
echo 2. Run setup_environment.bat to install dependencies
echo 3. Use deploy_trading.bat to start paper trading
echo.
echo Configuration file: config.yaml
echo Trading script: deploy_trading.bat
echo Model import script: scripts\import_models.py
echo.
pause
goto MENU

:SHOW_STATUS
echo.
echo ========================================
echo   Current Environment Status
echo ========================================
echo.
python src\config\environment_manager.py --status
echo.
pause
goto MENU

:VALIDATE_CONFIG
echo.
echo ========================================
echo   Configuration Validation
echo ========================================
echo.
python src\config\environment_manager.py --validate
echo.
pause
goto MENU

:GENERATE_REPORT
echo.
echo ========================================
echo   Generate Environment Report
echo ========================================
echo.
set "report_file=environment_report_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.yaml"
set "report_file=%report_file: =0%"

echo Generating environment report...
python src\config\environment_manager.py --report "%report_file%"
if errorlevel 1 (
    echo ERROR: Failed to generate report
) else (
    echo.
    echo Report generated: %report_file%
    echo.
    echo Would you like to view the report?
    set /p view="View report now? (Y/N): "
    if /i "!view!"=="Y" (
        type "%report_file%"
    )
)
echo.
pause
goto MENU

:EXIT
echo.
echo Thank you for using the Trading Bot Environment Manager!
echo.
echo Quick Reference:
echo - Training: Use train_models.bat after setup
echo - Trading: Import models, then use deploy_trading.bat
echo - Help: See DISTRIBUTED_TRAINING_GUIDE.md
echo.
pause
exit /b 0