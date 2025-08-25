@echo off
setlocal enabledelayedexpansion

echo ========================================
echo    Enhanced Model Import Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "scripts" (
    echo ERROR: Please run this script from the Bot_kilo root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Checking for transfer packages...
echo.

REM Look for transfer packages (or use the provided argument)
set "PACKAGE_FOUND=0"
set "TRANSFER_PACKAGE="

REM If an argument is provided, and it exists, use it directly
if not "%~1"=="" (
    if exist "%~1" (
        set "TRANSFER_PACKAGE=%~1"
        set "PACKAGE_FOUND=1"
        echo Using transfer package (from argument): %~1
    )
)

REM If not provided or not found, search current directory
if "!PACKAGE_FOUND!"=="0" (
    for %%f in (*.zip) do (
        echo Found transfer package: %%f
        set "PACKAGE_FOUND=1"
        set "TRANSFER_PACKAGE=%%f"
        goto :afterSearch
    )
)

:afterSearch

REM Handle missing/invalid argument or no zip found
if "!PACKAGE_FOUND!"=="0" (
    if not "%~1"=="" (
        echo ERROR: Provided package not found: %~1
        pause
        exit /b 1
    ) else (
        echo No transfer packages found in current directory.
        echo.
        echo Please copy your transfer package (*.zip) to this directory and run again.
        echo The transfer package should be created on your Linux training computer using:
        echo   ./train_models.sh
        echo.
        echo Or manually create a package using:
        echo   python scripts/cross_platform_transfer.py create --source ./models --output transfer_package.zip
        pause
        exit /b 1
    )
)

echo Using transfer package: !TRANSFER_PACKAGE!
echo.

REM Validate the transfer package first
echo Validating transfer package...
python scripts\cross_platform_transfer.py validate --package "!TRANSFER_PACKAGE!"

if errorlevel 1 (
    echo.
    echo ERROR: Transfer package validation failed!
    echo Please check the package and try again.
    pause
    exit /b 1
)

echo Package validation passed!
echo.

REM Create models directory if it doesn't exist
if not exist "models" mkdir models

echo Importing models using cross-platform transfer...
python scripts\cross_platform_transfer.py import --package "!TRANSFER_PACKAGE!" --destination models

if errorlevel 1 (
    echo.
    echo ERROR: Model import failed!
    pause
    exit /b 1
)

REM Clean up - move the transfer package to processed folder
if not exist "processed_packages" mkdir processed_packages
move "!TRANSFER_PACKAGE!" "processed_packages\!TRANSFER_PACKAGE!"

echo.
echo ========================================
echo Models imported successfully!
echo ========================================
echo.
echo Transfer package moved to: processed_packages\!TRANSFER_PACKAGE!
echo.
echo Next steps:
echo 1. Run 'deploy_trading.bat' to start paper trading
echo 2. Check the logs directory for any issues
echo 3. Verify models are working with 'validate_models.bat'
echo.
pause