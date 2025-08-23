@echo off
setlocal enabledelayedexpansion

echo ========================================
echo      Model Validation Script
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

REM Check if models directory exists
if not exist "models" (
    echo ERROR: Models directory not found!
    echo Please import models first using 'import_models.bat'
    pause
    exit /b 1
)

REM Check if models directory has content
dir /b models 2>nul | findstr . >nul
if errorlevel 1 (
    echo ERROR: Models directory is empty!
    echo Please import models first using 'import_models.bat'
    pause
    exit /b 1
)

echo Validating imported models...
echo.

REM Check if validate_models.py exists
if exist "scripts\validate_models.py" (
    echo Running model validation...
    python scripts\validate_models.py --models-dir models --verbose
    
    if errorlevel 1 (
        echo.
        echo WARNING: Model validation found issues!
        echo Check the output above for details.
        echo.
        echo You can still proceed with trading, but some models may not work properly.
        echo.
        set /p "CONTINUE=Continue anyway? (y/N): "
        if /i "!CONTINUE!" neq "y" (
            echo Validation cancelled.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo ========================================
        echo All models validated successfully!
        echo ========================================
    )
else (
    echo validate_models.py not found, performing basic validation...
    echo.
    
    REM Basic validation - check for common model files
    set "MODELS_FOUND=0"
    
    for /d %%d in (models\*) do (
        if exist "%%d" (
            echo Checking %%d...
            
            REM Look for model files
            if exist "%%d\*.pkl" set "MODELS_FOUND=1"
            if exist "%%d\*.pt" set "MODELS_FOUND=1"
            if exist "%%d\*.joblib" set "MODELS_FOUND=1"
            
            REM Look for subdirectories with models
            for /d %%s in ("%%d\*") do (
                if exist "%%s\*.pkl" set "MODELS_FOUND=1"
                if exist "%%s\*.pt" set "MODELS_FOUND=1"
                if exist "%%s\*.joblib" set "MODELS_FOUND=1"
            )
        )
    )
    
    if !MODELS_FOUND!==1 (
        echo.
        echo ========================================
        echo Basic validation passed!
        echo ========================================
        echo Found model files in the models directory.
    ) else (
        echo.
        echo WARNING: No model files found!
        echo Expected file types: *.pkl, *.pt, *.joblib
        echo.
        echo Please check your model import or re-import models.
        pause
        exit /b 1
    )
)

echo.
echo Models are ready for trading!
echo.
echo Next steps:
echo 1. Run 'deploy_trading.bat' to start paper trading
echo 2. Monitor the logs directory for trading activity
echo.
pause