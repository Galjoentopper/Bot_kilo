@echo off
setlocal enabledelayedexpansion

REM Trading Bot Backup Manager
REM User-friendly interface for backup and restore operations

echo ========================================
echo    Trading Bot Backup Manager
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "scripts\backup_system.py" (
    echo Error: backup_system.py not found in scripts directory
    echo Please run this script from the Bot_kilo root directory.
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Create backups directory if it doesn't exist
if not exist "backups" mkdir backups

:main_menu
cls
echo ========================================
echo    Trading Bot Backup Manager
echo ========================================
echo.
echo Current Directory: %CD%
echo Backup Directory: %CD%\backups
echo.
echo Available Actions:
echo.
echo 1. Create Full Backup (All components)
echo 2. Create Models Backup
echo 3. Create Config Backup
echo 4. Create Logs Backup
echo 5. List All Backups
echo 6. Restore Backup
echo 7. Verify Backup Integrity
echo 8. Cleanup Old Backups
echo 9. Show Backup Status
echo 0. Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto create_full_backup
if "%choice%"=="2" goto create_models_backup
if "%choice%"=="3" goto create_config_backup
if "%choice%"=="4" goto create_logs_backup
if "%choice%"=="5" goto list_backups
if "%choice%"=="6" goto restore_backup
if "%choice%"=="7" goto verify_backup
if "%choice%"=="8" goto cleanup_backups
if "%choice%"=="9" goto show_status
if "%choice%"=="0" goto exit

echo Invalid choice. Please try again.
pause
goto main_menu

:create_full_backup
cls
echo ========================================
echo    Creating Full Backup
echo ========================================
echo.
echo This will backup all components:
echo - Models and metadata
echo - Configuration files
echo - Log files
echo - Export packages
echo - Scripts
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto main_menu

echo.
echo Creating full backup...
python scripts\backup_system.py create --type full --verbose
if errorlevel 1 (
    echo.
    echo Backup failed! Check the error messages above.
) else (
    echo.
    echo Full backup created successfully!
)
echo.
pause
goto main_menu

:create_models_backup
cls
echo ========================================
echo    Creating Models Backup
echo ========================================
echo.
echo This will backup:
echo - All trained models (*.pkl, *.pt, *.zip)
echo - Model metadata (*.json)
echo - Feature definitions
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto main_menu

echo.
echo Creating models backup...
python scripts\backup_system.py create --type models --verbose
if errorlevel 1 (
    echo.
    echo Backup failed! Check the error messages above.
) else (
    echo.
    echo Models backup created successfully!
)
echo.
pause
goto main_menu

:create_config_backup
cls
echo ========================================
echo    Creating Config Backup
echo ========================================
echo.
echo This will backup:
echo - Configuration files (*.yaml, *.yml)
echo - Environment settings
echo - Trading parameters
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto main_menu

echo.
echo Creating config backup...
python scripts\backup_system.py create --type config --verbose
if errorlevel 1 (
    echo.
    echo Backup failed! Check the error messages above.
) else (
    echo.
    echo Config backup created successfully!
)
echo.
pause
goto main_menu

:create_logs_backup
cls
echo ========================================
echo    Creating Logs Backup
echo ========================================
echo.
echo This will backup:
echo - Trading logs
echo - Training logs
echo - Error logs (last 30 days)
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto main_menu

echo.
echo Creating logs backup...
python scripts\backup_system.py create --type logs --verbose
if errorlevel 1 (
    echo.
    echo Backup failed! Check the error messages above.
) else (
    echo.
    echo Logs backup created successfully!
)
echo.
pause
goto main_menu

:list_backups
cls
echo ========================================
echo    Available Backups
echo ========================================
echo.
python scripts\backup_system.py list
echo.
pause
goto main_menu

:restore_backup
cls
echo ========================================
echo    Restore Backup
echo ========================================
echo.
echo Available backups:
python scripts\backup_system.py list
echo.
set /p backup_name="Enter backup name to restore: "
if "%backup_name%"=="" (
    echo No backup name provided.
    pause
    goto main_menu
)

echo.
echo WARNING: This will overwrite existing files!
echo Backup to restore: %backup_name%
echo.
set /p confirm="Are you sure you want to continue? (y/n): "
if /i not "%confirm%"=="y" goto main_menu

echo.
echo Select components to restore:
echo 1. All components
echo 2. Models only
echo 3. Config only
echo 4. Logs only
echo 5. Custom selection
echo.
set /p restore_choice="Enter choice (1-5): "

if "%restore_choice%"=="1" (
    echo Restoring all components...
    python scripts\backup_system.py restore --name "%backup_name%" --verbose
) else if "%restore_choice%"=="2" (
    echo Restoring models only...
    python scripts\backup_system.py restore --name "%backup_name%" --components models --verbose
) else if "%restore_choice%"=="3" (
    echo Restoring config only...
    python scripts\backup_system.py restore --name "%backup_name%" --components config --verbose
) else if "%restore_choice%"=="4" (
    echo Restoring logs only...
    python scripts\backup_system.py restore --name "%backup_name%" --components logs --verbose
) else if "%restore_choice%"=="5" (
    echo Available components: models, config, logs, exports, scripts
    set /p components="Enter components (space-separated): "
    echo Restoring selected components...
    python scripts\backup_system.py restore --name "%backup_name%" --components !components! --verbose
) else (
    echo Invalid choice.
    pause
    goto main_menu
)

if errorlevel 1 (
    echo.
    echo Restore failed! Check the error messages above.
) else (
    echo.
    echo Backup restored successfully!
)
echo.
pause
goto main_menu

:verify_backup
cls
echo ========================================
echo    Verify Backup Integrity
echo ========================================
echo.
echo Available backups:
python scripts\backup_system.py list
echo.
set /p backup_name="Enter backup name to verify: "
if "%backup_name%"=="" (
    echo No backup name provided.
    pause
    goto main_menu
)

echo.
echo Verifying backup: %backup_name%
python scripts\backup_system.py verify --name "%backup_name%" --verbose
if errorlevel 1 (
    echo.
    echo Backup verification failed!
) else (
    echo.
    echo Backup verification successful!
)
echo.
pause
goto main_menu

:cleanup_backups
cls
echo ========================================
echo    Cleanup Old Backups
echo ========================================
echo.
echo This will remove old backups according to retention policy:
echo - Keep 7 daily backups
echo - Keep 4 weekly backups
echo - Keep 12 monthly backups
echo.
set /p confirm="Continue with cleanup? (y/n): "
if /i not "%confirm%"=="y" goto main_menu

echo.
echo Cleaning up old backups...
python scripts\backup_system.py cleanup --verbose
if errorlevel 1 (
    echo.
    echo Cleanup failed! Check the error messages above.
) else (
    echo.
    echo Cleanup completed successfully!
)
echo.
pause
goto main_menu

:show_status
cls
echo ========================================
echo    Backup System Status
echo ========================================
echo.
echo Current Directory: %CD%
echo Backup Directory: %CD%\backups
echo.

REM Check backup directory size
if exist "backups" (
    echo Backup Directory Status: EXISTS
    for /f "tokens=3" %%a in ('dir "backups" /-c ^| find "File(s)"') do set backup_size=%%a
    echo Backup Directory Size: !backup_size! bytes
) else (
    echo Backup Directory Status: NOT FOUND
)

echo.
echo Recent Backups:
python scripts\backup_system.py list 2>nul | head -n 10

echo.
echo Disk Space:
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do echo Free Space: %%a bytes

echo.
echo Python Version:
python --version

echo.
echo Backup System Script: 
if exist "scripts\backup_system.py" (
    echo   Status: FOUND
    for %%a in ("scripts\backup_system.py") do echo   Size: %%~za bytes
    for %%a in ("scripts\backup_system.py") do echo   Modified: %%~ta
) else (
    echo   Status: NOT FOUND
)

echo.
pause
goto main_menu

:exit
echo.
echo Thank you for using Trading Bot Backup Manager!
echo.
echo Next Steps:
echo - Set up scheduled backups using Task Scheduler
echo - Consider cloud storage integration for off-site backups
echo - Regularly verify backup integrity
echo - Keep backups in multiple locations
echo.
pause
exit /b 0