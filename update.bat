@echo off
cd /d "%~dp0"

echo ===================================================
echo       ComiCut AI - Auto Updater
echo ===================================================
echo.

:: 1. Git Check
where git >nul 2>nul
if %errorlevel% equ 0 goto check_git_repo

echo [ERROR] Git is not installed.
echo Please install Git first: https://git-scm.com/download/win
goto end_error

:check_git_repo
:: 2. Repo Check
if exist ".git" goto pull_code

echo [ERROR] This directory is not a Git repository.
echo You can only use this updater if you installed via 'git clone'.
goto end_error

:pull_code
echo [1/3] Pulling latest code...
git pull origin main
if %errorlevel% neq 0 goto pull_error
goto install_deps

:pull_error
echo.
echo [ERROR] Update failed (Conflict or Network issue).
echo Please try re-downloading or fixing conflicts.
goto end_error

:install_deps
echo.
echo [2/3] Checking dependencies...
if exist "venv" goto run_pip

echo [ERROR] 'venv' folder not found. Run install.bat first.
goto end_error

:run_pip
call venv\Scripts\activate
pip install -r requirements.txt
if %errorlevel% neq 0 echo [WARNING] Some dependencies failed to install.

echo.
echo [3/3] Update Complete!
echo You can now start ComiCut by running 'run.bat'.
goto end_success

:end_error
echo.
echo [UPDATE FAILED] Please check the error message above.
pause
exit /b

:end_success
echo.
echo ===================================================
echo press any key to close.
pause >nul
exit /b