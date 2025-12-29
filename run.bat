@echo off
setlocal

echo [ComiCut AI] Launching...

:: Check venv
if not exist "venv" (
    echo [ERROR] Virtual environment not found. Please run 'install.bat' first.
    pause
    exit /b
)

:: Activate venv
call venv\Scripts\activate

:: Check API Key (Simple check)
findstr "GEMINI_API_KEY=Put_Your" .env > nul
if %errorlevel% equ 0 (
    echo.
    echo [WARNING] It seems GEMINI_API_KEY is not set in .env file.
    echo Please check your .env file.
    echo.
    timeout /t 3 > nul
)

:: Auto-launch Browser (Wait 3 seconds for server to start)
start "" "http://127.0.0.1:8000/ui"

:: Run Server
echo Starting server... Press Ctrl+C to stop.
uvicorn main:app --reload --host 127.0.0.1 --port 8000