@echo off
setlocal

echo [ComiCut AI] Starting installation...
echo.

:: 1. Check Python
echo [1/4] Checking Python installation...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 'python' command not found.
    echo Please make sure Python 3.10+ is installed and added to PATH.
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b
)
python --version

:: 2. Create Virtual Environment
echo.
echo [2/4] Creating virtual environment (venv)...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo Please check your Python installation and permissions.
        pause
        exit /b
    )
) else (
    echo Virtual environment already exists. Skipping.
)

:: 3. Install Dependencies
echo.
echo [3/4] Installing libraries... (This may take a while)
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [ERROR] venv\Scripts\activate.bat not found. Virtualenv creation failed.
    pause
    exit /b
)

call venv\Scripts\activate.bat

pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip. Continuing...
)

if not exist "requirements.txt" (
    echo.
    echo [ERROR] requirements.txt not found!
    pause
    exit /b
)

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install libraries. Check the error message above.
    pause
    exit /b
)

:: 4. Setup .env
echo.
echo [4/4] Checking configuration...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env > nul
        echo [INFO] '.env' file created from template.
    ) else (
        echo [WARNING] .env.example not found. You must create .env manually.
    )
) else (
    echo [INFO] .env file already exists.
)

echo.
echo ========================================================
echo  Installation Complete!
echo  1. Open '.env' file and add your GEMINI_API_KEY.
echo  2. Run 'run.bat' to start the application.
echo ========================================================
echo.
pause
