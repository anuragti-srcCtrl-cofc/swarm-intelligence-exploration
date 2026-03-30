@echo off
REM ============================================================
REM  setup.bat  –  Swarm Intelligence Exploration
REM  Sets up a Python virtual environment and installs deps.
REM  Tested on Windows 10/11 with Python 3.9+
REM ============================================================

setlocal EnableDelayedExpansion

echo.
echo  ===========================================
echo   Swarm Intelligence Exploration – Setup
echo  ===========================================
echo.

REM --- Locate Python -------------------------------------------------
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on PATH.
    echo         Download it from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [INFO]  Found Python %PY_VER%

REM --- Create virtual environment ------------------------------------
if exist ".venv" (
    echo [INFO]  Virtual environment already exists at .venv\ — skipping creation.
) else (
    echo [INFO]  Creating virtual environment in .venv\ ...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]    Virtual environment created.
)

REM --- Install dependencies ------------------------------------------
echo [INFO]  Installing dependencies from requirements.txt ...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo [OK]    Dependencies installed.

echo.
echo  ===========================================
echo   Setup complete!
echo  ===========================================
echo.
echo   Activate the environment:
echo     .venv\Scripts\activate
echo.
echo   Then run a simulation, e.g.:
echo     python boids-sim.py
echo     python aco-maze.py
echo     python boids-maze.py
echo     python swarm-maze-fast.py
echo.
pause
