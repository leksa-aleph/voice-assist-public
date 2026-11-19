@echo off
setlocal

cd /d "%~dp0"

set "PYTHON="
for %%P in (
    "%~dp0venv\Scripts\python.exe"
) do if not defined PYTHON if exist %%P set "PYTHON=%%~P"

if not defined PYTHON (
    echo Python environment not found. See README.
    pause
    exit /b 1
)

echo Using: %PYTHON%
"%PYTHON%" app.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Application exited with error code %EXIT_CODE%.
    echo Log:
    echo %APPDATA%\voice-transcriber\voice-transcriber.log
    pause
)

exit /b %EXIT_CODE%