@echo off
chcp 65001 >nul
echo Запуск установщика...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
pause
