@echo off
rem Start Jarvis (the desktop app). No console window; uses .venv made by setup.bat.
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Jarvis is not set up yet. Double-click setup.bat first.
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" jarvis_gui.py
