@echo off
rem Start Jarvis in a minimized console window titled "Jarvis".
rem The window stays open (cmd /k) so errors are visible and the 'p' key works.
cd /d "%~dp0"

rem Use the local virtual environment if there is one, otherwise system python.
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

start "Jarvis" /min /d "%~dp0" cmd /k ""%PY%" jarvis_assistant.py"
