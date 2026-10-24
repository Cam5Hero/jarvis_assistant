@echo off
rem Start Jarvis in a minimized console window titled "Jarvis".
rem The window stays open (cmd /k) so errors are visible and the 'p' key works.
rem This script lives in tools\; the Jarvis folder is one level up.
for %%I in ("%~dp0..") do set "APP_DIR=%%~fI"
cd /d "%APP_DIR%"

rem Use the local virtual environment if there is one, otherwise system python.
set "PY=python"
if exist "%APP_DIR%\.venv\Scripts\python.exe" set "PY=%APP_DIR%\.venv\Scripts\python.exe"

start "Jarvis" /min /d "%APP_DIR%" cmd /k ""%PY%" jarvis_assistant.py"
