@echo off
rem Start the Jarvis desktop app (window + tray icon) with no console window.
rem To see error messages instead, open a terminal here and run: python jarvis_gui.py
cd /d "%~dp0"

rem Use the local virtual environment if there is one, otherwise system Python.
rem pythonw.exe is Python without a console window.
set "PYW=pythonw"
if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYW=%~dp0.venv\Scripts\pythonw.exe"

start "" "%PYW%" jarvis_gui.py
