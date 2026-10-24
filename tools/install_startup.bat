@echo off
rem Create a "Jarvis" shortcut in your Startup folder so Jarvis starts at login.
rem No admin rights needed.
rem This script lives in tools\; the Jarvis folder is one level up.
for %%I in ("%~dp0..") do set "APP_DIR=%%~fI"
set "SCRIPT_DIR=%APP_DIR%"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LNK=%STARTUP%\Jarvis.lnk"

rem Starts the desktop app (run_jarvis.bat in the Jarvis folder). WindowStyle 7 = minimized.
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); $s.TargetPath = Join-Path $env:SCRIPT_DIR 'run_jarvis.bat'; $s.WorkingDirectory = $env:SCRIPT_DIR; $s.WindowStyle = 7; $s.Description = 'Jarvis voice assistant'; $s.Save()"

if exist "%LNK%" (
    echo Success: Jarvis will start when you log in.
    echo Shortcut: "%LNK%"
) else (
    echo Failed to create the shortcut.
)
pause
