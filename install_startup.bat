@echo off
rem Create a "Jarvis" shortcut in your Startup folder so Jarvis starts at login.
rem No admin rights needed.
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LNK=%STARTUP%\Jarvis.lnk"

rem WindowStyle 7 = minimized.
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); $s.TargetPath = Join-Path $env:SCRIPT_DIR 'start_jarvis.bat'; $s.WorkingDirectory = $env:SCRIPT_DIR; $s.WindowStyle = 7; $s.Description = 'Jarvis voice assistant'; $s.Save()"

if exist "%LNK%" (
    echo Success: Jarvis will start when you log in.
    echo Shortcut: "%LNK%"
) else (
    echo Failed to create the shortcut.
)
pause
