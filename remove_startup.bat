@echo off
rem Remove the Jarvis shortcut from your Startup folder.
set "LNK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Jarvis.lnk"
if exist "%LNK%" (
    del "%LNK%"
    echo Removed: Jarvis will no longer start at login.
) else (
    echo No Jarvis startup shortcut found.
)
pause
