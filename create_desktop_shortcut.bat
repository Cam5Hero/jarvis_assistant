@echo off
rem Puts a "Jarvis" shortcut on your Desktop that opens the Jarvis app (no black console window).
rem Safe to run again: it just replaces the shortcut. No admin rights, nothing is downloaded.
cd /d "%~dp0"
set "JV_DIR=%~dp0"
set "JV_DIR=%JV_DIR:~0,-1%"

rem Find pythonw.exe (Python without a console window): this folder's .venv first,
rem then the one next to python.exe, then anything "where pythonw" finds.
set "JV_PYW="
if exist "%JV_DIR%\.venv\Scripts\pythonw.exe" set "JV_PYW=%JV_DIR%\.venv\Scripts\pythonw.exe"
if not defined JV_PYW for /f "delims=" %%P in ('python -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))" 2^>nul') do if exist "%%P" set "JV_PYW=%%P"
if not defined JV_PYW for /f "delims=" %%P in ('where pythonw 2^>nul') do if not defined JV_PYW set "JV_PYW=%%P"
if not defined JV_PYW (
    echo Could not find pythonw.exe. Install Python 3.12 from python.org first, then run this again.
    pause
    exit /b 1
)

rem Ask Windows (WScript.Shell) to make the shortcut. GetFolderPath('Desktop') also finds a OneDrive Desktop.
powershell -NoProfile -Command "$lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Jarvis.lnk'; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk); $s.TargetPath = $env:JV_PYW; $s.Arguments = [char]34 + (Join-Path $env:JV_DIR 'jarvis_gui.py') + [char]34; $s.WorkingDirectory = $env:JV_DIR; $ico = Join-Path $env:JV_DIR 'jarvis.ico'; if (Test-Path $ico) { $s.IconLocation = $ico }; $s.Description = 'Jarvis voice assistant'; $s.Save(); Write-Host ('Created ' + $lnk)"
if errorlevel 1 (
    echo Something went wrong creating the shortcut.
) else (
    echo Done. Double-click "Jarvis" on your Desktop to start it.
)
pause
