@echo off
rem Starts the Jarvis setup window (setup_gui.pyw). If Python is missing, offers to install Python 3.12.
rem Re-open this script minimized so the black window stays out of the way.
if not "%~1"=="min" (
    start "Jarvis setup" /min cmd /c ""%~f0" min"
    exit /b 0
)
cd /d "%~dp0"
set "PY_URL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PY_DIR=%LOCALAPPDATA%\Programs\Python\Python312"

call :find_python
if defined PY goto :launch

rem ---- No Python 3.10+ found: ask to install Python 3.12 (per user, no admin needed) ----
set "JV_MSG=Hey! Python isn't installed on this computer. Jarvis needs it to run.||Install Python 3.12 now? (about 25 MB, takes a minute or two)"
call :ask
if errorlevel 1 (
    set "JV_MSG=Setup cancelled. Jarvis needs Python to run."
    call :tell
    exit /b 1
)

set "INST=%TEMP%\python-3.12.10-amd64.exe"
set "PY_OK="
where curl.exe >nul 2>&1
if not errorlevel 1 (
    curl.exe -L --fail --silent --show-error -o "%INST%" "%PY_URL%"
    if not errorlevel 1 if exist "%INST%" (
        rem /passive shows Python's own progress bar and asks nothing. Everything is per user, no admin.
        start "" /wait "%INST%" /passive InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
        set "PY_OK=1"
    )
    if exist "%INST%" del /q "%INST%"
)
if not exist "%PY_DIR%\python.exe" (
    rem Download or installer failed: try winget if this PC has it.
    where winget >nul 2>&1
    if not errorlevel 1 winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
)

rem PATH isn't refreshed in this window, so look in the install folder directly.
set "PY="
if exist "%PY_DIR%\pythonw.exe" (
    set "PY=%PY_DIR%\pythonw.exe"
) else if exist "%PY_DIR%\python.exe" (
    set "PY=%PY_DIR%\python.exe"
)
if not defined PY call :find_python
if defined PY goto :launch

set "JV_MSG=Python couldn't be installed automatically.||Please install Python 3.12 from https://www.python.org/downloads/ and on the first screen tick 'Add python.exe to PATH'. Then double-click setup.bat again."
call :tell
exit /b 1

:launch
start "" /min "%PY%" "%~dp0setup_gui.pyw"
exit /b 0

rem ---- helpers ----
:find_python
rem setup_gui.pyw --find-pythonw prints pythonw.exe (or python.exe) for that Python, only if it is 3.10 or newer.
set "PY="
for %%C in ("py -3.12" "py -3" "python") do if not defined PY for /f "delims=" %%P in ('%%~C "%~dp0setup_gui.pyw" --find-pythonw 2^>nul') do set "PY=%%P"
exit /b 0

:ask
rem Yes/No box. Text comes from JV_MSG (| = new line). errorlevel 0 = Yes, 1 = No.
powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $r = [System.Windows.Forms.MessageBox]::Show($env:JV_MSG.Replace('|', [string][char]10), 'Jarvis setup', 'YesNo', 'Question'); if ($r -eq 'Yes') { exit 0 } else { exit 1 }"
exit /b %errorlevel%

:tell
rem OK box with the text from JV_MSG. If PowerShell can't show it, print it instead.
powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show($env:JV_MSG.Replace('|', [string][char]10), 'Jarvis setup', 'OK', 'Information')" 2>nul
if errorlevel 1 (
    echo %JV_MSG%
    pause
)
exit /b 0
