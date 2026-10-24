@echo off
rem This script lives in tools\; the Jarvis folder is one level up.
for %%I in ("%~dp0..") do set "APP_DIR=%%~fI"
cd /d "%APP_DIR%"
echo Updating Jarvis from GitHub...
where git >nul 2>&1
if errorlevel 1 (
  echo Git is not installed. If you downloaded Jarvis as a ZIP, download the new ZIP instead
  echo and copy your .env file into the new folder. See SETUP_GUIDE.md.
  pause
  exit /b 1
)
git pull
if errorlevel 1 (
  echo.
  echo Update failed. Make sure Git is installed and you have no unsaved edits to the code.
  pause
  exit /b 1
)
echo.
echo Installing any new packages...
set "PY=python"
if exist "%APP_DIR%\.venv\Scripts\python.exe" set "PY=%APP_DIR%\.venv\Scripts\python.exe"
"%PY%" -m pip install -r requirements.txt
echo.
echo Jarvis is up to date!
pause
